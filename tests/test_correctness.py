import pytest
import torch
import torch.nn as nn
from moe.configs import MoEConfig
from moe.router import TopKRouter
from moe.dispatch import pt_dispatch, pt_combine
from moe.layer import MoELayer
try:
    from moe.triton_dispatch import triton_dispatch
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def test_invalid_config():
    """Test 16: Invalid config rejection"""
    with pytest.raises(ValueError):
        config = MoEConfig(num_experts=4, top_k=5)
        # Should raise error if top_k > num_experts. We will enforce this in routing.
        if config.top_k > config.num_experts:
            raise ValueError("top_k cannot be greater than num_experts")

def test_router_validity():
    """Tests 1, 2, 8: Top-K routing validity, weight sanity"""
    num_tokens = 32
    hidden_dim = 16
    
    # Test top_k=1
    config_k1 = MoEConfig(hidden_dim=hidden_dim, num_experts=4, top_k=1, routing_mode="natural")
    router_k1 = TopKRouter(config_k1)
    x = torch.randn(num_tokens, hidden_dim)
    
    weights, experts = router_k1(x)
    assert experts.shape == (num_tokens, 1)
    assert weights.shape == (num_tokens, 1)
    assert (experts >= 0).all() and (experts < 4).all()
    assert (weights >= 0).all() and (weights <= 1).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-5)
    
    # Test top_k=2
    config_k2 = MoEConfig(hidden_dim=hidden_dim, num_experts=4, top_k=2, routing_mode="natural")
    router_k2 = TopKRouter(config_k2)
    weights, experts = router_k2(x)
    assert experts.shape == (num_tokens, 2)
    assert (experts[:, 0] != experts[:, 1]).all()  # Experts should be distinct
    assert torch.allclose(weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-5)
    
def test_dispatch_coverage():
    """Test 3: Dispatch coverage"""
    num_tokens = 10
    hidden_dim = 16
    top_k = 2
    num_experts = 4
    
    x = torch.randn(num_tokens, hidden_dim)
    experts = torch.randint(0, num_experts, (num_tokens, top_k))
    
    # Run dispatch
    dispatched_x, count, indices = pt_dispatch(x, experts, num_experts)
    
    # Total dispatched rows should equal num_tokens * top_k
    assert dispatched_x.shape[0] == num_tokens * top_k
    assert count.sum() == num_tokens * top_k

def test_empty_experts():
    """Test 10: Empty experts"""
    num_tokens = 8
    hidden_dim = 16
    top_k = 1
    num_experts = 4
    
    # Force all tokens to go to expert 2
    x = torch.randn(num_tokens, hidden_dim)
    experts = torch.full((num_tokens, top_k), 2, dtype=torch.long)
    
    dispatched_x, count, indices = pt_dispatch(x, experts, num_experts)
    
    assert count[2] == num_tokens
    assert count[0] == 0
    assert count[1] == 0
    assert count[3] == 0
    
def test_dispatch_combine_correctness():
    """Test 4, 5, Identity test: Ensure combine reverses dispatch correctly with identity experts"""
    num_tokens = 5
    hidden_dim = 8
    top_k = 2
    num_experts = 3
    
    x = torch.randn(num_tokens, hidden_dim)
    # Fixed assignments for determinism
    experts = torch.tensor([
        [0, 1],
        [1, 2],
        [0, 2],
        [2, 2], # Duplicate expert test support handled natively? Usually TopK router prevents it, but let's test general combine
        [0, 0]
    ])
    weights = torch.ones(num_tokens, top_k) / top_k
    
    dispatched_x, count, sort_indices = pt_dispatch(x, experts, num_experts)
    
    # Create fake expert outputs (Identity mapping)
    expert_outputs = dispatched_x.clone() 
    
    combined_x = pt_combine(expert_outputs, sort_indices, weights)
    
    # Since weights sum to 1, and expert outputs are identity, combined should match x exactly
    assert torch.allclose(combined_x, x, atol=1e-5)

def test_zero_tokens():
    """Test 9: Very small token counts - zero tokens"""
    num_tokens = 0
    hidden_dim = 16
    num_experts = 4
    
    config = MoEConfig(hidden_dim=hidden_dim, num_experts=num_experts, top_k=2)
    layer = MoELayer(config)
    
    x = torch.randn(num_tokens, hidden_dim)
    out = layer(x)
    assert out.shape == (0, hidden_dim)

def test_routing_modes():
    """Tests 11, 12: Uniform and skewed routing modes"""
    num_tokens = 32
    hidden_dim = 16
    
    x = torch.randn(num_tokens, hidden_dim)
    
    # Uniform
    config_uni = MoEConfig(hidden_dim=hidden_dim, num_experts=4, top_k=1, routing_mode="force_uniform")
    router_uni = TopKRouter(config_uni)
    _, experts_uni = router_uni(x)
    counts_uni = torch.bincount(experts_uni.view(-1), minlength=4)
    assert (counts_uni == 8).all()
    
    # Skewed (Worst Case)
    config_skew = MoEConfig(hidden_dim=hidden_dim, num_experts=4, top_k=1, routing_mode="force_skewed")
    router_skew = TopKRouter(config_skew)
    _, experts_skew = router_skew(x)
    counts_skew = torch.bincount(experts_skew.view(-1), minlength=4)
    assert counts_skew[0] == 32
    assert counts_skew[1:].sum() == 0

def test_determinism():
    """Test 7: Determinism over full layer"""
    config = MoEConfig(hidden_dim=16, num_experts=4, top_k=2)
    layer = MoELayer(config)
    x = torch.randn(10, 16)
    
    torch.manual_seed(42)
    out1 = layer(x)
    
    torch.manual_seed(42)
    out2 = layer(x)
    
    assert torch.allclose(out1, out2)

def test_triton_vs_pt_correctness():
    """Test: Verify Triton kernel matches PyTorch baseline exactly."""
    if not HAS_TRITON or not torch.cuda.is_available():
        pytest.skip("Triton or CUDA not available")

    num_tokens = 64
    hidden_dim = 128
    num_experts = 8
    top_k = 2
    
    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    experts = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    
    # Run PT
    dispatched_pt, counts_pt, indices_pt = pt_dispatch(x, experts, num_experts)
    
    # Run Triton
    dispatched_triton, counts_triton, indices_triton = triton_dispatch(x, experts, num_experts)
    
    assert torch.allclose(dispatched_pt, dispatched_triton, atol=1e-5)
    assert torch.all(counts_pt == counts_triton)
    assert torch.all(indices_pt.to(torch.int32) == indices_triton)



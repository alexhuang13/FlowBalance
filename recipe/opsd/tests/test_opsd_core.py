import torch

from recipe.opsd.opsd_core_algos import clip_pointwise_divergence


def test_pointwise_clip_caps_each_token():
    values = torch.tensor([[0.01, 0.06, 2.0]])
    clipped, fraction = clip_pointwise_divergence(values, 0.06)
    assert torch.equal(clipped, torch.tensor([[0.01, 0.06, 0.06]]))
    assert fraction == 2 / 3


def test_clip_disabled_is_identity():
    values = torch.rand(2, 3)
    clipped, fraction = clip_pointwise_divergence(values, None)
    assert clipped is values
    assert fraction == 0.0


def test_nonpositive_clip_rejected():
    try:
        clip_pointwise_divergence(torch.ones(1), 0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")

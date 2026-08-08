from types import SimpleNamespace

import pytest

from core.workers.reward_manager.custom_dapo import compute_overlong_reward


def _cfg(enable=True, length=2048, factor=1.0):
    return SimpleNamespace(enable=enable, len=length, penalty_factor=factor)


def test_overlong_penalty_is_zero_before_buffer():
    assert compute_overlong_reward(6144, 8192, _cfg()) == 0.0
    assert compute_overlong_reward(4096, 8192, _cfg()) == 0.0


def test_overlong_penalty_grows_linearly_to_hard_limit():
    assert compute_overlong_reward(7168, 8192, _cfg()) == -0.5
    assert compute_overlong_reward(8192, 8192, _cfg()) == -1.0


def test_overlong_penalty_can_be_disabled():
    assert compute_overlong_reward(8192, 8192, _cfg(enable=False)) == 0.0


def test_overlong_penalty_rejects_empty_buffer():
    with pytest.raises(ValueError, match="must be > 0"):
        compute_overlong_reward(8192, 8192, _cfg(length=0))

from types import SimpleNamespace

import numpy as np
import torch

from recipe.flowsd.flowsd_core_algos import compute_flowsd_target, resolve_flowsd_coefficients


def _cfg(**kwargs):
    defaults = dict(
        beta_q=0.0,
        eta_R=15.0,
        beta_q_start=None,
        beta_q_end=None,
        eta_R_start=None,
        eta_R_end=None,
        schedule_steps=0,
        clip_B=4.0,
        rho=1.0,
        gate_no_context="drop",
        min_group_valid=8,
        reference_source="frozen_ref",
        reward_type="grpo_advantage",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_beta_zero_uses_all_rollouts_and_grpo_advantage():
    response_mask = torch.ones(4, 2)
    ref_log_prob = torch.tensor([[-2.0, -2.0]] * 4)
    old_log_prob = torch.tensor(
        [
            [-1.8, -1.8],
            [-2.2, -2.2],
            [-1.5, -1.5],
            [-2.5, -2.5],
        ]
    )
    advantages = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, -1.0],
            [0.5, 0.5],
            [-0.5, -0.5],
        ]
    )

    target, mask, metrics = compute_flowsd_target(
        teacher_log_prob=None,
        ref_log_prob=ref_log_prob,
        old_log_probs=old_log_prob,
        response_mask=response_mask,
        reward_signal=advantages,
        advantage_signal=advantages,
        self_distillation_mask=torch.zeros(4),
        uids=np.array(["a", "a", "b", "b"], dtype=object),
        flowsd_config=_cfg(),
    )

    # beta_q=0 ignores privileged-context availability and min_group_valid=8.
    torch.testing.assert_close(mask, torch.ones(4))
    assert metrics["flowsd/beta_zero_flowrl_path"] == 1.0
    assert metrics["flowsd/reward_is_grpo_advantage"] == 1.0

    # Target uses mean-token GRPO advantage with eta_R=15 and a uid baseline.
    ref_seq = ref_log_prob.mean(dim=-1)
    old_seq = old_log_prob.mean(dim=-1)
    reward = advantages.mean(dim=-1)
    log_f = ref_seq + 15.0 * reward
    expected = log_f.clone()
    for indices in ([0, 1], [2, 3]):
        expected[indices] += (old_seq[indices] - log_f[indices]).mean()
    torch.testing.assert_close(target, expected)


def test_beta_zero_schedule_is_resolved_for_fast_path():
    cfg = _cfg(beta_q=1.0, beta_q_start=0.0, beta_q_end=1.0, schedule_steps=10)
    beta_q, eta_R, beta_progress, _ = resolve_flowsd_coefficients(cfg, global_step=0)
    assert beta_q == 0.0
    assert eta_R == 15.0
    assert beta_progress == 0.0


def test_positive_beta_keeps_privileged_gate():
    target, mask, metrics = compute_flowsd_target(
        teacher_log_prob=torch.zeros(2, 2),
        ref_log_prob=torch.zeros(2, 2),
        old_log_probs=torch.zeros(2, 2),
        response_mask=torch.ones(2, 2),
        reward_signal=torch.zeros(2, 2),
        advantage_signal=torch.zeros(2, 2),
        self_distillation_mask=torch.tensor([1.0, 0.0]),
        uids=np.array(["a", "a"], dtype=object),
        flowsd_config=_cfg(beta_q=1.0, min_group_valid=1),
    )
    del target
    torch.testing.assert_close(mask, torch.tensor([1.0, 0.0]))
    assert metrics["flowsd/beta_zero_flowrl_path"] == 0.0


def test_positive_beta_multiplies_gain_by_grpo_advantage_sign():
    response_mask = torch.ones(3, 2)
    ref_log_prob = torch.zeros(3, 2)
    teacher_log_prob = torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])

    _, _, metrics = compute_flowsd_target(
        teacher_log_prob=teacher_log_prob,
        ref_log_prob=ref_log_prob,
        old_log_probs=torch.zeros(3, 2),
        response_mask=response_mask,
        reward_signal=advantages,
        advantage_signal=advantages,
        self_distillation_mask=torch.ones(3),
        uids=np.array(["a", "a", "a"], dtype=object),
        flowsd_config=_cfg(beta_q=1.0, min_group_valid=1),
    )

    # Raw gain is +2 for every sample; signed gain becomes [+2, -2, 0].
    assert metrics["flowsd/G_q_raw_mean"] == 2.0
    assert metrics["flowsd/G_q_mean"] == 0.0
    assert metrics["flowsd/advantage_zero_fraction"] == 1.0 / 3.0

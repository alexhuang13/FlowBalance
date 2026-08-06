"""RLSD trainer hook on the stable_rl PPO trainer."""
from typing import Any

from verl import DataProto
from recipe.sdpo.sdpo_ray_trainer import SDPORayPPOTrainer
from recipe.rlsd.rlsd_core_algos import resolve_lambda


class RLSDRayPPOTrainer(SDPORayPPOTrainer):
    def _update_actor(self, batch: DataProto) -> DataProto:
        cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        metrics: dict[str, float] = {}
        if cfg is not None and loss_mode == "rlsd":
            sd_data = self._maybe_build_self_distillation_batch(
                batch,
                batch.batch["token_level_scores"],
                self._reconstruct_reward_extra_infos(batch, cfg),
            )
            if sd_data is not None:
                sd_batch, metrics = sd_data
                batch = batch.union(sd_batch)
            rlsd_cfg = self.config.actor_rollout_ref.actor.rlsd
            effective = resolve_lambda(
                float(rlsd_cfg.lambda_),
                int(self.global_steps),
                int(rlsd_cfg.lambda_warmup_steps),
                int(rlsd_cfg.lambda_decay_steps),
            )
            batch.meta_info["rlsd_lambda"] = effective
            batch.meta_info["global_step"] = int(self.global_steps)
            metrics["rlsd/effective_lambda"] = effective

        # Skip SDPORayPPOTrainer._update_actor to avoid a second reprompt build.
        actor_output = super(SDPORayPPOTrainer, self)._update_actor(batch)
        if metrics:
            metric_dict = actor_output.meta_info.setdefault("metrics", {})
            for key, value in metrics.items():
                metric_dict[key] = [value]
        return actor_output

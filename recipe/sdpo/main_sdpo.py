# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SDPO Hydra entry point (overlay).

Subclasses stable_rl's ``CustomTaskRunner`` to:
  * register the SDPO actor-rollout(-ref) worker, and
  * redirect trainer construction to ``SDPORayPPOTrainer``.

GRPO baseline and SDPO share this single entry point; they differ only by the
``actor_rollout_ref.actor.policy_loss.loss_mode`` config (``vanilla`` vs ``sdpo``).

Run with::

    python3 -m recipe.sdpo.main_sdpo <hydra overrides>
"""

import os


def _prepend_env_path(key: str, value: str) -> None:
    current = os.environ.get(key)
    if current:
        parts = current.split(":")
        if value not in parts:
            os.environ[key] = f"{value}:{current}"
    else:
        os.environ[key] = value


# Force safe distributed/JIT defaults before any distributed, vLLM or Triton
# initialization happens in the Ray driver or child actors.
toolchain_dir = os.environ.get(
    "SDPO_TOOLCHAIN_DIR",
    "/apdcephfs_gy4/share_303378103/user/audenhuang/FlowSD/recipe/sdpo/toolchain",
)
os.environ["PATH"] = f"{toolchain_dir}:{os.environ.get('PATH', '')}"
os.environ["NCCL_ALGO"] = "Ring"
os.environ["VLLM_USE_V1"] = "1"
os.environ["CC"] = f"{toolchain_dir}/gcc"
os.environ["CXX"] = f"{toolchain_dir}/g++"
os.environ["TRITON_CC"] = f"{toolchain_dir}/gcc"
os.environ["CUDAHOSTCXX"] = f"{toolchain_dir}/g++"
os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0")
_prepend_env_path("LIBRARY_PATH", "/usr/lib/gcc/x86_64-TencentOS-linux/12")
_prepend_env_path("LIBRARY_PATH", "/usr/lib64")
_prepend_env_path("LD_LIBRARY_PATH", "/usr/lib/gcc/x86_64-TencentOS-linux/12")
_prepend_env_path("LD_LIBRARY_PATH", "/usr/lib64")
_prepend_env_path("COMPILER_PATH", "/usr/libexec/gcc/x86_64-TencentOS-linux/12")
os.environ.setdefault("LDFLAGS", "-L/usr/lib/gcc/x86_64-TencentOS-linux/12 -L/usr/lib64")

import hydra
import ray

from core.trainer.main_ppo import CustomTaskRunner, run_ppo
from verl.utils.device import auto_set_device


class SDPOTaskRunner(CustomTaskRunner):
    """TaskRunner that wires the SDPO worker + trainer into stable_rl's pipeline."""

    def add_actor_rollout_worker(self, config):
        """Register the SDPO actor-rollout worker.

        For ``loss_mode == "sdpo"`` we register a hybrid ``ActorRolloutRef`` worker so
        that the teacher (reference module) is co-located with the actor and can be
        forwarded / EMA-updated inside ``update_policy``. For the GRPO baseline
        (``vanilla``) we fall back to the standard ``ActorRollout`` role.
        """
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from recipe.sdpo.sdpo_fsdp_workers import SDPOAsyncActorRolloutRefWorker

        strategy = config.actor_rollout_ref.actor.strategy
        if strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError(f"SDPO overlay only supports fsdp/fsdp2, got strategy={strategy}")

        actor_rollout_cls = SDPOAsyncActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        role = Role.ActorRolloutRef if loss_mode in {"sdpo", "opsd"} else Role.ActorRollout

        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """For SDPO the reference/teacher is fused into the hybrid worker, so we never
        spawn a separate ref worker. For GRPO defer to the parent logic."""
        loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode in {"sdpo", "opsd"}:
            return
        super().add_ref_policy_worker(config, ref_policy_cls)

    def run(self, config):
        # Overlay the trainer class used by the inherited run(): the parent method
        # resolves ``CustomRayPPOTrainer`` from the ``core.trainer.main_ppo`` module
        # globals at call time, so swapping it here redirects trainer construction to
        # ``SDPORayPPOTrainer`` without copying the (large) run() body.
        import core.trainer.main_ppo as core_main

        from recipe.sdpo.sdpo_ray_trainer import SDPORayPPOTrainer

        core_main.CustomRayPPOTrainer = SDPORayPPOTrainer
        return super().run(config)


@hydra.main(config_path="config", config_name="sdpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SDPOTaskRunner))


if __name__ == "__main__":
    main()

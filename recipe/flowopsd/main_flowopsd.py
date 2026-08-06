# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FlowOPSD Hydra entry point."""

from __future__ import annotations

import os


def _prepend_env_path(key: str, value: str) -> None:
    current = os.environ.get(key)
    if current:
        parts = current.split(":")
        if value not in parts:
            os.environ[key] = f"{value}:{current}"
    else:
        os.environ[key] = value


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


class FlowOPSDTaskRunner(CustomTaskRunner):
    """TaskRunner that wires FlowOPSD worker and trainer into FlowSD."""

    @staticmethod
    def _uses_ref(config) -> bool:
        loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        return loss_mode in {"sdpo", "flowopsd"}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from recipe.flowopsd.flowopsd_fsdp_workers import FlowOPSDAsyncActorRolloutRefWorker

        strategy = config.actor_rollout_ref.actor.strategy
        if strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError(f"FlowOPSD only supports fsdp/fsdp2, got strategy={strategy}")
        role = Role.ActorRolloutRef if self._uses_ref(config) else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(FlowOPSDAsyncActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        return FlowOPSDAsyncActorRolloutRefWorker, RayWorkerGroup

    def add_ref_policy_worker(self, config, ref_policy_cls):
        if self._uses_ref(config):
            return
        super().add_ref_policy_worker(config, ref_policy_cls)

    def run(self, config):
        import core.trainer.main_ppo as core_main

        from recipe.flowopsd.flowopsd_ray_trainer import FlowOPSDRayPPOTrainer

        core_main.CustomRayPPOTrainer = FlowOPSDRayPPOTrainer
        return super().run(config)


@hydra.main(config_path="config", config_name="flowopsd_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(FlowOPSDTaskRunner))


if __name__ == "__main__":
    main()

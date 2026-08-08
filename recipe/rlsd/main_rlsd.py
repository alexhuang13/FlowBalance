"""RLSD Hydra entrypoint on the unified FlowSD/verl."""
from __future__ import annotations
import os


def _prepend(key: str, value: str):
    current = os.environ.get(key, "")
    if value not in current.split(":"):
        os.environ[key] = f"{value}:{current}" if current else value


toolchain = os.environ.get("SDPO_TOOLCHAIN_DIR", "/apdcephfs_gy4/share_303378103/user/audenhuang/FlowSD/recipe/sdpo/toolchain")
os.environ["PATH"] = f"{toolchain}:{os.environ.get('PATH','')}"
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["CC"] = f"{toolchain}/gcc"
os.environ["CXX"] = f"{toolchain}/g++"
os.environ["TRITON_CC"] = f"{toolchain}/gcc"
os.environ["CUDAHOSTCXX"] = f"{toolchain}/g++"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
_prepend("LIBRARY_PATH", "/usr/lib/gcc/x86_64-TencentOS-linux/12")
_prepend("LIBRARY_PATH", "/usr/lib64")
_prepend("LD_LIBRARY_PATH", "/usr/lib/gcc/x86_64-TencentOS-linux/12")
_prepend("LD_LIBRARY_PATH", "/usr/lib64")
_prepend("COMPILER_PATH", "/usr/libexec/gcc/x86_64-TencentOS-linux/12")
os.environ.setdefault("LDFLAGS", "-L/usr/lib/gcc/x86_64-TencentOS-linux/12 -L/usr/lib64")

import hydra
import ray
from core.trainer.main_ppo import CustomTaskRunner, run_ppo
from verl.utils.device import auto_set_device


class RLSDTaskRunner(CustomTaskRunner):
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from recipe.rlsd.rlsd_fsdp_workers import RLSDAsyncActorRolloutRefWorker
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(RLSDAsyncActorRolloutRefWorker)
        self.mapping[Role.ActorRolloutRef] = "global_pool"
        return RLSDAsyncActorRolloutRefWorker, RayWorkerGroup

    def add_ref_policy_worker(self, config, ref_policy_cls):
        return

    def run(self, config):
        import core.trainer.main_ppo as core_main
        from recipe.rlsd.rlsd_ray_trainer import RLSDRayPPOTrainer
        core_main.CustomRayPPOTrainer = RLSDRayPPOTrainer
        return super().run(config)


@hydra.main(config_path="config", config_name="rlsd_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(RLSDTaskRunner))


if __name__ == "__main__":
    main()

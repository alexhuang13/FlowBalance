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
"""SDPO FSDP workers (overlay).

Re-uses FlowSD's ``CustomAsyncActorRolloutRefWorker.init_model`` to build the
actor / rollout / (fused) reference modules, then:
  * re-wraps ``self.actor`` as :class:`SDPODataParallelPPOActor`, and
  * attaches the teacher module (EMA: the co-located ref module; trust-region: a
    :class:`TrustRegionTeacher` mixing ref + student).

No verl or core file is modified.
"""

import logging
import os


def _prepend_env_path(key: str, value: str) -> None:
    current = os.environ.get(key)
    if current:
        parts = current.split(":")
        if value not in parts:
            os.environ[key] = f"{value}:{current}"
    else:
        os.environ[key] = value


def _ensure_runtime_env() -> None:
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


_ensure_runtime_env()

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name

from core.workers.fsdp_workers import CustomAsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class SDPOAsyncActorRolloutRefWorker(CustomAsyncActorRolloutRefWorker):
    """Async actor-rollout(-ref) worker with an SDPO actor and a teacher module."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        _ensure_runtime_env()
        if self.rank == 0:
            logger.warning(
                "[SDPO] env before FSDP init: NCCL_ALGO=%s CC=%s LIBRARY_PATH=%s",
                os.environ.get("NCCL_ALGO"),
                os.environ.get("CC"),
                os.environ.get("LIBRARY_PATH"),
            )
        # Build actor / rollout / (fused) ref using the unmodified parent pipeline.
        super().init_model()

        if not self._is_actor:
            return

        from recipe.sdpo.sdpo_dp_actor import SDPODataParallelPPOActor, TrustRegionTeacher

        # Re-wrap the actor as the SDPO actor, re-using the same FSDP module + optimizer.
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = SDPODataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )

        # The checkpoint manager built by the parent references the same optimizer
        # object we just re-passed, so no rebinding is required.

        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        sd_cfg = self.config.actor.get("self_distillation", None)
        if loss_mode not in {"sdpo", "opsd"} or sd_cfg is None:
            return

        teacher_regularization = sd_cfg.get("teacher_regularization", "ema")
        if teacher_regularization == "trust-region":
            self.actor.teacher_module = TrustRegionTeacher(
                ref_module=self.ref_module_fsdp,
                student_module=self.actor_module_fsdp,
                mix_coef=sd_cfg.get("teacher_update_rate", 0.0),
            )
        else:
            # EMA teacher: start from the reference module; EMA-updated each step.
            self.actor.teacher_module = self.ref_module_fsdp

        if self.rank == 0:
            logger.warning(
                "[SDPO] actor wrapped with SDPODataParallelPPOActor; "
                "teacher_regularization=%s",
                teacher_regularization,
            )

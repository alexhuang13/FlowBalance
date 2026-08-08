"""RLSD worker using the proven stable_rl SDPO FSDP/vLLM infrastructure."""
import logging
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass

from recipe.sdpo.sdpo_fsdp_workers import SDPOAsyncActorRolloutRefWorker
from recipe.rlsd.rlsd_dp_actor import RLSDDataParallelPPOActor

logger = logging.getLogger(__file__)


class RLSDAsyncActorRolloutRefWorker(SDPOAsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        teacher = getattr(self.actor, "teacher_module", None)
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = RLSDDataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        self.actor.teacher_module = teacher or self.ref_module_fsdp
        if self.rank == 0:
            logger.warning("[RLSD] actor wrapped; frozen/synced reference teacher enabled")

"""OPSD Hydra entry point on the shared Ray/FSDP/vLLM stack."""

import hydra
import ray

from core.trainer.main_ppo import run_ppo
from recipe.sdpo.main_sdpo import SDPOTaskRunner
from verl.utils.device import auto_set_device


class OPSDTaskRunner(SDPOTaskRunner):
    def run(self, config):
        import core.trainer.main_ppo as core_main
        from recipe.opsd.opsd_ray_trainer import OPSDRayPPOTrainer

        core_main.CustomRayPPOTrainer = OPSDRayPPOTrainer
        # Call the parent of SDPOTaskRunner directly; its run() would overwrite
        # the OPSD trainer alias with SDPORayPPOTrainer.
        return super(SDPOTaskRunner, self).run(config)


@hydra.main(config_path="config", config_name="opsd_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(OPSDTaskRunner))


if __name__ == "__main__":
    main()

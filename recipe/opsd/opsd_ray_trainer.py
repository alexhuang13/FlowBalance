"""OPSD trainer alias.

The shared SDPO trainer constructs privileged teacher prompts. OPSD selects the
external ground-truth solution when present and falls back to a successful
same-rollout solution, as configured by ``solution_source=external_first``.
"""

from recipe.sdpo.sdpo_ray_trainer import SDPORayPPOTrainer


class OPSDRayPPOTrainer(SDPORayPPOTrainer):
    pass

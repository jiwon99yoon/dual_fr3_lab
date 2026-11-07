"""Curriculum for the cabinet book-insert task: learn the insertion first, then make the book wait for the door.

DoorFirst from scratch (45 cm, 2026-11-07) never moved the book: the -20 wait penalty hit as soon as the book left its
start while the book rewards were still far off. Here the run starts with the Upright-like task (no wait penalty, book
rewards from 60-80 deg) and the DoorFirst terms ramp in once the insertion succeeds.
"""

from __future__ import annotations

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .rewards import insert_success


def _lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


class door_first_ramp(ManagerTermBase):
    """Progress p in [0, 1] from the success rate of finished episodes (EMA over ``ema_episodes``).

    p rises by 1 / ``ramp_steps`` per env step while the EMA is at least ``promote_success`` and holds otherwise (a
    stricter stage that drops the success pauses the ramp until the policy recovers). With p:
      - ``wait_term`` weight 0 -> ``wait_weight``,
      - ``gate_terms`` door gates ``gate_start_deg`` -> ``gate_end_deg``,
      - ``success_term`` min door angle 0 -> ``min_door_end_deg``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        self.progress = float(cfg.params.get("initial_progress", 0.0))
        self.success_ema = 0.0
        self._last_step = env.common_step_counter
        self._apply(env, cfg.params)

    def _apply(self, env, params: dict):
        rm, p = env.reward_manager, self.progress
        wait = rm.get_term_cfg(params["wait_term"])
        wait.weight = _lerp(0.0, params["wait_weight"], p)
        rm.set_term_cfg(params["wait_term"], wait)
        gate = tuple(_lerp(a, b, p) for a, b in zip(params["gate_start_deg"], params["gate_end_deg"]))
        for name in params["gate_terms"]:
            rm.get_term_cfg(name).params["gate_deg"] = gate
        rm.get_term_cfg(params["success_term"]).params["min_door_deg"] = _lerp(0.0, params["min_door_end_deg"], p)

    def __call__(
        self,
        env,
        env_ids,
        wait_term: str,
        wait_weight: float,
        gate_terms: list[str],
        gate_start_deg: tuple[float, float],
        gate_end_deg: tuple[float, float],
        success_term: str = "insert_success",
        min_door_end_deg: float = 85.0,
        promote_success: float = 0.7,
        ramp_steps: int = 24000,
        ema_episodes: float = 2000.0,
        initial_progress: float = 0.0,
    ) -> dict[str, float]:
        # called before the reset, so the finished episodes still hold their last state
        ok = insert_success(env, **env.reward_manager.get_term_cfg(success_term).params)[env_ids]
        n = ok.numel()
        if n:
            k = 1.0 - (1.0 - 1.0 / ema_episodes) ** n
            self.success_ema += k * (ok.mean().item() - self.success_ema)
        step = env.common_step_counter
        if self.success_ema >= promote_success and self.progress < 1.0:
            self.progress = min(1.0, self.progress + (step - self._last_step) / ramp_steps)
            self._apply(env, self.cfg.params)
        self._last_step = step
        wait_w = env.reward_manager.get_term_cfg(wait_term).weight
        gate_lo = env.reward_manager.get_term_cfg(gate_terms[0]).params["gate_deg"][0]
        return {"progress": self.progress, "success_ema": self.success_ema, "wait_weight": wait_w, "gate_lo_deg": gate_lo}

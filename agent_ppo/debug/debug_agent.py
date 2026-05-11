#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Interactive debug agent for testing features, buffs, marks, and actions.

Adapted from wty-yy's debug_agent for CC protocol.  Replaces the
dfs_iter_apply_fn + info2dict dependency with direct field iteration
on cc_state_dict typed objects.

Usage:
  1. Set GameConfig.debug_agent = True in conf.py
  2. Run train_test (1 episode, 1 hero as selfplay opponent)
  3. Blue camp (camp 0) auto-tests skills 1/2/3 near (0,0)
  4. Red camp (camp 1) tracks HP deltas, deaths, bullets, buffs/marks
  5. Watch console output for [DEBUG] lines

Action format (same as the 6-head action space):
  [button, move_x, move_z, skill_x, skill_z, target]
  button: 2=move, 3=normal_attack, 4=skill1, 5=skill2, 6=skill3,
          7=recover, 8=summoner, 9=recall
"""

import math
import numpy as np
from typing import Optional

from agent_ppo.feature.cc_state_dict import Info, ActorInfo
from agent_ppo.conf.conf import GameConfig

NO_ACTION = [0] * 6


class DebugAgent:
    """Manual action controller for inspecting game state."""

    def __init__(self):
        self.last_position = None
        self.behaves = set()
        self.soldier_behaves = set()
        self.max_bullets = 0
        self.buff_skill_ids = set()
        self.buff_mark_ids = set()
        self.max_money_delta = 0
        self.last_money = 0
        self.last_hp = 0
        self.max_hp_delta_minus = 0
        # Auto-skill-test state
        self.first_use_skill1 = False
        self.first_use_skill2 = False
        self.first_use_skill3 = False
        self.last_use_skill = False
        self.last_at_target = False
        self.last_level = 0

    def act(self, info: Info) -> list:
        """Return a 6-dim action list based on current frame info."""
        self.info = info
        self.camp = info.player_camp  # 0=blue, 1=red
        self.n_frame = info.n_frame

        # Collect behave modes
        if info.hero_our and info.hero_our.info:
            self.behaves.add(info.hero_our.info.behave)
        if info.hero_enemy and info.hero_enemy.info:
            self.behaves.add(info.hero_enemy.info.behave)
        for soldiers in [info.soldiers_our, info.soldiers_enemy]:
            for s in soldiers.merge:
                self.soldier_behaves.add(s.behave)

        # Collect buff/mark IDs (flat scan of all typed objects)
        for obj in self._iter_actor_infos():
            if obj.buff is not None:
                for sid in obj.buff.skill_ids:
                    self.buff_skill_ids.add(sid)
                for mid in obj.buff.marks_ids:
                    self.buff_mark_ids.add(mid)

        # Track bullets
        n = len(info.bullets_our.merge) + len(info.bullets_enemy.merge)
        self.max_bullets = max(n, self.max_bullets)

        # Death events
        for dead in info.deads:
            d = dead.get("death", {})
            k = dead.get("killer", {})
            if d and k:
                dtype = d.get("actor_type", "?")
                ktype = k.get("actor_type", "?")
                if dtype == 1:
                    self._log(f"hero was killed by: {ktype}")

        # Periodic summary every ~1000 frames
        if info.n_frame % 1000 < 6 or info.n_frame > 4000 - 10:
            self._log(f"behaves={sorted(self.behaves)}")
            self._log(f"soldier_behaves={sorted(self.soldier_behaves)}")
            self._log(f"buff_skill_ids={sorted(self.buff_skill_ids)}")
            self._log(f"buff_mark_ids={sorted(self.buff_mark_ids)}")
            self._log(f"max_bullets={self.max_bullets}")
            self._log(f"hero_config_id={info.hero_our.info.config_id}")

        self.action = NO_ACTION[:]

        if self.camp == 0:  # Blue camp: auto-test skills near origin
            dist = self._move_to(0, 0)
            if dist < 1000 and not self.last_at_target:
                self._log("at (0, 0)")
                self.last_at_target = True
            elif dist >= 1000:
                self.last_at_target = False

            use_skill = False
            if self.n_frame > 500:
                if not self.first_use_skill1:
                    use_skill = self._use_skill(0)
                    if use_skill and not self.first_use_skill1:
                        self._log("Use skill 1")
                if info.hero_our.skill.first.cd != 0 and not self.first_use_skill1:
                    self._log("Use skill 1 Good")
                    self.first_use_skill1 = True

                if not use_skill and not self.first_use_skill2:
                    use_skill = self._use_skill(1)
                    if use_skill and not self.first_use_skill2:
                        self._log("Use skill 2")
                if info.hero_our.skill.second.cd != 0 and not self.first_use_skill2:
                    self._log("Use skill 2 Good")
                    self.first_use_skill2 = True

                if not use_skill and not self.first_use_skill3:
                    use_skill = self._use_skill(2)
                    if use_skill and not self.first_use_skill3:
                        self._log("Use skill 3")
                if info.hero_our.skill.thrid.cd != 0 and not self.first_use_skill3:
                    self._log("Use skill 3 Good")
                    self.first_use_skill3 = True

                if dist < 1000 and not use_skill and not self.last_use_skill:
                    self._normal_attack()

            self.last_use_skill = use_skill

        if self.camp == 1:  # Red camp: passive observation + HP tracking
            dist = self._move_to(0, 0)
            if dist < 1000 and not self.last_at_target:
                self._log("at (0, 0)")
                self.last_at_target = True
            elif dist >= 1000:
                self.last_at_target = False

            # HP delta tracking
            now = info.hero_our.info.hp
            delta = now - self.last_hp
            self.last_hp = now
            self.max_hp_delta_minus = min(self.max_hp_delta_minus, delta)
            if delta < 0:
                self._log(f"HP delta={delta}, now={now}")

            # Level tracking
            if info.hero_our.level != self.last_level and info.n_frame > 300:
                self.last_level = info.hero_our.level
                self._log(f"LEVEL UP -> {self.last_level}")

        # Money delta tracking
        now_money = info.hero_our.money_total
        delta_m = now_money - self.last_money
        self.max_money_delta = max(self.max_money_delta, delta_m)
        if delta_m != 0:
            self._log(f"Money delta={delta_m}")
        self.last_money = now_money

        return self.action

    # ---- action helpers ----

    def _normal_attack(self, target: str = None):
        a = [3, 0, 0, 0, 0, 0]
        if target == "hero":
            a[-1] = 1
        elif target and "soldier" in target:
            a[-1] = int(target[-1]) + 3
        elif target == "organ":
            a[-1] = 7
        self.action = a

    def _move_to(self, x: float, z: float) -> float:
        target = np.array([x, z], np.float32)
        now = np.array(self.info.hero_our.info.position, np.float32)
        delta = self._delta_16x16(now, target)
        self.action = [2, int(delta[0]), int(delta[1]), 0, 0, 0]
        return math.dist(now, target)

    def _follow_target(self, actor: ActorInfo) -> float:
        return self._move_to(actor.position[0], actor.position[1])

    def _use_skill(self, skill_id: int, target: str = None,
                   skill_x: int = 1, skill_z: int = 1) -> bool:
        """Use skill slot. skill_id: 0=skill1, 1=skill2, 2=skill3, 3=recover, 4=flash, 5=recall."""
        button = 4 + skill_id
        info = self.info
        la = info.legal_action
        if not la or len(la) == 0:
            return False
        if not la[button]:
            return False
        sub = info.sub_action_mask
        if not sub or button >= len(sub):
            return False
        mask = sub[button]
        legal_target = mask[-1] if len(mask) > 5 else mask
        if target is None:
            if isinstance(legal_target, (list, np.ndarray)):
                indices = np.argwhere(np.array(legal_target, dtype=bool)).reshape(-1)
                if len(indices) == 0:
                    return False
                target = int(indices[0])
            else:
                target = legal_target
        else:
            tmap = {"hero": 1, "soldier0": 3, "soldier1": 4, "soldier2": 5, "soldier3": 6, "organ": 7}
            target = tmap.get(target, 0)
        if isinstance(legal_target, (list, np.ndarray)) and not legal_target[target]:
            return False
        mask_arr = np.array(mask, np.int32)
        self.action = (np.array([button, 0, 0, skill_x, skill_z, target], np.int32) * mask_arr).tolist()
        return True

    @staticmethod
    def _delta_16x16(center: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - center
        max_abs = np.max(np.abs(delta))
        if max_abs < 1:
            return np.array([8, 8])
        return np.ceil(delta / max_abs * 7).astype(np.int32) + np.array([8, 8])

    def _log(self, msg: str):
        print(f"[DEBUG] Frame{self.n_frame} Camp{self.camp + 1}: {msg}")

    def _iter_actor_infos(self):
        """Yield all ActorInfo objects in the current Info tree."""
        info = self.info
        for hero in [info.hero_our, info.hero_enemy]:
            if hero and hero.info:
                yield hero.info
        for soldiers in [info.soldiers_our, info.soldiers_enemy]:
            for s in soldiers.merge:
                yield s
        for organ in [info.organ_our, info.organ_enemy]:
            if organ.sub_tower:
                yield organ.sub_tower
            if organ.crystal:
                yield organ.crystal
            if organ.spring:
                yield organ.spring
        if info.river_crab is not None:
            yield info.river_crab

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Reward system — 8 active core items + 2 hero-specific bonuses.
========================================================================

Active rewards (non-zero weight in conf.py):
  death_penalty (1.0)    — -1.0 per death, cause classified by proximity to tower
  hp_diff       (2.0)    — clipped delta between hero HP rates (zero-sum)
  last_hit      (0.5)    — 0.5 per last-hit, inferred from NPC distance comparison
  money_diff    (0.006)  — clipped delta/1000 (AAAI paper: minimal farming signal)
  exp_diff      (0.006)  — clipped delta/1000
  hurt_to_hero  (2.0)    — clipped damage/enemy_max_hp
  hurt_to_tower (3.0)    — clipped damage/10000
  tower_hp_diff (10.0)   — zero-sum tower HP rate delta (highest weight: pushing wins)
  kill_hero     (5.0)    — 1.0 per kill
  destroy_tower (10.0)   — binary flag when enemy tower destroyed
  forward       (0.5)    — frame-differential distance-to-enemy-tower / 300
  ── Hero-specific ───────────────────────────────────────────────────────
  luban_passive_combo (1.0) — 鲁班 burst: ult→skill1→skill2 within 40f window
  dirj_cleanse_reward (0.5) — 狄仁杰 skill2 used while debuffed (active marks)

Detection mechanisms:
  - Skill usage:    CD transition from 0→positive (reliable, no button index needed)
  - Last-hit:       NPC HP tracking + distance comparison (main < enemy → last-hit)
  - Death cause:    hero_death within TOWER_ATTACK_RANGE*2 of enemy tower → tower kill
  - Tower dive:     taking damage while in enemy tower attack range
  - Heal/flash:     HP jump (>5% hp_rate) / position jump (>FLASH_DISTANCE_THRESHOLD)
                    when summoner skill CD transitions

Reward clipping:  individual values clipped to [-5, 5] before × weight.
No time decay (TIME_SCALE_ARG=0).
"""

import math
from agent_ppo.conf.conf import (
    GameConfig,
    MINION_MAX_HP_RANGE,
    MONSTER_MAX_HP_THRESHOLD,
    TOWER_ATTACK_RANGE,
    FLASH_DISTANCE_THRESHOLD,
)

REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 5.0


def _clip(value, lo=REWARD_CLIP_MIN, hi=REWARD_CLIP_MAX):
    return max(lo, min(hi, value))


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.weights = GameConfig.REWARD_WEIGHT_DICT
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG

        self.prev = {}
        self.first_frame = True

        # 局内统计器
        self.death_by_hero = 0
        self.death_by_tower = 0
        self.skill_usage = [0, 0, 0]
        self.summoner_usage = 0
        self.minion_kills = 0
        self.monster_kills = 0
        self.last_hit_count = 0
        self.forward_acc = 0.0
        self.forward_count = 0
        self.destroy_tower_flag = 0
        self.last_frame_no = 0

        # 帧差分推断状态
        self.prev_skill_cd = []
        self.prev_summoner_cd = 0
        self.prev_npcs = {}
        self.alive = True

        # 批次2:生存与发育跟踪
        self.prev_hp_rate = None
        self.prev_enemy_hp_rate = None
        self.prev_pos = None
        self.prev_enemy_pos = None
        self.prev_dist_to_enemy_tower = None
        self.heal_count = 0
        self.heal_smart_count = 0
        self.flash_count = 0
        self.flash_smart_count = 0
        self.tower_dive_count = 0
        self.tower_dive_death = 0
        self.idle_frames = 0
        self.frames_since_action = 0
        self.prev_last_hit_count = 0

        # 批次3:英雄特定技能追踪
        self.hero_config_id = 0
        self.skill2_check_window = 0
        self.skill2_ref_hp = 0.0
        self.dirj_skill2_use_count = 0
        self.dirj_skill2_hit_count = 0
        self.dirj_cleanse_count = 0
        self.lane_arrival_done = False
        self.lane_arrival_frame = 0

        # 鲁班连招追踪: 大招→1技能→2技能
        self.luban_combo_stage = 0  # 0=待机, 1=大招已放等1技能, 2=1技能已放等2技能
        self.luban_combo_window = 0
        self.luban_combo_count = 0

        # 敌方血包安全吃掉追踪（仅己方小兵扛塔且敌方英雄阵亡/视野外时）
        self.prev_enemy_cake_pos = None  # 上一帧敌方血包位置
        self.safe_cake_eat_count = 0

        # Matchup 开局引导追踪
        self.enemy_config_id = 0
        self.prev_dist_to_enemy = None  # 上一帧到敌方英雄距离
        self.prev_dist_to_my_tower = None  # 上一帧到己方塔距离
        self.my_tower_pos = None  # 己方塔位置（固定，开局计算一次）

    def _hero_by_camp(self, frame_data, camp):
        for h in frame_data["hero_states"]:
            if h["camp"] == camp:
                return h
        return None

    def _find_camps(self, frame_data):
        main_camp, enemy_camp = None, None
        for h in frame_data["hero_states"]:
            if h["runtime_id"] == self.main_hero_player_id:
                main_camp = h["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = h["camp"]
        return main_camp, enemy_camp

    def _tower_hp_rate(self, frame_data, camp):
        for npc in frame_data["npc_states"]:
            if npc.get("sub_type") == 21 and npc["camp"] == camp:
                max_hp = max(npc.get("max_hp", 1), 1)
                return npc.get("hp", 0) / max_hp
        return 0.0

    def _forward_value(self, main_hero, frame_data, main_camp):
        if main_hero is None:
            return 0.0
        enemy_tower = None
        for npc in frame_data["npc_states"]:
            if npc.get("sub_type") == 21 and npc["camp"] != main_camp:
                enemy_tower = npc
                break
        if enemy_tower is None:
            return 0.0
        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        en_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        cur_dist = math.dist(hero_pos, en_pos)
        if self.prev_dist_to_enemy_tower is None:
            self.prev_dist_to_enemy_tower = cur_dist
            return 0.0
        delta = self.prev_dist_to_enemy_tower - cur_dist  # 正值=靠近敌塔
        self.prev_dist_to_enemy_tower = cur_dist
        # 归一化到 [-1, 1]，用地图对角线约30000作为参考
        return max(-1.0, min(1.0, delta / 300.0))

    def result(self, frame_data):
        main_camp, enemy_camp = self._find_camps(frame_data)
        main_hero = self._hero_by_camp(frame_data, main_camp)
        enemy_hero = self._hero_by_camp(frame_data, enemy_camp)

        cur = {}
        cur["main_hp_rate"] = main_hero.get("hp", 0) / max(main_hero.get("max_hp", 1), 1) if main_hero else 0.0
        cur["enemy_hp_rate"] = enemy_hero.get("hp", 0) / max(enemy_hero.get("max_hp", 1), 1) if enemy_hero else 0.0
        cur["main_money"] = main_hero.get("money", 0) if main_hero else 0
        cur["enemy_money"] = enemy_hero.get("money", 0) if enemy_hero else 0
        cur["main_exp"] = main_hero.get("exp", 0) if main_hero else 0
        cur["enemy_exp"] = enemy_hero.get("exp", 0) if enemy_hero else 0
        cur["main_level"] = main_hero.get("level", 1) if main_hero else 1
        cur["enemy_level"] = enemy_hero.get("level", 1) if enemy_hero else 1
        cur["main_hurt_to_hero"] = main_hero.get("total_hurt_to_hero", 0) if main_hero else 0
        cur["main_hurt_to_tower"] = main_hero.get("total_hurt_to_tower", 0) if main_hero else 0
        cur["main_kill"] = main_hero.get("kill_count", 0) if main_hero else 0
        cur["main_dead"] = main_hero.get("dead_count", 0) if main_hero else 0
        cur["main_tower_hp"] = self._tower_hp_rate(frame_data, main_camp) if main_camp else 0.0
        cur["enemy_tower_hp"] = self._tower_hp_rate(frame_data, enemy_camp) if enemy_camp else 0.0

        if self.first_frame:
            self.prev = cur.copy()
            self.first_frame = False
            if main_hero:
                self.prev_hp_rate = cur["main_hp_rate"]
                self.prev_enemy_hp_rate = cur["enemy_hp_rate"]
                loc = main_hero.get("location", {})
                self.prev_pos = (loc.get("x", 0), loc.get("z", 0))
            return {"reward_sum": 0.0}

        reward_dict = {}
        w = self.weights

        # ── 基础 12 项 ───────────────────────────────────────
        d_main_dead = cur["main_dead"] - self.prev["main_dead"]
        reward_dict["death_penalty"] = _clip(-1.0 * d_main_dead) * w.get("death_penalty", 0)

        d_hp = (cur["main_hp_rate"] - self.prev["main_hp_rate"]) - (cur["enemy_hp_rate"] - self.prev["enemy_hp_rate"])
        reward_dict["hp_diff"] = _clip(d_hp) * w.get("hp_diff", 0)

        d_kill = cur["main_kill"] - self.prev["main_kill"]
        reward_dict["last_hit"] = 0.0  # 占位,NPC 推断后更新

        d_money = (cur["main_money"] - self.prev["main_money"]) - (cur["enemy_money"] - self.prev["enemy_money"])
        reward_dict["money_diff"] = _clip(d_money / 1000.0) * w.get("money_diff", 0)

        d_exp = (cur["main_exp"] - self.prev["main_exp"]) - (cur["enemy_exp"] - self.prev["enemy_exp"])
        reward_dict["exp_diff"] = _clip(d_exp / 1000.0) * w.get("exp_diff", 0)

        d_level = (cur["main_level"] - cur["enemy_level"])
        reward_dict["level_diff"] = _clip(d_level) * w.get("level_diff", 0)

        d_hurt_hero = cur["main_hurt_to_hero"] - self.prev["main_hurt_to_hero"]
        enemy_max_hp = max(enemy_hero.get("max_hp", 1), 1) if enemy_hero else 1
        reward_dict["hurt_to_hero"] = _clip(d_hurt_hero / enemy_max_hp) * w.get("hurt_to_hero", 0)

        d_hurt_tower = cur["main_hurt_to_tower"] - self.prev["main_hurt_to_tower"]
        reward_dict["hurt_to_tower"] = _clip(d_hurt_tower / 10000.0) * w.get("hurt_to_tower", 0)

        d_tower = (cur["main_tower_hp"] - self.prev["main_tower_hp"]) - (cur["enemy_tower_hp"] - self.prev["enemy_tower_hp"])
        reward_dict["tower_hp_diff"] = _clip(d_tower) * w.get("tower_hp_diff", 0)

        reward_dict["kill_hero"] = _clip(1.0 * d_kill) * w.get("kill_hero", 0)

        tower_destroyed = 0.0
        if self.prev["enemy_tower_hp"] > 0 and cur["enemy_tower_hp"] <= 0:
            tower_destroyed = 1.0
        reward_dict["destroy_tower"] = _clip(tower_destroyed) * w.get("destroy_tower", 0)

        fwd = self._forward_value(main_hero, frame_data, main_camp)
        reward_dict["forward"] = _clip(fwd) * w.get("forward", 0)

        # ── 批次2/3 占位 ─────────────────────────────────────
        reward_dict["minion_attack"] = 0.0
        reward_dict["monster_attack"] = 0.0
        reward_dict["heal_smart"] = 0.0
        reward_dict["flash_smart"] = 0.0
        reward_dict["tower_dive_penalty"] = 0.0
        reward_dict["retreat_smart"] = 0.0
        reward_dict["idle_penalty"] = 0.0
        reward_dict["luban_skill_0_clear"] = 0.0
        reward_dict["dirj_skill_2_hit"] = 0.0
        reward_dict["dirj_skill_2_miss"] = 0.0
        reward_dict["dirj_cleanse_reward"] = 0.0
        reward_dict["lane_arrival"] = 0.0
        reward_dict["early_aggression_penalty"] = 0.0
        reward_dict["safe_cake_eat"] = 0.0
        reward_dict["aggressive_forward"] = 0.0
        reward_dict["stay_near_tower"] = 0.0
        reward_dict["mirror_farm_bonus"] = 0.0
        reward_dict["avoid_enemy_hero"] = 0.0

        # 获取英雄 config_id + 敌方 config_id
        if main_hero is not None:
            self.hero_config_id = main_hero.get("config_id", self.hero_config_id)
        if enemy_hero is not None:
            self.enemy_config_id = enemy_hero.get("config_id", self.enemy_config_id)

        # ── 死亡原因推断 ────────────────────────────────────
        if d_main_dead > 0 and main_hero is not None and enemy_camp is not None:
            enemy_tower_pos = None
            for npc in frame_data.get("npc_states", []):
                if npc.get("sub_type") == 21 and npc["camp"] == enemy_camp:
                    loc = npc.get("location", {})
                    enemy_tower_pos = (loc.get("x", 0), loc.get("z", 0))
                    break
            if enemy_tower_pos is not None:
                hero_loc = main_hero.get("location", {})
                hero_pos_d = (hero_loc.get("x", 0), hero_loc.get("z", 0))
                dist_to_etower = math.dist(hero_pos_d, enemy_tower_pos)
                if dist_to_etower <= TOWER_ATTACK_RANGE * 2:
                    self.death_by_tower += d_main_dead
                    self.tower_dive_death += d_main_dead
                else:
                    self.death_by_hero += d_main_dead
            self.alive = False

        # ── destroy_tower_flag ──────────────────────────────
        if tower_destroyed > 0:
            self.destroy_tower_flag = 1

        # ── 技能使用推断 ────────────────────────────────────
        summoner_triggered = False
        if main_hero is not None:
            skill_slots = main_hero.get("skill_slot_list", [])
            if not self.prev_skill_cd:
                self.prev_skill_cd = [s.get("cool_down", 0) for s in skill_slots]
            for i in range(min(3, len(skill_slots))):
                prev_cd = self.prev_skill_cd[i] if i < len(self.prev_skill_cd) else 0
                cur_cd = skill_slots[i].get("cool_down", 0)
                if prev_cd == 0 and cur_cd > 0:
                    self.skill_usage[i] += 1
            prev_skill_cd_snapshot = list(self.prev_skill_cd)
            self.prev_skill_cd = [s.get("cool_down", 0) for s in skill_slots]
            if len(skill_slots) > 3:
                prev_s_cd = self.prev_summoner_cd
                cur_s_cd = skill_slots[3].get("cool_down", 0)
                if prev_s_cd == 0 and cur_s_cd > 0:
                    self.summoner_usage += 1
                    summoner_triggered = True
                self.prev_summoner_cd = cur_s_cd
        else:
            prev_skill_cd_snapshot = []

        # ── 小兵/怪物/补刀推断（含攻击奖励）────────────────
        minion_attacked = False
        monster_attacked = False
        if main_hero is not None:
            main_loc = main_hero.get("location", {})
            main_pos = (main_loc.get("x", 0), main_loc.get("z", 0))
            enemy_hero_obj = self._hero_by_camp(frame_data, enemy_camp) if enemy_camp else None
            enemy_loc = enemy_hero_obj.get("location", {}) if enemy_hero_obj else {}
            enemy_pos = (enemy_loc.get("x", 0), enemy_loc.get("z", 0))

            cur_npcs = {}
            for npc in frame_data.get("npc_states", []):
                rid = npc.get("runtime_id")
                if rid is None:
                    continue
                loc = npc.get("location", {})
                cur_npcs[rid] = {
                    "hp": npc.get("hp", 0),
                    "max_hp": npc.get("max_hp", 1),
                    "sub_type": npc.get("sub_type", 0),
                    "camp": npc.get("camp", -1),
                    "x": loc.get("x", 0),
                    "z": loc.get("z", 0),
                }

            for rid, prev_npc in self.prev_npcs.items():
                if prev_npc["hp"] <= 0:
                    continue
                cur_npc = cur_npcs.get(rid)
                sub = prev_npc["sub_type"]
                mhp = prev_npc["max_hp"]

                if cur_npc is None or cur_npc["hp"] <= 0:
                    if sub == 21:
                        pass
                    elif MINION_MAX_HP_RANGE[0] <= mhp <= MINION_MAX_HP_RANGE[1]:
                        self.minion_kills += 1
                        npc_pos = (prev_npc["x"], prev_npc["z"])
                        d_main = math.dist(main_pos, npc_pos)
                        d_enemy = math.dist(enemy_pos, npc_pos) if enemy_hero_obj else float("inf")
                        if d_main < d_enemy:
                            self.last_hit_count += 1
                    elif mhp > MONSTER_MAX_HP_THRESHOLD:
                        self.monster_kills += 1

                elif cur_npc["hp"] < prev_npc["hp"]:
                    if sub != 21 and MINION_MAX_HP_RANGE[0] <= mhp <= MINION_MAX_HP_RANGE[1] and not minion_attacked:
                        npc_pos = (prev_npc["x"], prev_npc["z"])
                        if math.dist(main_pos, npc_pos) < 1500:
                            minion_attacked = True
                    elif sub != 21 and mhp > MONSTER_MAX_HP_THRESHOLD and not monster_attacked:
                        npc_pos = (prev_npc["x"], prev_npc["z"])
                        if math.dist(main_pos, npc_pos) < 2000:
                            monster_attacked = True

            self.prev_npcs = cur_npcs

        # ── last_hit 奖励 ───────────────────────────────────
        d_last_hit = self.last_hit_count - self.prev_last_hit_count
        if d_last_hit > 0:
            reward_dict["last_hit"] = _clip(d_last_hit * 0.5) * w.get("last_hit", 0)
        self.prev_last_hit_count = self.last_hit_count

        # ── 发育奖励 ────────────────────────────────────────
        if minion_attacked:
            reward_dict["minion_attack"] = 0.1 * w.get("minion_attack", 0)
        if monster_attacked:
            reward_dict["monster_attack"] = 0.1 * w.get("monster_attack", 0)

        # ── 治疗/闪现智能使用奖励 ──────────────────────────
        if summoner_triggered and main_hero is not None:
            cur_hp_rate = cur["main_hp_rate"]
            cur_pos = None
            loc = main_hero.get("location", {})
            cur_pos = (loc.get("x", 0), loc.get("z", 0))

            hp_jumped = (self.prev_hp_rate is not None) and (cur_hp_rate > self.prev_hp_rate + 0.05)
            pos_jumped = False
            if self.prev_pos is not None and cur_pos is not None:
                pos_jumped = math.dist(cur_pos, self.prev_pos) > FLASH_DISTANCE_THRESHOLD

            if hp_jumped:
                self.heal_count += 1
                if cur_hp_rate < 0.5:
                    self.heal_smart_count += 1
                    bonus = (0.5 - self.prev_hp_rate) * 2.0
                    reward_dict["heal_smart"] = _clip(bonus) * w.get("heal_smart", 0)
                else:
                    reward_dict["heal_smart"] = _clip(-1.0) * w.get("heal_smart", 0)

            elif pos_jumped:
                self.flash_count += 1
                enemy_hero_obj = self._hero_by_camp(frame_data, enemy_camp) if enemy_camp else None
                if enemy_hero_obj and cur_pos and self.prev_pos:
                    e_loc = enemy_hero_obj.get("location", {})
                    e_pos = (e_loc.get("x", 0), e_loc.get("z", 0))
                    dist_before = math.dist(self.prev_pos, e_pos)
                    dist_after = math.dist(cur_pos, e_pos)
                    moved_away = dist_after > dist_before
                    if cur["main_hp_rate"] < 0.3 and moved_away:
                        self.flash_smart_count += 1
                        reward_dict["flash_smart"] = 1.0 * w.get("flash_smart", 0)
                    elif cur["main_hp_rate"] < 0.3 and not moved_away:
                        reward_dict["flash_smart"] = _clip(-1.0) * w.get("flash_smart", 0)
                    elif cur["main_hp_rate"] > 0.5:
                        reward_dict["flash_smart"] = _clip(-0.3) * w.get("flash_smart", 0)

        # ── 越塔惩罚 ────────────────────────────────────────
        if main_hero is not None and enemy_camp is not None:
            cur_hp_rate = cur["main_hp_rate"]
            hp_dropped = (self.prev_hp_rate is not None) and (cur_hp_rate < self.prev_hp_rate - 0.005)
            if hp_dropped:
                hero_loc = main_hero.get("location", {})
                hero_pos_now = (hero_loc.get("x", 0), hero_loc.get("z", 0))
                for npc in frame_data.get("npc_states", []):
                    if npc.get("sub_type") == 21 and npc.get("camp") == enemy_camp:
                        t_loc = npc.get("location", {})
                        t_pos = (t_loc.get("x", 0), t_loc.get("z", 0))
                        dx = hero_pos_now[0] - t_pos[0]
                        dz = hero_pos_now[1] - t_pos[1]
                        dist_to_etower = (dx * dx + dz * dz) ** 0.5
                        if dist_to_etower < TOWER_ATTACK_RANGE:
                            reward_dict["tower_dive_penalty"] = _clip(-0.5) * w.get("tower_dive_penalty", 0)
                            self.tower_dive_count += 1
                        break

        # ── 残血撤退奖励 ────────────────────────────────────
        if main_hero is not None and cur["main_hp_rate"] < 0.3:
            enemy_hp_rate = cur["enemy_hp_rate"]
            if enemy_hp_rate > cur["main_hp_rate"] * 1.5:
                my_tower_pos = None
                for npc in frame_data.get("npc_states", []):
                    if npc.get("sub_type") == 21 and npc.get("camp") == main_camp:
                        t_loc = npc.get("location", {})
                        my_tower_pos = (t_loc.get("x", 0), t_loc.get("z", 0))
                        break
                if my_tower_pos is not None and self.prev_pos is not None:
                    hero_loc = main_hero.get("location", {})
                    cur_pos_r = (hero_loc.get("x", 0), hero_loc.get("z", 0))
                    d_now = math.dist(cur_pos_r, my_tower_pos)
                    d_prev = math.dist(self.prev_pos, my_tower_pos)
                    if d_now < d_prev:
                        reward_dict["retreat_smart"] = 0.2 * w.get("retreat_smart", 0)

        # ── 批次3:英雄特定技能引导 ─────────────────────────
        if main_hero is not None:
            skill_slots = main_hero.get("skill_slot_list", [])

            # 鲁班一技能清线
            if self.hero_config_id == 112 and len(skill_slots) > 0:
                skill_0_triggered = (
                    len(prev_skill_cd_snapshot) > 0
                    and prev_skill_cd_snapshot[0] == 0
                    and skill_slots[0].get("cool_down", 0) > 0
                )
                if skill_0_triggered and enemy_camp is not None:
                    hero_loc = main_hero.get("location", {})
                    hero_pos_s = (hero_loc.get("x", 0), hero_loc.get("z", 0))
                    nearby_enemy_minions = 0
                    for npc in frame_data.get("npc_states", []):
                        if npc.get("camp") == enemy_camp and npc.get("hp", 0) > 0:
                            mhp = npc.get("max_hp", 0)
                            if MINION_MAX_HP_RANGE[0] <= mhp <= MINION_MAX_HP_RANGE[1]:
                                nloc = npc.get("location", {})
                                npos = (nloc.get("x", 0), nloc.get("z", 0))
                                if math.dist(hero_pos_s, npos) < 2000:
                                    nearby_enemy_minions += 1
                    if nearby_enemy_minions >= 2:
                        reward_dict["luban_skill_0_clear"] = 0.3 * w.get("luban_skill_0_clear", 0)

                # 鲁班连招: 大招(slots[2]) → 1技能(slots[0]) → 2技能(slots[1])
                # 检测 CD 跳变判定技能使用: prev_cd==0 → cur_cd>0
                def _skill_i_triggered(idx):
                    return (
                        len(prev_skill_cd_snapshot) > idx
                        and prev_skill_cd_snapshot[idx] == 0
                        and len(skill_slots) > idx
                        and skill_slots[idx].get("cool_down", 0) > 0
                    )

                if self.luban_combo_window > 0:
                    self.luban_combo_window -= 1
                    if self.luban_combo_window == 0:
                        self.luban_combo_stage = 0

                if _skill_i_triggered(2):  # 大招
                    self.luban_combo_stage = 1
                    self.luban_combo_window = 40
                elif _skill_i_triggered(0) and self.luban_combo_stage == 1:  # 1技能
                    self.luban_combo_stage = 2
                    self.luban_combo_window = 40
                elif _skill_i_triggered(1) and self.luban_combo_stage == 2:  # 2技能
                    self.luban_combo_count += 1
                    reward_dict["luban_passive_combo"] = 1.0 * w.get("luban_passive_combo", 0)
                    self.luban_combo_stage = 0
                    self.luban_combo_window = 0
                elif _skill_i_triggered(0) or _skill_i_triggered(1) or _skill_i_triggered(2):
                    self.luban_combo_stage = 0
                    self.luban_combo_window = 0

            # 狄仁杰大招命中判定
            if self.hero_config_id == 133 and len(skill_slots) > 2:
                skill_2_triggered = (
                    len(prev_skill_cd_snapshot) > 2
                    and prev_skill_cd_snapshot[2] == 0
                    and skill_slots[2].get("cool_down", 0) > 0
                )
                if skill_2_triggered:
                    self.dirj_skill2_use_count += 1
                    self.skill2_check_window = 8
                    self.skill2_ref_hp = cur["enemy_hp_rate"]

            if self.skill2_check_window > 0:
                self.skill2_check_window -= 1
                if cur["enemy_hp_rate"] < self.skill2_ref_hp - 0.01:
                    self.dirj_skill2_hit_count += 1
                    reward_dict["dirj_skill_2_hit"] = 1.0 * w.get("dirj_skill_2_hit", 0)
                    self.skill2_check_window = 0
                elif self.skill2_check_window == 0:
                    reward_dict["dirj_skill_2_miss"] = _clip(-0.3) * w.get("dirj_skill_2_miss", 0)

            # 狄仁杰二技能解控 (skill_slots[1] = 二技能)
            if self.hero_config_id == 133 and len(skill_slots) > 1:
                _debuffed = False
                _bs = main_hero.get("buff_state", {})
                if isinstance(_bs, dict):
                    for _m in _bs.get("buff_marks", []):
                        if isinstance(_m, dict) and _m.get("layer", 0) > 0:
                            _debuffed = True
                            break
                if _debuffed:
                    _cleanse_triggered = (
                        len(prev_skill_cd_snapshot) > 1
                        and prev_skill_cd_snapshot[1] == 0
                        and skill_slots[1].get("cool_down", 0) > 0
                    )
                    if _cleanse_triggered:
                        self.dirj_cleanse_count += 1
                        reward_dict["dirj_cleanse_reward"] = 0.5 * w.get("dirj_cleanse_reward", 0)

        # ── 批次3:1级阶段清线优先 ──────────────────────────
        if cur["main_level"] == 1 and d_hurt_hero > 0:
            reward_dict["early_aggression_penalty"] = _clip(-0.1) * w.get("early_aggression_penalty", 0)

        # ── 批次3:开局路径引导(前 450 帧向中线靠近)────────
        if main_hero is not None and self.last_frame_no < 450:
            hero_loc = main_hero.get("location", {})
            cur_pos_l = (hero_loc.get("x", 0), hero_loc.get("z", 0))
            dist_to_mid = math.dist(cur_pos_l, (0, 0))
            if not self.lane_arrival_done and dist_to_mid < 8000:
                self.lane_arrival_done = True
                self.lane_arrival_frame = self.last_frame_no
            if not self.lane_arrival_done and self.prev_pos is not None:
                prev_dist_to_mid = math.dist(self.prev_pos, (0, 0))
                delta = prev_dist_to_mid - dist_to_mid
                reward_dict["lane_arrival"] = _clip(delta / 1000.0) * 0.3 * w.get("lane_arrival", 0)

        # ── 安全吃敌方塔后血包 ──────────────────────────────
        # 条件：敌方血包本帧消失 + 英雄在血包位置附近 + 敌方英雄阵亡或视野外
        #       + 己方小兵在敌方塔附近扛塔 + 主英雄未被敌方塔选为攻击目标
        cur_enemy_cake_pos = None
        for cake in frame_data.get("cakes", []) or []:
            collider = cake.get("collider", {}) if isinstance(cake, dict) else {}
            loc = collider.get("location", {}) if isinstance(collider, dict) else {}
            cx = loc.get("x", 0) if isinstance(loc, dict) else 0
            cz = loc.get("z", 0) if isinstance(loc, dict) else 0
            # 敌方血包启发式：位置 x 符号与敌方塔一致（主阵营 0→蓝方塔在-5000侧，敌方=红方=+5000侧）
            # 简化：只要不是己方血包就当敌方，用距离敌塔近做判定
            # 这里先记录所有血包，下面用位置匹配判定归属
            if cx == 0 and cz == 0:
                continue
            # 判定是否是敌方血包：距离敌方塔的距离 < 距离己方塔的距离
            e_tower_pos, m_tower_pos = None, None
            for npc in frame_data.get("npc_states", []):
                if npc.get("sub_type") == 21:
                    nloc = npc.get("location", {})
                    npos = (nloc.get("x", 0), nloc.get("z", 0))
                    if npc.get("camp") == enemy_camp:
                        e_tower_pos = npos
                    elif npc.get("camp") == main_camp:
                        m_tower_pos = npos
            if e_tower_pos is None:
                continue
            cake_pos = (cx, cz)
            d_to_etower = math.dist(cake_pos, e_tower_pos)
            d_to_mtower = math.dist(cake_pos, m_tower_pos) if m_tower_pos else float("inf")
            if d_to_etower < d_to_mtower:
                cur_enemy_cake_pos = cake_pos
                break

        # 检测"敌方血包消失"事件：上一帧有，这一帧没了
        if (self.prev_enemy_cake_pos is not None and cur_enemy_cake_pos is None
                and main_hero is not None and enemy_camp is not None):
            hero_loc = main_hero.get("location", {})
            hero_pos_c = (hero_loc.get("x", 0), hero_loc.get("z", 0))
            dist_to_cake = math.dist(hero_pos_c, self.prev_enemy_cake_pos)
            # 条件1：主英雄在血包位置附近(消失时)
            near_cake = dist_to_cake < 1500

            # 条件2：敌方英雄阵亡或不在视野内
            enemy_dead_or_unseen = True
            if enemy_hero is not None:
                en_hp = enemy_hero.get("hp", 0)
                en_loc = enemy_hero.get("location", {})
                en_x = en_loc.get("x", 100000)
                if en_hp > 0 and en_x != 100000:  # 敌方活着且可见
                    # 敌方英雄距主英雄 < 视野阈值(8000)视为可见威胁
                    if math.dist(hero_pos_c, (en_x, en_loc.get("z", 0))) < 8000:
                        enemy_dead_or_unseen = False

            # 条件3：己方小兵在敌方塔附近扛塔
            minion_tanking = False
            e_tower_pos_chk = None
            for npc in frame_data.get("npc_states", []):
                if npc.get("sub_type") == 21 and npc.get("camp") == enemy_camp:
                    tloc = npc.get("location", {})
                    e_tower_pos_chk = (tloc.get("x", 0), tloc.get("z", 0))
                    break
            if e_tower_pos_chk is not None:
                for npc in frame_data.get("npc_states", []):
                    if npc.get("camp") == main_camp and npc.get("hp", 0) > 0:
                        mhp = npc.get("max_hp", 0)
                        if MINION_MAX_HP_RANGE[0] <= mhp <= MINION_MAX_HP_RANGE[1]:
                            nloc = npc.get("location", {})
                            npos_m = (nloc.get("x", 0), nloc.get("z", 0))
                            if math.dist(npos_m, e_tower_pos_chk) < TOWER_ATTACK_RANGE:
                                minion_tanking = True
                                break

            # 条件4：主英雄不是敌方塔的攻击目标
            not_tower_target = True
            for npc in frame_data.get("npc_states", []):
                if npc.get("sub_type") == 21 and npc.get("camp") == enemy_camp:
                    if npc.get("attack_target", 0) == main_hero.get("runtime_id", -1):
                        not_tower_target = False
                    break

            if near_cake and enemy_dead_or_unseen and minion_tanking and not_tower_target:
                self.safe_cake_eat_count += 1
                reward_dict["safe_cake_eat"] = 1.0 * w.get("safe_cake_eat", 0)

        self.prev_enemy_cake_pos = cur_enemy_cake_pos

        # ── 空闲惩罚 ────────────────────────────────────────
        if main_hero is not None:
            hero_loc = main_hero.get("location", {})
            cur_pos_i = (hero_loc.get("x", 0), hero_loc.get("z", 0))
            skill_triggered_any = summoner_triggered or (
                any(
                    (prev_skill_cd_snapshot[i] if i < len(prev_skill_cd_snapshot) else 1) == 0 and
                    main_hero.get("skill_slot_list", [{}])[i].get("cool_down", 0) > 0
                    for i in range(min(3, len(main_hero.get("skill_slot_list", []))))
                )
                if prev_skill_cd_snapshot else False
            )
            pos_moved = self.prev_pos is not None and math.dist(cur_pos_i, self.prev_pos) > 500

            if not skill_triggered_any and not pos_moved:
                self.frames_since_action += 1
            else:
                self.frames_since_action = 0

            if self.frames_since_action > 90:
                reward_dict["idle_penalty"] = -0.05 * w.get("idle_penalty", 0)
                self.idle_frames += 1

        # ── Matchup 开局引导（前 2000 帧）───────────────────
        if main_hero is not None and enemy_hero is not None and self.last_frame_no < 2000:
            matchup = (self.hero_config_id, self.enemy_config_id)
            hero_pos_g = (main_hero.get("location", {}).get("x", 0),
                         main_hero.get("location", {}).get("z", 0))
            e_loc = enemy_hero.get("location", {})
            enemy_pos_g = (e_loc.get("x", 100000), e_loc.get("z", 100000))
            enemy_visible = enemy_pos_g[0] != 100000

            # 己方塔位置（缓存，只算一次）
            if self.my_tower_pos is None:
                for npc in frame_data.get("npc_states", []):
                    if npc.get("sub_type") == 21 and npc.get("camp") == main_camp:
                        tloc = npc.get("location", {})
                        self.my_tower_pos = (tloc.get("x", 0), tloc.get("z", 0))
                        break

            # 距离衰减系数：权重在 2000 帧内线性衰减到 0
            fade = max(0.0, 1.0 - self.last_frame_no / 2000.0)

            if matchup in ((112, 112), (133, 133)):  # mirror: 清线优先
                if d_hurt_hero > 0 and cur["main_level"] <= 2:
                    reward_dict["mirror_farm_bonus"] = _clip(-0.05 * fade) * w.get("mirror_farm_bonus", 0)
                if minion_attacked:
                    reward_dict["mirror_farm_bonus"] += _clip(0.05 * fade) * w.get("mirror_farm_bonus", 0)

            elif matchup == (112, 133):  # 鲁班打狄仁杰: 主动换血
                if enemy_visible and self.prev_dist_to_enemy is not None:
                    cur_d = math.dist(hero_pos_g, enemy_pos_g)
                    delta = self.prev_dist_to_enemy - cur_d  # 正=靠近
                    reward_dict["aggressive_forward"] = _clip(delta / 300.0 * fade) * w.get("aggressive_forward", 0)
                self.prev_dist_to_enemy = math.dist(hero_pos_g, enemy_pos_g) if enemy_visible else None

            elif matchup == (133, 112):  # 狄仁杰打鲁班: 塔前防守+绕开敌方清线
                if self.my_tower_pos is not None:
                    d_tower = math.dist(hero_pos_g, self.my_tower_pos)
                    if self.last_frame_no < 1000 and d_tower < 2000 and cur["main_hp_rate"] > 0.8:
                        reward_dict["stay_near_tower"] = _clip(0.03 * fade) * w.get("stay_near_tower", 0)
                    if self.last_frame_no >= 1000:
                        if enemy_visible:
                            d_enemy = math.dist(hero_pos_g, enemy_pos_g)
                            if d_enemy > 5000:
                                reward_dict["avoid_enemy_hero"] = _clip(0.03 * fade) * w.get("avoid_enemy_hero", 0)
                            elif d_enemy < 3000:
                                reward_dict["avoid_enemy_hero"] = _clip(-0.03 * fade) * w.get("avoid_enemy_hero", 0)

        # ── 残血撤退引导（全局）─────────────────────────────
        if main_hero is not None and enemy_hero is not None:
            hp_rate_r = cur["main_hp_rate"]
            enemy_alive = enemy_hero.get("hp", 0) > 0
            e_loc_r = enemy_hero.get("location", {})
            enemy_r_visible = e_loc_r.get("x", 100000) != 100000
            enemy_hp_rate_r = cur["enemy_hp_rate"]

            should_retreat = (
                hp_rate_r < 0.35
                and enemy_alive
                and enemy_r_visible
                and enemy_hp_rate_r > hp_rate_r * 1.3
            )

            if should_retreat and self.my_tower_pos is not None:
                hero_pos_r = (main_hero.get("location", {}).get("x", 0),
                             main_hero.get("location", {}).get("z", 0))
                cur_d_tower = math.dist(hero_pos_r, self.my_tower_pos)
                if self.prev_dist_to_my_tower is not None:
                    delta_t = self.prev_dist_to_my_tower - cur_d_tower
                    reward_dict["retreat_smart"] = _clip(delta_t / 300.0) * 0.3 * w.get("retreat_smart", 0)
                self.prev_dist_to_my_tower = cur_d_tower
            else:
                self.prev_dist_to_my_tower = None

        # ── 更新持久化状态 ──────────────────────────────────
        if main_hero is not None:
            self.prev_hp_rate = cur["main_hp_rate"]
            self.prev_enemy_hp_rate = cur["enemy_hp_rate"]
            h_loc = main_hero.get("location", {})
            self.prev_pos = (h_loc.get("x", 0), h_loc.get("z", 0))

        # ── forward 累加 ────────────────────────────────────
        self.forward_acc += fwd
        self.forward_count += 1

        # ── last_frame_no 更新 ──────────────────────────────
        self.last_frame_no = frame_data.get("frame_no", self.last_frame_no)

        # ── reward_sum ──────────────────────────────────────
        reward_sum = sum(v for k, v in reward_dict.items() if k != "reward_sum")
        reward_dict["reward_sum"] = reward_sum

        self.prev = cur.copy()
        return reward_dict

    def get_episode_stats(self):
        avg_forward = self.forward_acc / max(self.forward_count, 1)
        survive_ratio = 1.0 if self.alive else round(
            self.last_frame_no / max(self.last_frame_no, 1), 4
        )
        return {
            "death_by_hero": self.death_by_hero,
            "death_by_tower": self.death_by_tower,
            "skill_0_usage": self.skill_usage[0],
            "skill_1_usage": self.skill_usage[1],
            "skill_2_usage": self.skill_usage[2],
            "summoner_usage": self.summoner_usage,
            "minion_kills": self.minion_kills,
            "monster_kills": self.monster_kills,
            "last_hit_count": self.last_hit_count,
            "avg_forward_value": round(avg_forward, 4),
            "destroy_tower_flag": self.destroy_tower_flag,
            "survive_ratio": survive_ratio,
            "death_frame_no": self.last_frame_no if not self.alive else 0,
            "heal_efficiency": round(self.heal_smart_count / max(self.heal_count, 1), 3),
            "flash_efficiency": round(self.flash_smart_count / max(self.flash_count, 1), 3),
            "tower_dive_count": self.tower_dive_count,
            "tower_dive_death": self.tower_dive_death,
            "idle_ratio": round(self.idle_frames / max(self.last_frame_no, 1), 3),
            "dirj_skill2_hit_ratio": round(
                self.dirj_skill2_hit_count / max(self.dirj_skill2_use_count, 1), 3
            ),
            "dirj_skill2_use_count": self.dirj_skill2_use_count,
            "dirj_cleanse_count": self.dirj_cleanse_count,
            "lane_arrival_frame": self.lane_arrival_frame,
            "luban_combo_count": self.luban_combo_count,
            "safe_cake_eat_count": self.safe_cake_eat_count,
        }

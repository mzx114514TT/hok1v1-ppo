#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


SOLDIER_SUB_TYPES = (24, 25, 26, 27)  # MELEE, RANGED, CANNON, SUPER

# Pushing past the midline (forward_ratio > 0.5) without ally minion cover is
# risky on 墨家机关道; modulate the forward reward by ally proximity.
FORWARD_COVER_RANGE = 4000.0
FORWARD_NO_COVER_MULT = 0.4
FORWARD_FULL_COVER_MULT = 1.0

# Tower diving risk: continuous range + ally minion cover
TOWER_RANGE = 7000.0
TOWER_COVER_RANGE = 4000.0  # ally minions within this dist of tower → cover
TOWER_COVER_MIN = 0.3       # risk multiplier floor when ally minions are right at tower

# Skill combo: reward Skill→NormalAttack transitions
COMBO_RANGE = 6000.0        # must be within this dist of enemy for combo to count
COMBO_PER_SKILL = 0.3       # reward per skill cast in combat range


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self._last_main_hero_pos = None
        self._last_main_hp = None
        self._last_main_hp_rate = None
        self._last_enemy_hp_rate = None
        self._trade_accumulator = 0.0
        self._last_skill_cds = {}
        self._combo_accumulator = 0.0

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero = None
        enemy_hero = None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        # Get both defense towers
        # 获取双方防御塔
        main_tower, enemy_tower = None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == 21:
                    main_tower = organ
            else:
                if organ_subtype == 21:
                    enemy_tower = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Tower health points
            # 塔血量
            if reward_name == "tower_hp_point":
                if main_tower and enemy_tower:
                    reward_struct.cur_frame_value = (
                        1.0 * main_tower["hp"] / max(main_tower["max_hp"], 1)
                        - 1.0 * enemy_tower["hp"] / max(enemy_tower["max_hp"], 1)
                    )
                else:
                    reward_struct.cur_frame_value = 0.0
            # Own tower absolute HP rate
            # 己方防御塔血量（非零和）
            elif reward_name == "own_tower_hp_point":
                if main_tower:
                    reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / max(main_tower["max_hp"], 1)
                else:
                    reward_struct.cur_frame_value = 0.0
            # Hero health points
            # 英雄血量
            elif reward_name == "hp_point":
                if main_hero and enemy_hero:
                    reward_struct.cur_frame_value = (
                        1.0 * main_hero["hp"] / max(main_hero["max_hp"], 1)
                        - 1.0 * enemy_hero["hp"] / max(enemy_hero["max_hp"], 1)
                    )
                else:
                    reward_struct.cur_frame_value = 0.0
            # Kill count
            # 击杀数
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero.get("kill_count", 0) if main_hero else 0
            # Death count
            # 死亡数
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero.get("dead_count", 0) if main_hero else 0
            # Money
            # 经济
            elif reward_name == "money":
                if main_hero and enemy_hero:
                    reward_struct.cur_frame_value = (
                        main_hero["money"] - enemy_hero["money"]
                    ) / 5000.0
                else:
                    reward_struct.cur_frame_value = 0.0
            # Experience
            # 经验
            elif reward_name == "exp":
                if main_hero and enemy_hero:
                    reward_struct.cur_frame_value = (
                        self._calc_exp_progress(main_hero)
                        - self._calc_exp_progress(enemy_hero)
                    )
                else:
                    reward_struct.cur_frame_value = 0.0
            # Last hit
            # 补刀数
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = main_hero.get("last_hit", 0) if main_hero else 0
            # Low HP penalty (continuous gradient with recovery detection)
            # 低血量惩罚（连续梯度 + 恢复检测）
            elif reward_name == "low_hp_penalty":
                if main_hero:
                    hp_ratio = main_hero["hp"] / max(main_hero["max_hp"], 1)
                    if hp_ratio >= 0.4:
                        reward_struct.cur_frame_value = 0.0
                    elif hp_ratio >= 0.25:
                        # Linear gradient 0→0.6 across 0.25–0.4
                        reward_struct.cur_frame_value = (0.4 - hp_ratio) / 0.15 * 0.6
                    else:
                        # Critical: check if HP is recovering (recalling/healing)
                        if self._last_main_hp is not None and main_hero["hp"] > self._last_main_hp + 5:
                            reward_struct.cur_frame_value = 0.3
                        else:
                            reward_struct.cur_frame_value = 1.0
                else:
                    reward_struct.cur_frame_value = 0.0
            # Under tower risk penalty (continuous + minion-cover exemption)
            # 越塔风险惩罚（连续化 + 兵线抗塔豁免）
            elif reward_name == "under_tower_risk":
                if camp == self.main_hero_camp and main_hero and enemy_tower:
                    hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
                    tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
                    dist = math.dist(hero_pos, tower_pos)
                    tower_risk = max(0.0, 1.0 - dist / TOWER_RANGE)
                    hp_ratio = main_hero["hp"] / max(main_hero["max_hp"], 1)
                    hp_risk = max(0.0, 1.0 - hp_ratio / 0.5)
                    # Ally minions near tower reduce risk (they tank tower shots)
                    ally_cover = self._ally_near_enemy_tower(enemy_tower, frame_data)
                    cover_mult = TOWER_COVER_MIN + (1.0 - TOWER_COVER_MIN) * (1.0 - ally_cover)
                    reward_struct.cur_frame_value = tower_risk * hp_risk * cover_mult
                else:
                    reward_struct.cur_frame_value = 0.0
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower, frame_data)
            # Kiting: reward moving while near enemy (encourages stutter-step)
            # 走A奖励：靠近敌人同时保持移动
            elif reward_name == "kiting":
                if main_hero and enemy_hero and main_hero.get("hp", 0) > 0:
                    cur_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
                    enemy_pos = (enemy_hero["location"]["x"], enemy_hero["location"]["z"])
                    dist = math.sqrt((cur_pos[0]-enemy_pos[0])**2 + (cur_pos[1]-enemy_pos[1])**2)
                    near_enemy = max(0.0, 1.0 - dist / 6000.0)

                    moved = 0.0
                    if self._last_main_hero_pos is not None:
                        dx = cur_pos[0] - self._last_main_hero_pos[0]
                        dz = cur_pos[1] - self._last_main_hero_pos[1]
                        move_dist = math.sqrt(dx*dx + dz*dz)
                        moved = min(move_dist / 300.0, 1.0)

                    reward_struct.cur_frame_value = near_enemy * moved
                else:
                    reward_struct.cur_frame_value = 0.0
            # Trade / skill-hit: reward favorable HP exchanges
            # 换血/技能命中奖励：成功换血时给予正反馈
            elif reward_name == "trade":
                if camp == self.main_hero_camp and main_hero and enemy_hero and self._last_main_hp_rate is not None:
                    cur_main_rate = main_hero["hp"] / max(main_hero["max_hp"], 1)
                    cur_enemy_rate = enemy_hero["hp"] / max(enemy_hero["max_hp"], 1)
                    main_delta = cur_main_rate - self._last_main_hp_rate
                    enemy_delta = cur_enemy_rate - self._last_enemy_hp_rate
                    max_delta = max(abs(main_delta), abs(enemy_delta))
                    if max_delta > 0.02:
                        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
                        enemy_pos = (enemy_hero["location"]["x"], enemy_hero["location"]["z"])
                        dist = math.dist(hero_pos, enemy_pos)
                        if dist < 8000:
                            trade_raw = main_delta - enemy_delta
                            trade_raw = max(-1.0, min(1.0, trade_raw))
                            self._trade_accumulator += trade_raw
                reward_struct.cur_frame_value = self._trade_accumulator
            # Skill combo: reward using skills in combat range (enables Skill→NormalAttack)
            # 技能连招奖励：战斗中释放技能时给予奖励（鼓励技能→普攻连招）
            elif reward_name == "combo":
                if camp == self.main_hero_camp and main_hero and enemy_hero:
                    casts = 0
                    for slot in main_hero.get("skill_slot_list", []):
                        idx = slot["slot_index"]
                        cd = slot["cool_down"]
                        prev_cd = self._last_skill_cds.get(idx, 0)
                        # Skill went on cooldown this frame (was cast)
                        if cd > prev_cd + 0.1:
                            casts += 1
                    if casts > 0:
                        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
                        enemy_pos = (enemy_hero["location"]["x"], enemy_hero["location"]["z"])
                        dist = math.dist(hero_pos, enemy_pos)
                        if dist < COMBO_RANGE:
                            self._combo_accumulator += COMBO_PER_SKILL * casts
                reward_struct.cur_frame_value = self._combo_accumulator

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower, frame_data):
        if not main_hero or not main_tower or not enemy_tower:
            return 0.0
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["location"]["x"],
            main_hero["location"]["z"],
        )
        total_dist = math.dist(main_tower_pos, enemy_tower_pos)
        if total_dist == 0:
            return 0.0
        dist_to_main = math.dist(hero_pos, main_tower_pos)
        # Forward ratio: 0 = at own tower, 1 = at enemy tower
        # 前进比例：0=在己方塔下，1=在敌方塔下
        forward_ratio = dist_to_main / total_dist
        # Adjust by HP: lower HP reduces forward incentive
        # 根据血量调整：血量低时减少前压奖励
        hp_ratio = main_hero["hp"] / max(main_hero["max_hp"], 1)
        safe_forward = forward_ratio * min(hp_ratio + 0.3, 1.0)

        # Past midline: scale by ally minion cover
        # 过中线：按友方小兵掩护程度打折
        if forward_ratio > 0.5:
            ally_cover = self._ally_minion_cover(hero_pos, main_hero["camp"], frame_data)
            cover_mult = FORWARD_NO_COVER_MULT + (
                FORWARD_FULL_COVER_MULT - FORWARD_NO_COVER_MULT
            ) * ally_cover
            safe_forward *= cover_mult
        return safe_forward

    # Return [0,1] indicating ally minion proximity around the hero.
    # 0 = no ally minions, 1 = ally minion right next to hero.
    def _ally_minion_cover(self, hero_pos, main_camp, frame_data):
        closest = float("inf")
        for npc in frame_data.get("npc_states", []):
            if npc.get("sub_type") not in SOLDIER_SUB_TYPES:
                continue
            if npc.get("camp") != main_camp:
                continue
            npc_pos = (npc["location"]["x"], npc["location"]["z"])
            d = math.dist(hero_pos, npc_pos)
            if d < closest:
                closest = d
        if closest == float("inf"):
            return 0.0
        return max(0.0, 1.0 - closest / FORWARD_COVER_RANGE)

    # Return [0,1] indicating ally minion proximity around enemy tower.
    # 0 = no ally minions near tower, 1 = ally minion right next to tower.
    def _ally_near_enemy_tower(self, enemy_tower, frame_data):
        tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        closest = float("inf")
        for npc in frame_data.get("npc_states", []):
            if npc.get("sub_type") not in SOLDIER_SUB_TYPES:
                continue
            if npc.get("camp") != self.main_hero_camp:
                continue
            npc_pos = (npc["location"]["x"], npc["location"]["z"])
            d = math.dist(tower_pos, npc_pos)
            if d < closest:
                closest = d
        if closest == float("inf"):
            return 0.0
        return max(0.0, 1.0 - closest / TOWER_COVER_RANGE)

    # Calculate experience progress for a hero
    # 计算英雄的经验进度
    def _calc_exp_progress(self, hero):
        if not hero:
            return 0.0
        level = hero.get("level", 1)
        exp = hero.get("exp", 0)
        max_exp = self.m_each_level_max_exp.get(level, 2000)
        return level + exp / max_exp

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

        # Track main hero position for kiting reward
        # 记录己方英雄位置，用于走A奖励
        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                self._last_main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                self._last_main_hp = hero["hp"]
                self._last_main_hp_rate = hero["hp"] / max(hero["max_hp"], 1)
                self._last_skill_cds.clear()
                for slot in hero.get("skill_slot_list", []):
                    self._last_skill_cds[slot["slot_index"]] = slot["cool_down"]
            else:
                self._last_enemy_hp_rate = hero["hp"] / max(hero["max_hp"], 1)

    # Get game phase weight multiplier based on hero level
    # 根据英雄等级获取游戏阶段权重倍数
    def _get_phase_weights(self, level):
        if level <= 4:
            # Early game: encourage leaving fountain + L2/L4 power-spike racing
            return {
                "money": 3.0, "exp": 5.0, "kill": 0.5,
                "forward": 1.5, "tower_hp_point": 0.8, "last_hit": 1.5,
                "combo": 0.3,
            }
        elif level <= 8:
            return {
                "money": 1.0, "exp": 1.0, "kill": 1.5,
                "forward": 1.0, "tower_hp_point": 1.2, "last_hit": 1.0,
                "combo": 1.0,
            }
        else:
            return {
                "money": 0.8, "exp": 0.5, "kill": 1.2,
                "forward": 1.5, "tower_hp_point": 1.5, "last_hit": 0.8,
                "combo": 1.2,
            }

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        # Get main hero level for phase-based weighting
        # 获取主英雄等级用于阶段权重
        main_hero_level = 1
        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_hero_level = hero.get("level", 1)
                break
        phase_weights = self._get_phase_weights(main_hero_level)

        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name in ["kill", "death", "last_hit"]:
                # Non-zero-sum rewards: use main camp delta directly
                # 非零和奖励：直接使用己方变化量
                main_cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                main_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                delta = main_cur - main_last
                if reward_name in ["death", "low_hp_penalty", "under_tower_risk"]:
                    reward_struct.value = -delta
                else:
                    reward_struct.value = delta
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # Apply phase-based dynamic weighting
            # 应用基于游戏阶段的动态权重
            effective_weight = reward_struct.weight * phase_weights.get(reward_name, 1.0)
            weight_sum += effective_weight
            reward_sum += reward_struct.value * effective_weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum

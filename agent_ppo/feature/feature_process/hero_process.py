#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 21
        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):
        self.generate_hero_info_list(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        return main_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_hp_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_mp_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero.get("max_mp", 0) > 0:
            value = hero.get("mp", 0) / hero.get("max_mp", 1)
        vector_feature.append(value)

    def get_level(self, hero, vector_feature, feature_name):
        value = hero.get("level", 1)
        vector_feature.append(value)

    def get_move_speed(self, hero, vector_feature, feature_name):
        value = hero.get("move_speed", 0)
        vector_feature.append(value)

    def get_phy_attack(self, hero, vector_feature, feature_name):
        value = hero.get("phy_attack", 0)
        vector_feature.append(value)

    def get_phy_defense(self, hero, vector_feature, feature_name):
        value = hero.get("phy_defense", 0)
        vector_feature.append(value)

    def get_attack_speed(self, hero, vector_feature, feature_name):
        value = hero.get("attack_speed", 0)
        vector_feature.append(value)

    def get_phy_suck(self, hero, vector_feature, feature_name):
        value = hero.get("phy_suck", 0)
        vector_feature.append(value)

    def get_cool_down_reduce(self, hero, vector_feature, feature_name):
        value = hero.get("cool_down_reduce", 0)
        vector_feature.append(value)

    # Known revive / cheat-death items: 贤者的庇护(1683) / 名刀(1132) / 辉月(1157) / 血魔(1124)
    REVIVE_ITEM_IDS = {1683, 1132, 1157, 1124}

    def get_has_revive_item(self, hero, vector_feature, feature_name):
        has_revive = 0
        for slot in hero.get("equip_slot_list", []):
            equip = slot.get("equip", {})
            if equip.get("equip_id", 0) in self.REVIVE_ITEM_IDS:
                has_revive = 1
                break
        vector_feature.append(has_revive)

    def get_money(self, hero, vector_feature, feature_name):
        value = hero.get("money", 0)
        vector_feature.append(value)

    def get_exp_progress(self, hero, vector_feature, feature_name):
        level = hero.get("level", 1)
        exp = hero.get("exp", 0)
        max_exp_map = {1: 160, 2: 298, 3: 446, 4: 524, 5: 613, 6: 713, 7: 825, 8: 950, 9: 1088, 10: 1240, 11: 1406, 12: 1585, 13: 1778, 14: 1984}
        max_exp = max_exp_map.get(level, 2000)
        value = level + exp / max_exp
        vector_feature.append(value)

    def _get_skill_cd_from_slot(self, hero, slot_index):
        slots = hero.get("skill_slot_list", [])
        for slot in slots:
            if slot.get("slot_index") == slot_index:
                max_cd = slot.get("max_cool_down", 1)
                cd = slot.get("cool_down", 0)
                if max_cd > 0:
                    return cd / max_cd
                return 0.0
        return 0.0

    def get_skill1_cd(self, hero, vector_feature, feature_name):
        value = self._get_skill_cd_from_slot(hero, 0)
        vector_feature.append(value)

    def get_skill2_cd(self, hero, vector_feature, feature_name):
        value = self._get_skill_cd_from_slot(hero, 1)
        vector_feature.append(value)

    def get_skill3_cd(self, hero, vector_feature, feature_name):
        value = self._get_skill_cd_from_slot(hero, 2)
        vector_feature.append(value)

    def get_summon_cd(self, hero, vector_feature, feature_name):
        value = self._get_skill_cd_from_slot(hero, 4)
        vector_feature.append(value)

    def _cal_dist(self, pos1, pos2):
        return math.sqrt((pos1["x"] - pos2["x"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2)

    def get_dist_to_enemy(self, hero, vector_feature, feature_name):
        enemy = None
        for e in self.enemy_camp_hero_dict.values():
            enemy = e
            break
        if enemy and hero and hero.get("hp", 0) > 0 and enemy.get("hp", 0) > 0:
            dist = self._cal_dist(hero["location"], enemy["location"])
            vector_feature.append(min(dist, 30000))
        else:
            vector_feature.append(30000)

    def get_passive_stack(self, hero, vector_feature, feature_name):
        stack = 0
        max_stack = 5
        passive_skills = hero.get("passive_skill_list", [])
        for ps in passive_skills:
            skill_id = ps.get("skill_id", 0)
            if skill_id in [11200, 13300]:
                stack = ps.get("stack_count", 0)
                if skill_id == 11200:
                    max_stack = 4
                elif skill_id == 13300:
                    max_stack = 5
                break
        normalized_stack = stack / max_stack if max_stack > 0 else 0
        vector_feature.append(normalized_stack)

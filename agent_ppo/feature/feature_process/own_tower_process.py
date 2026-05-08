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


class OwnTowerProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_hero_info = None
        self.get_tower_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.own_tower_feature_config)
        self.one_unit_feature_num = 3
        self.unit_buff_num = 1

    def get_tower_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "own_tower_feature_config.ini")
        self.config.read(config_path)

        self.own_tower_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.own_tower_feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_tower(self, frame_state):
        # Find main hero first — needed by distance_to_hero
        self.main_hero_info = None
        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == self.main_camp:
                self.main_hero_info = hero
                break

        own_tower = None
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == 21 and npc.get("camp") == self.main_camp:
                own_tower = npc
                break

        vector_feature = []
        if own_tower and self.main_hero_info:
            self._generate_tower_feature(own_tower, vector_feature)
        else:
            self._no_tower_feature(vector_feature)

        return vector_feature

    def _generate_tower_feature(self, tower, vector_feature):
        for feature_name, feature_func in self.feature_func_map.items():
            value = []
            feature_func(tower, value, feature_name)
            if feature_name not in self.map_feature_to_norm:
                assert False
            for k in value:
                norm_func, *params = self.map_feature_to_norm[feature_name]
                normalized_value = norm_func(k, *params)
                if isinstance(normalized_value, list):
                    vector_feature.extend(normalized_value)
                else:
                    vector_feature.append(normalized_value)

    def _no_tower_feature(self, vector_feature):
        for _ in range(self.unit_buff_num * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_hp_rate(self, tower, vector_feature, feature_name):
        value = 0.0
        if tower.get("max_hp", 0) > 0:
            value = tower["hp"] / tower["max_hp"]
        vector_feature.append(value)

    def is_alive(self, tower, vector_feature, feature_name):
        value = 1.0 if tower.get("hp", 0) > 0 else 0.0
        vector_feature.append(value)

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def distance_to_hero(self, tower, vector_feature, feature_name):
        if self.main_hero_info:
            tower_pos = tower["location"]
            hero_pos = self.main_hero_info["location"]
            dist = self.cal_dist(tower_pos, hero_pos)
            vector_feature.append(min(dist, 30000))
        else:
            vector_feature.append(30000)

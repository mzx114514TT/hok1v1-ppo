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


SOLDIER_SUB_TYPES = {24, 25, 26, 27}  # MELEE, RANGED, CANNON, SUPER

# Distance over which a soldier is no longer considered relevant for last-hit
# awareness. ~6000 units roughly matches normal-attack engagement range on the
# 墨家机关道 lane; further candidates are treated as "no candidate".
LAST_HIT_DIST_RANGE = 6000.0


class SoldierProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_hero_info = None
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self._subtype_verified = False
        self.get_soldier_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.soldier_feature_config)

        # Three soldier slots:
        #   A: nearest enemy soldier  -> 4 dims (rel_x, rel_z, hp_rate, exists)
        #   B: nearest ally soldier   -> 4 dims (rel_x, rel_z, hp_rate, exists)
        #   C: lowest-hp enemy in range (last-hit candidate) -> 2 dims (hp_rate, dist_norm)
        self.slot_feature_dims = {
            "enemy_near": 4,
            "ally_near": 4,
            "last_hit": 2,
        }
        self.total_feature_num = sum(self.slot_feature_dims.values())

    def get_soldier_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "soldier_feature_config.ini")
        self.config.read(config_path)

        self.soldier_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.soldier_feature_config.append(f"{feature}:{config}")

    def process_vec_soldier(self, frame_state):
        # Verify sub_types once per episode
        if not self._subtype_verified:
            subtypes_seen = set()
            for npc in frame_state.get("npc_states", []):
                subtypes_seen.add(npc.get("sub_type"))
            print(f"[SUB_TYPE_CHECK] NPC sub_types: {sorted(subtypes_seen)}")
            self._subtype_verified = True

        self.main_hero_info = None
        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == self.main_camp:
                self.main_hero_info = hero
                break

        if self.main_hero_info is None:
            return [0.0] * self.total_feature_num

        hero_pos = self.main_hero_info["location"]

        nearest_enemy = None
        nearest_enemy_dist = float("inf")
        nearest_ally = None
        nearest_ally_dist = float("inf")
        last_hit_target = None
        last_hit_score = float("inf")  # lower hp -> better candidate
        last_hit_dist = float("inf")

        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") not in SOLDIER_SUB_TYPES:
                continue
            npc_pos = npc["location"]
            dist = math.sqrt(
                (hero_pos["x"] - npc_pos["x"]) ** 2 + (hero_pos["z"] - npc_pos["z"]) ** 2
            )
            is_enemy = npc.get("camp") != self.main_camp

            if is_enemy:
                if dist < nearest_enemy_dist:
                    nearest_enemy_dist = dist
                    nearest_enemy = npc
                if dist <= LAST_HIT_DIST_RANGE:
                    max_hp = npc.get("max_hp", 0)
                    hp_rate = npc["hp"] / max_hp if max_hp > 0 else 1.0
                    # prefer lower hp; break ties by closer distance
                    score = hp_rate + dist / (LAST_HIT_DIST_RANGE * 1000.0)
                    if score < last_hit_score:
                        last_hit_score = score
                        last_hit_target = npc
                        last_hit_dist = dist
            else:
                if dist < nearest_ally_dist:
                    nearest_ally_dist = dist
                    nearest_ally = npc

        vector_feature = []
        self._emit_unit_features(nearest_enemy, vector_feature)
        self._emit_unit_features(nearest_ally, vector_feature)
        self._emit_last_hit_features(last_hit_target, last_hit_dist, vector_feature)
        return vector_feature

    def _emit_unit_features(self, soldier, vector_feature):
        if soldier is None:
            self._append_norm("rel_x", 0.0, vector_feature)
            self._append_norm("rel_z", 0.0, vector_feature)
            self._append_norm("hp_rate", 0.0, vector_feature)
            self._append_norm("exists", 0, vector_feature)
            return

        soldier_x = soldier["location"]["x"]
        soldier_z = soldier["location"]["z"]
        hero_x = self.main_hero_info["location"]["x"]
        hero_z = self.main_hero_info["location"]["z"]
        x_diff = soldier_x - hero_x
        z_diff = soldier_z - hero_z
        if self.transform_camp2_to_camp1:
            x_diff = -x_diff
            z_diff = -z_diff

        max_hp = soldier.get("max_hp", 0)
        hp_rate = soldier["hp"] / max_hp if max_hp > 0 else 0.0

        self._append_norm("rel_x", x_diff, vector_feature)
        self._append_norm("rel_z", z_diff, vector_feature)
        self._append_norm("hp_rate", hp_rate, vector_feature)
        self._append_norm("exists", 1, vector_feature)

    def _emit_last_hit_features(self, soldier, dist, vector_feature):
        if soldier is None:
            self._append_norm("last_hit_hp_rate", 0.0, vector_feature)
            self._append_norm("last_hit_dist", LAST_HIT_DIST_RANGE, vector_feature)
            return
        max_hp = soldier.get("max_hp", 0)
        hp_rate = soldier["hp"] / max_hp if max_hp > 0 else 0.0
        self._append_norm("last_hit_hp_rate", hp_rate, vector_feature)
        self._append_norm("last_hit_dist", dist, vector_feature)

    def _append_norm(self, feature_name, raw_value, vector_feature):
        if feature_name not in self.map_feature_to_norm:
            raise KeyError(f"missing soldier feature config: {feature_name}")
        norm_func, *params = self.map_feature_to_norm[feature_name]
        normalized = norm_func(raw_value, *params)
        if isinstance(normalized, list):
            vector_feature.extend(normalized)
        else:
            vector_feature.append(normalized)

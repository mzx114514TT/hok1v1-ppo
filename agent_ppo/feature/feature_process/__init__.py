#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.own_tower_process import OwnTowerProcess
from agent_ppo.feature.feature_process.soldier_process import SoldierProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        self.enemy_hero_process = HeroProcess(enemy_camp)
        self.organ_process = OrganProcess(camp)
        self.own_tower_process = OwnTowerProcess(camp)
        self.soldier_process = SoldierProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        self.enemy_hero_process = HeroProcess(enemy_camp)
        self.organ_process = OrganProcess(camp)
        self.own_tower_process = OwnTowerProcess(camp)
        self.soldier_process = SoldierProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_enemy_hero_feature(self, frame_state):
        return self.enemy_hero_process.process_vec_hero(frame_state)

    def process_own_tower_feature(self, frame_state):
        return self.own_tower_process.process_vec_tower(frame_state)

    def process_soldier_feature(self, frame_state):
        return self.soldier_process.process_vec_soldier(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        enemy_camp_hero_vector_feature = self.process_enemy_hero_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        own_tower_feature = self.process_own_tower_feature(frame_state)
        soldier_feature = self.process_soldier_feature(frame_state)

        # DIAGNOSTIC
        print(f"[FEATURE_DIMS] hero={len(main_camp_hero_vector_feature)} "
              f"enemy={len(enemy_camp_hero_vector_feature)} "
              f"organ={len(organ_feature)} "
              f"tower={len(own_tower_feature)} "
              f"soldier={len(soldier_feature)} "
              f"TOTAL={len(main_camp_hero_vector_feature + enemy_camp_hero_vector_feature + organ_feature + own_tower_feature + soldier_feature)}")

        feature = (
            main_camp_hero_vector_feature
            + enemy_camp_hero_vector_feature
            + organ_feature
            + own_tower_feature
            + soldier_feature
        )

        return feature

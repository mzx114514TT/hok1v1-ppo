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
from agent_ppo.feature.feature_process.enhanced_features import (
    ENHANCED_DIM_TOTAL,
    MoneyTracker,
    build_enhanced_features,
)
from agent_ppo.feature.cc_state_dict import Info, _camp_int
from agent_ppo.feature.cc_obs_builder import ObsBuilder, DIM_ALL


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.camp_int = _camp_int(camp)
        self.hero_process = HeroProcess(camp)
        enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        self.enemy_hero_process = HeroProcess(enemy_camp)
        self.organ_process = OrganProcess(camp)
        self.own_tower_process = OwnTowerProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.money_tracker = MoneyTracker()
        self.info = Info()
        self.obs_builder = ObsBuilder()

    def reset(self, camp):
        self.camp = camp
        self.camp_int = _camp_int(camp)
        self.hero_process = HeroProcess(camp)
        enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        self.enemy_hero_process = HeroProcess(enemy_camp)
        self.organ_process = OrganProcess(camp)
        self.own_tower_process = OwnTowerProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.money_tracker.reset()
        self.obs_builder.reset()

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
        # Use Info + ObsBuilder pipeline (wty-yy adapted, ~3578 dims)
        self.info.update(observation)
        feat = self.obs_builder.build_observation(self.info)
        return feat.tolist()

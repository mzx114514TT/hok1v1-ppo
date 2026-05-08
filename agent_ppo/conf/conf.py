#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # 22 项奖励权重 — 从齐梓桐模型移植 + CC 原有 tower_hp_point/hp_point 保留语义
    REWARD_WEIGHT_DICT = {
        # ── 基础 12 项（原齐梓桐批次1，对齐 CC 语义）──────────
        "death_penalty": 3.0,
        "hp_diff": 1.0,
        "last_hit": 2.0,
        "money_diff": 1.0,
        "exp_diff": 0.5,
        "level_diff": 0.5,
        "hurt_to_hero": 1.0,
        "hurt_to_tower": 2.0,
        "tower_hp_diff": 3.0,
        "kill_hero": 5.0,
        "destroy_tower": 5.0,
        "forward": 0.5,
        # ── 批次2：发育与生存（7 项）─────────────────────────
        "minion_attack": 0.5,
        "monster_attack": 0.5,
        "heal_smart": 1.0,
        "flash_smart": 1.0,
        "tower_dive_penalty": 1.5,
        "retreat_smart": 1.0,
        "idle_penalty": 0.5,
        # ── 批次3：英雄特定（5 项，鲁班/狄仁杰）───────────────
        "luban_skill_0_clear": 1.0,
        "dirj_skill_2_hit": 2.0,
        "dirj_skill_2_miss": 1.0,
        "lane_arrival": 1.0,
        "early_aggression_penalty": 1.0,
    }
    TIME_SCALE_ARG = 0  # 22 项奖励不使用时间衰减（已按语义设计）
    MODEL_SAVE_INTERVAL = 1800


# ── 数据协议标定值（默认值，运行 NPC_DEBUG/BUTTON_DEBUG 后校准）──
BUTTON_INDEX = {
    "noop_or_attack": 0,
    "skill_1": -1,
    "skill_2": -1,
    "skill_3": -1,
    "summoner": -1,
    "recall": -1,
}

# NPC max_hp 范围（通过 NPC_DEBUG 日志标定后调整）
MINION_MAX_HP_RANGE = (1000, 6000)
MONSTER_MAX_HP_THRESHOLD = 6000

# 距离阈值（游戏坐标系，待校准）
TOWER_ATTACK_RANGE = 1500
FLASH_DISTANCE_THRESHOLD = 1500


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_FEATURE = [63]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        63 + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(63,), (85,)]
    INIT_LEARNING_RATE_START = 3e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 5000
    WARMUP_STEPS = 500
    PPO_EPOCHS = 3
    WEIGHT_DECAY = 1e-4
    BETA_START = 0.05
    BC_WARMUP_STEPS = 600
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.1

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(63 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.99
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 1.0

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

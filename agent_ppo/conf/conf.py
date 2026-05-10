#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # 8 项核心奖励 — 对齐 AAAI 2020 论文设计，删除未校准的推断类奖励
    REWARD_WEIGHT_DICT = {
        "death_penalty": 1.0,    # 论文 -1.0
        "hp_diff": 2.0,          # 论文 2.0
        "last_hit": 0.5,         # 论文 0.5
        "money_diff": 0.006,     # 论文 0.008，零和金币差
        "exp_diff": 0.006,       # 论文 0.008，零和经验差
        "hurt_to_hero": 2.0,     # 保留，鼓励对英雄输出
        "hurt_to_tower": 3.0,    # 保留，鼓励推塔
        "tower_hp_diff": 10.0,   # 论文 10.0（最高权重，推塔是核心目标）
        "kill_hero": 5.0,        # 保留击杀奖励
        "destroy_tower": 10.0,   # 配合 tower_hp_diff
        "forward": 0.5,          # 保留前进引导
        # 以下推断类奖励权重清零（BUTTON_INDEX/距离阈值未校准，易发错信号）
        "level_diff": 0.0,
        "minion_attack": 0.0,
        "monster_attack": 0.0,
        "heal_smart": 0.0,
        "flash_smart": 0.0,
        "tower_dive_penalty": 0.0,
        "retreat_smart": 0.0,
        "idle_penalty": 0.0,
        "luban_passive_combo": 1.0,  # 鲁班连招: 大招→1技能→2技能
        "luban_skill_0_clear": 0.0,
        "dirj_skill_2_hit": 0.0,
        "dirj_skill_2_miss": 0.0,
        "lane_arrival": 0.0,
        "early_aggression_penalty": 0.0,
    }
    TIME_SCALE_ARG = 0
    DUAL_CLIP_C = 3.0  # Dual-clip PPO 下界常数（AAAI 2020 论文值）
    MODEL_SAVE_INTERVAL = 1800

    # ── Debug: 帧数据导出 ──────────────────────────────────
    DEBUG_DUMP_FRAMES = False  # True 时保存每帧 observation 到 JSON
    DEBUG_DUMP_INTERVAL = 1000  # 每隔 N 帧保存一次
    DEBUG_DUMP_MAX_FRAMES = 10  # 每个 episode 最多保存帧数
    DEBUG_DUMP_EPISODES = None  # [1, 2] 只保存指定 episode，None=所有


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
# wty-yy 特征工程适配：Hero(580) + Soldiers(1360) + Organs(338) + Bullets(1300) = 3578
FEATURE_DIM = 3578


class DimConfig:
    DIM_OF_FEATURE = [FEATURE_DIM]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + 85,
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
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (85,)]
    INIT_LEARNING_RATE_START = 3e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 5000
    WARMUP_STEPS = 500
    PPO_EPOCHS = 3
    WEIGHT_DECAY = 1e-4
    BETA_START = 0.1
    BC_WARMUP_STEPS = 200
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

    CLIP_PARAM = 0.2
    DUAL_CLIP_C = 3.0  # Dual-clip PPO 下界常数（AAAI 2020 论文值）

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 64

    # ── Entity Attention (from 齐梓桐 SCAN) ──────────────────
    ENTITY_DIM = 128          # per-entity encoding dimension
    NUM_ENTITY_HEADS = 4      # attention heads for entity self-attention
    NUM_ENTITIES = 12         # self_hero + enemy_hero + 4 our_soldiers + 4 enemy_soldiers + our_tower + enemy_tower
    ENTITY_SELF_HERO = 0
    ENTITY_ENEMY_HERO = 1
    ENTITY_OUR_SOLDIERS = (2, 6)
    ENTITY_ENEMY_SOLDIERS = (6, 10)
    ENTITY_OUR_TOWER = 10
    ENTITY_ENEMY_TOWER = 11

    data_shapes = [
        [(FEATURE_DIM + 85) * 16],
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

    GAMMA = 0.997
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 1.0

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

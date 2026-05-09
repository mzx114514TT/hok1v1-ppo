#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Enhanced feature builder (inspired by wty-yy obs_builder, adapted for CC protocol).

Adds ~160 dims of discretized / one-hot encodings on top of the original 63-dim
flat feature vector. Design goals:
  - Position one-hot so the network can "see" locations without needing to
    infer them from raw coords.
  - HP / CD / money discretization so threshold-based decisions (flee at low
    HP, cast when CD ready) don't have to be learned from a single scalar.
  - Safe defaults: any missing field produces zeros. No dependency on buffs,
    marks, or bullet tracking (those need per-hero calibration).
"""
import math

# Dimension constants
RELATIVE_UNIT = 600
RELATIVE_MAX = 24600  # half-diagonal
REL_HALF_IDX = RELATIVE_MAX // RELATIVE_UNIT + 1  # 42
REL_DIM = REL_HALF_IDX * 2 + 1  # one slot per axis sign, center, bin

WHOLE_UNIT = 5000
WHOLE_MAX = 90000
WHOLE_HALF_BINS = WHOLE_MAX // WHOLE_UNIT  # 18

HP_UNIT = 100
HP_MAX = 2400
HP_BINS = HP_MAX // HP_UNIT + 2  # 26

CD_UNIT = 1
CD_MAX = 10
CD_BINS = CD_MAX // CD_UNIT + 2  # 12

MONEY_UNIT = 20
MONEY_MAX = 300
MONEY_BINS = MONEY_MAX // MONEY_UNIT + 1  # 16

# Per-unit-group dims
DIM_POS_REL = REL_DIM * 2 + 1  # (85+1) but REL_DIM=85, so this is 2*85+1 = 171
DIM_POS_WHOLE = WHOLE_HALF_BINS * 2 + 2  # 38
DIM_POS = DIM_POS_REL + DIM_POS_WHOLE  # 209 (too big for our budget)

# We use a compact version: only absolute-position one-hot + relative distance scalar
DIM_POS_COMPACT = WHOLE_HALF_BINS * 2 + 2 + 1  # 39 per unit (x-bin onehot + z-bin onehot + xratio + zratio + dist_scalar)

DIM_HP = HP_BINS + 1  # 27: (hp_rate_scalar, onehot 26 bins)
DIM_CD = CD_BINS + 2  # 14: (cd_rate_scalar, onehot 12 bins, unusable flag)
DIM_MONEY = MONEY_BINS + 2  # 18: (onehot 16 bins, in_idle_range_flag, total_money_scalar)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _floor(x):
    return math.floor(x) if x > 0 else math.ceil(x)


def _whole_position_onehot(x, z):
    """Absolute position encoded as two axis-aligned one-hots plus sub-bin ratios.

    Returns a list of length DIM_POS_COMPACT.
    """
    feat = [0.0] * DIM_POS_COMPACT
    if x == 100000 or z == 100000:
        return feat
    # x bin
    x_bin = int(_clip(math.floor((x + WHOLE_MAX / 2) / WHOLE_UNIT), 0, WHOLE_HALF_BINS - 1))
    z_bin = int(_clip(math.floor((z + WHOLE_MAX / 2) / WHOLE_UNIT), 0, WHOLE_HALF_BINS - 1))
    feat[x_bin] = 1.0
    feat[WHOLE_HALF_BINS + z_bin] = 1.0
    # sub-bin ratios
    feat[2 * WHOLE_HALF_BINS] = _clip(
        (x + WHOLE_MAX / 2) / WHOLE_UNIT - x_bin, -1, 1
    )
    feat[2 * WHOLE_HALF_BINS + 1] = _clip(
        (z + WHOLE_MAX / 2) / WHOLE_UNIT - z_bin, -1, 1
    )
    return feat


def _relative_distance_scalar(self_pos, other_pos):
    """Normalized distance in [0, 1]."""
    if self_pos is None or other_pos is None:
        return 0.0
    if other_pos.get("x", 100000) == 100000:
        return 0.0
    d = math.dist(
        (self_pos.get("x", 0), self_pos.get("z", 0)),
        (other_pos.get("x", 0), other_pos.get("z", 0)),
    )
    return _clip(d / (RELATIVE_MAX / 2), 0, 1)


def _hp_encoding(hp, max_hp):
    """Scalar rate + one-hot bin. Returns list of length DIM_HP."""
    feat = [0.0] * DIM_HP
    if max_hp <= 0:
        return feat
    feat[0] = _clip(hp / max_hp, 0, 1)
    bin_idx = min(int(math.ceil(max(hp, 0) / HP_UNIT)), HP_BINS - 1)
    feat[1 + bin_idx] = 1.0
    return feat


def _cd_encoding(cd, max_cd, usable):
    """Scalar rate + one-hot bin + unusable flag."""
    feat = [0.0] * DIM_CD
    if max_cd > 0:
        feat[0] = _clip(cd / max_cd, 0, 1)
    bin_idx = min(int(math.ceil(max(cd, 0) / CD_UNIT)), CD_BINS - 1)
    feat[1 + bin_idx] = 1.0
    if not usable:
        feat[-1] = 1.0
    return feat


class MoneyTracker:
    """Tracks money delta between frames for encoding."""

    def __init__(self):
        self.last_money = {"main": None, "enemy": None}

    def reset(self):
        self.last_money = {"main": None, "enemy": None}

    def encode(self, money_total: int, is_enemy: bool) -> list:
        key = "enemy" if is_enemy else "main"
        prev = self.last_money[key]
        if prev is None:
            prev = money_total
        delta = money_total - prev
        self.last_money[key] = money_total

        feat = [0.0] * DIM_MONEY
        if 0 < delta < MONEY_UNIT:
            feat[-2] = 1.0  # passive regen range
        else:
            bin_idx = max(
                0, min(int(math.floor(delta / MONEY_UNIT)), MONEY_BINS - 1)
            )
            feat[bin_idx] = 1.0
        feat[-1] = _clip(money_total / 10000.0, 0, 1)
        return feat


def build_enhanced_features(
    frame_state: dict,
    main_camp: str,
    money_tracker: MoneyTracker,
) -> list:
    """Build the enhanced feature vector.

    Returns a list of floats of length ENHANCED_DIM_TOTAL (see bottom).
    """
    hero_states = frame_state.get("hero_states", [])
    npc_states = frame_state.get("npc_states", [])

    main_hero = None
    enemy_hero = None
    for h in hero_states:
        if h.get("camp") == main_camp:
            main_hero = h
        else:
            enemy_hero = h

    # Find towers (sub_type == 21 is the standard tower marker in this codebase)
    main_tower = None
    enemy_tower = None
    for npc in npc_states:
        if npc.get("sub_type") == 21:
            if npc.get("camp") == main_camp:
                main_tower = npc
            else:
                enemy_tower = npc

    main_pos = main_hero.get("location", {}) if main_hero else None

    feats: list = []

    # --- Main hero: position (39) + HP (27) + 3 skill CDs (14*3 = 42) + money (18)
    if main_hero:
        loc = main_hero.get("location", {})
        feats += _whole_position_onehot(loc.get("x", 100000), loc.get("z", 100000))
        feats += _hp_encoding(main_hero.get("hp", 0), main_hero.get("max_hp", 1))
        for slot_idx in (0, 1, 2):
            cd, max_cd, usable = _get_skill_slot(main_hero, slot_idx)
            feats += _cd_encoding(cd, max_cd, usable)
        feats += money_tracker.encode(main_hero.get("money", 0), is_enemy=False)
    else:
        feats += [0.0] * (DIM_POS_COMPACT + DIM_HP + DIM_CD * 3 + DIM_MONEY)

    # --- Enemy hero: position (39) + HP (27) + relative distance (1) + money (18)
    if enemy_hero:
        eloc = enemy_hero.get("location", {})
        feats += _whole_position_onehot(eloc.get("x", 100000), eloc.get("z", 100000))
        feats += _hp_encoding(enemy_hero.get("hp", 0), enemy_hero.get("max_hp", 1))
        feats.append(_relative_distance_scalar(main_pos, eloc))
        feats += money_tracker.encode(
            enemy_hero.get("money", 0), is_enemy=True
        )
    else:
        feats += [0.0] * (DIM_POS_COMPACT + DIM_HP + 1 + DIM_MONEY)

    # --- Main tower: HP (27) + relative distance (1)
    if main_tower:
        feats += _hp_encoding(main_tower.get("hp", 0), main_tower.get("max_hp", 1))
        feats.append(_relative_distance_scalar(main_pos, main_tower.get("location", {})))
    else:
        feats += [0.0] * (DIM_HP + 1)

    # --- Enemy tower: HP (27) + relative distance (1)
    if enemy_tower:
        feats += _hp_encoding(enemy_tower.get("hp", 0), enemy_tower.get("max_hp", 1))
        feats.append(_relative_distance_scalar(main_pos, enemy_tower.get("location", {})))
    else:
        feats += [0.0] * (DIM_HP + 1)

    return feats


def _get_skill_slot(hero: dict, slot_index: int):
    for slot in hero.get("skill_slot_list", []):
        if slot.get("slot_index") == slot_index:
            return (
                slot.get("cool_down", 0),
                slot.get("max_cool_down", 1),
                bool(slot.get("can_cast", slot.get("enable", True))),
            )
    return 0, 1, False


# Sum:
#   Main hero:   DIM_POS_COMPACT + DIM_HP + DIM_CD*3 + DIM_MONEY = 39 + 27 + 42 + 18 = 126
#   Enemy hero:  DIM_POS_COMPACT + DIM_HP + 1        + DIM_MONEY = 39 + 27 + 1  + 18 = 85
#   Main tower:  DIM_HP + 1 = 28
#   Enemy tower: DIM_HP + 1 = 28
# Total = 126 + 85 + 28 + 28 = 267
ENHANCED_DIM_TOTAL = (
    (DIM_POS_COMPACT + DIM_HP + DIM_CD * 3 + DIM_MONEY)
    + (DIM_POS_COMPACT + DIM_HP + 1 + DIM_MONEY)
    + (DIM_HP + 1)
    + (DIM_HP + 1)
)

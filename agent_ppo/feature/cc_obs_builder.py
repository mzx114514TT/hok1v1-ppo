#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Observation builder adapted from wty-yy's obs_builder for CC 2026 protocol.

Encoding: position one-hot + HP discretization + skill CD discretization
+ money delta discretization + buff/mark flat count. Grouped by unit type
and concatenated into a single flat feature vector.

Dimensions are computed in build_obs_dims() and stored as OBS_DIMS.
"""

import math
import numpy as np
from typing import List, Tuple, Optional

from agent_ppo.feature.cc_state_dict import (
    Info, ActorInfo, HeroInfo, SlotInfo, OrganInfo, SoldierInfo,
    BulletInfo, CakeInfo,
    _sf, _cvt_pos,
)

# ---- Dimension constants ----

RELATIVE_DISTANCE_UNIT = 600
RELATIVE_DISTANCE_MAX = 24600
WHOLE_DISTANCE_UNIT = 5000
WHOLE_DISTANCE_MAX = 90000

HP_UNIT = 100
HP_MAX = 2400

CD_UNIT = 1
CD_MAX = 10

EP_UNIT = 30
EP_MAX = 240

LEVEL_MAX = 15

MONEY_UNIT = 20
MONEY_MAX = 300

SOLDIER_MAX_NUM = 4

# Hero IDs for CC 2026
HERO_CONFIG_IDS = [112, 133]

# Behav modes for hero
HERO_BEHAVES = [
    "State_Dead", "State_Idle", "Direction_Move", "Normal_Attack",
    "State_Revive", "UseSkill_1", "UseSkill_2", "UseSkill_3",
]

# Behav modes for soldier
SOLDIER_BEHAVES = ["State_Dead", "Attack_Path"]

# Soldier config_id groups (wty-yy defaults, may need CC calibration)
SOLDIER_CONFIG_IDS = [[6801, 6804], [6800, 6803], [6802, 6805]]

# Bullet slot types
BULLET_SLOTS = [
    "SLOT_SKILL_0", "SLOT_SKILL_1", "SLOT_SKILL_2",
    "SLOT_SKILL_3", "SLOT_SKILL_VALID",
]
BULLET_MAX_NUM = 10


def _clip(x, lo, hi):
    return max(lo, min(x, hi))


def _floor(x):
    return math.floor(x) if x > 0 else math.ceil(x)


def _ceil(x):
    return math.ceil(x)


# ---- Computed dimensions ----

def _dim_position():
    """Position encoding: relative one-hot + absolute one-hot."""
    # Relative
    max_idx = RELATIVE_DISTANCE_MAX / RELATIVE_DISTANCE_UNIT + 1  # 42
    rpos_dim = int(2 * (max_idx / 2) + 1)  # 43
    x_rpos = rpos_dim * 2 + 1  # 87
    # Absolute
    wpos_bins = int(WHOLE_DISTANCE_MAX / WHOLE_DISTANCE_UNIT)  # 18
    x_wpos = wpos_bins * 2 + 2  # 38
    return x_rpos + x_wpos  # 125


DIM_POSITION = _dim_position()  # 125

# HP encoding: scalar + 26 bins
HP_BINS = int(HP_MAX / HP_UNIT) + 2  # 26
DIM_HP = 1 + HP_BINS  # 27

# CD encoding: scalar + 12 bins + unusable flag
CD_BINS = int(CD_MAX / CD_UNIT) + 2  # 12
DIM_CD = 2 + CD_BINS  # 14

# EP encoding: scalar + 9 bins
EP_BINS = int(EP_MAX / EP_UNIT) + 1  # 9
DIM_EP = 1 + EP_BINS  # 10

# Money encoding: 16 bins + passive flag + total scalar
MONEY_BINS = int(MONEY_MAX / MONEY_UNIT) + 1  # 16
DIM_MONEY = 2 + MONEY_BINS  # 18

# Hero ID encoding: len(HERO_CONFIG_IDS) + 1 (for unknown)
DIM_HERO_ID = len(HERO_CONFIG_IDS) + 1  # 3

# Buff/Mark flat encoding (without per-ID calibration)
DIM_BUFF_SKILLS = 5   # count of buff_skills (0-4 capped)
DIM_BUFF_MARKS = 5    # count of buff_marks (0-4 capped)
DIM_BUFF = DIM_BUFF_SKILLS + DIM_BUFF_MARKS  # 10

# Common unit encoding (position + HP + buff/mark)
DIM_UNIT = DIM_POSITION + DIM_HP + DIM_BUFF  # 125 + 27 + 10 = 162

# Hero encoding: DIM_UNIT + hero_id + behave + ep + 5*CD + level + money + grass + organ_flags
DIM_HERO = (
    DIM_UNIT          # 162
    + DIM_HERO_ID     # 3
    + len(HERO_BEHAVES) + 1  # 9
    + DIM_EP          # 10
    + DIM_CD * 5      # 70 (skill1/2/3/flash/recover)
    + LEVEL_MAX       # 15
    + DIM_MONEY       # 18
    + 1               # grass flag
    + 2               # tower range flag + tower attack target flag
)  # = 162+3+9+10+70+15+18+1+2 = 290

# Soldier encoding: DIM_UNIT + behave + config_id_group + tower_flags
DIM_SOLDIER = (
    DIM_UNIT
    + len(SOLDIER_BEHAVES) + 1  # 3
    + len(SOLDIER_CONFIG_IDS)   # 3
    + 2  # tower range + tower target
)  # = 162+3+3+2 = 170

# Organ tower encoding: DIM_UNIT + target_type(5) + cake_flags(2)
DIM_ORGAN = DIM_UNIT + 5 + 2  # 162+5+2 = 169

# Bullet encoding: slot_type(5) + position(125)
DIM_BULLET = len(BULLET_SLOTS) + DIM_POSITION  # 5+125 = 130

# Totals
DIM_ALL_HEROES = DIM_HERO * 2  # 580
DIM_ALL_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM * 2  # 1360
DIM_ALL_ORGANS = DIM_ORGAN * 2  # 338
DIM_ALL_BULLETS = DIM_BULLET * BULLET_MAX_NUM  # 1300

DIM_ALL = DIM_ALL_HEROES + DIM_ALL_SOLDIERS + DIM_ALL_ORGANS + DIM_ALL_BULLETS
# = 580 + 1360 + 338 + 1300 = 3578


class ObsBuilder:
    """Builds a flat feature vector from a cc_state_dict.Info object."""

    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]

    def process_position(self, position: list) -> list:
        """Encode position as relative one-hot + absolute one-hot."""
        p = position
        if p[0] == Info.UNSEEN_PADDING:
            return [0.0] * DIM_POSITION

        # Relative position (centered at hero position)
        unit_size = RELATIVE_DISTANCE_UNIT
        max_size = RELATIVE_DISTANCE_MAX
        max_idx = int(max_size / unit_size + 1)  # 42
        max_idx = max_idx / 2  # 21
        rpos = [
            int(_clip(_floor((p[i] - self.pos[i]) / unit_size), -max_idx, max_idx) + max_idx)
            for i in range(2)
        ]
        x_dis = _clip(math.dist(p, self.pos) / (max_size / 2), 0, 1)
        rpos_dim = int(2 * max_idx + 1)  # 43
        x_rpos = [0.0] * (rpos_dim * 2 + 1)
        x_rpos[rpos[0]] = x_rpos[rpos[1] + rpos_dim] = 1.0
        x_rpos[-1] = x_dis

        # Absolute (world) position
        unit_size = WHOLE_DISTANCE_UNIT
        max_size = WHOLE_DISTANCE_MAX
        wpos_dim = int(max_size / unit_size)  # 18
        wpos = [
            int(_clip(math.floor((p[i] + max_size / 2) / unit_size), 0, wpos_dim - 1))
            for i in range(2)
        ]
        x_wratio = [
            _clip((p[i] + max_size / 2) / unit_size - wpos[i], -1, 1)
            for i in range(2)
        ]
        x_wpos = [0.0] * (wpos_dim * 2 + 2)
        x_wpos[wpos[0]] = x_wpos[wpos[1] + wpos_dim] = 1.0
        x_wpos[-2] = x_wratio[0]
        x_wpos[-1] = x_wratio[1]

        return x_rpos + x_wpos

    def process_unit(self, info: ActorInfo) -> list:
        """Common unit encoding."""
        x_pos = self.process_position(info.position)
        # HP
        hp_dim = HP_BINS  # 26
        x_hp = [0.0] * DIM_HP  # 27
        max_hp = max(info.hp_max, 1)
        x_hp[0] = _clip(info.hp / max_hp, 0, 1)
        x_hp[1 + min(_ceil(max(info.hp, 0) / HP_UNIT), hp_dim - 1)] = 1.0
        # Buff/Mark flat encoding
        x_buff = [0.0] * DIM_BUFF
        if info.buff is not None:
            n_skills = min(len(info.buff.skill_ids), DIM_BUFF_SKILLS - 1)
            x_buff[n_skills] = 1.0
            n_marks = min(len(info.buff.marks_ids), DIM_BUFF_MARKS - 1)
            x_buff[DIM_BUFF_SKILLS + n_marks] = 1.0
        return x_hp + x_buff + x_pos

    def process_skill(self, info: SlotInfo) -> list:
        """Encode a single skill slot."""
        cd_dim = CD_BINS  # 12
        x_cd = [0.0] * DIM_CD  # 14
        if info.cd_max > 0:
            x_cd[0] = _clip(info.cd / info.cd_max, 0, 1)
        x_cd[1 + min(_ceil(max(info.cd, 0) / CD_UNIT), cd_dim - 1)] = 1.0
        if not info.usable and info.level == 0:
            x_cd[-1] = 1.0
        return x_cd

    def process_money(self, money: int, is_enemy: bool) -> list:
        idx = int(is_enemy)
        delta = money - self.last_money[idx]
        self.last_money[idx] = money
        x_money = [0.0] * DIM_MONEY  # 18
        if 0 < delta < MONEY_UNIT:
            x_money[-2] = 1.0
        else:
            bin_idx = max(0, min(int(math.floor(delta / MONEY_UNIT)), MONEY_BINS - 1))
            x_money[bin_idx] = 1.0
        x_money[-1] = min(money / 10000.0, 1.0)
        return x_money

    def process_organ_flags(self, hero: HeroInfo, enemy_sub_tower) -> list:
        x_organ = [0.0, 0.0]
        if enemy_sub_tower is not None:
            if hasattr(enemy_sub_tower, 'position') and hero.info.position[0] != Info.UNSEEN_PADDING:
                x_organ[0] = float(
                    math.dist(hero.info.position, enemy_sub_tower.position)
                    <= enemy_sub_tower.attack_range
                )
            x_organ[1] = float(hero.info.id == enemy_sub_tower.attack_target)
        return x_organ

    def process_hero(self, hero: HeroInfo, is_enemy: bool, enemy_sub_tower: ActorInfo) -> list:
        # Common unit encoding
        x_unit = self.process_unit(hero.info)
        # Hero config ID
        x_hero_id = [0.0] * DIM_HERO_ID
        if hero.info.config_id in HERO_CONFIG_IDS:
            x_hero_id[HERO_CONFIG_IDS.index(hero.info.config_id)] = 1.0
        else:
            x_hero_id[-1] = 1.0
        # Behav mode
        x_behave = [0.0] * (len(HERO_BEHAVES) + 1)
        idx = -1
        if hero.info.behave in HERO_BEHAVES:
            idx = HERO_BEHAVES.index(hero.info.behave)
        x_behave[idx] = 1.0
        # EP
        ep_dim = EP_BINS
        x_ep = [0.0] * DIM_EP
        ep_max = max(hero.info.ep_max, 1)
        x_ep[0] = _clip(hero.info.ep / ep_max, 0, 1)
        x_ep[1 + min(int(math.floor(hero.info.ep / EP_UNIT)), ep_dim - 1)] = 1.0
        # Skills
        s = hero.skill
        x_skill1 = self.process_skill(s.first)
        x_skill2 = self.process_skill(s.second)
        x_skill3 = self.process_skill(s.thrid)
        x_flash = self.process_skill(s.flash)
        x_recover = self.process_skill(s.recover)
        # Level
        x_level = [0.0] * LEVEL_MAX
        x_level[min(hero.level - 1, LEVEL_MAX - 1)] = 1.0
        # Money
        x_money = self.process_money(hero.money_total, is_enemy)
        # Grass
        x_grass = [float(hero.flag_in_grass)]
        # Organ flags
        x_organ = self.process_organ_flags(hero, enemy_sub_tower)
        return (
            x_unit + x_hero_id + x_behave + x_ep
            + x_skill1 + x_skill2 + x_skill3 + x_flash + x_recover
            + x_level + x_money + x_grass + x_organ
        )

    def process_soldier(self, soldiers: List[ActorInfo], opposed_sub_tower: ActorInfo) -> Tuple[list, list]:
        soldiers = sorted(
            soldiers, key=lambda s: math.dist(s.position, self.pos) if s.position[0] != Info.UNSEEN_PADDING else 1e9
        )
        soldiers = soldiers[:SOLDIER_MAX_NUM]
        x_soldiers = []
        mask = [0.0] * SOLDIER_MAX_NUM
        soldier_dim = DIM_SOLDIER
        for i, soldier in enumerate(soldiers):
            prefix = [0.0] * (soldier_dim - DIM_UNIT)
            base_idx = 0
            # Behav
            idx = len(SOLDIER_BEHAVES)  # dead/unknown
            if soldier.behave in SOLDIER_BEHAVES:
                idx = SOLDIER_BEHAVES.index(soldier.behave)
            prefix[base_idx + idx] = 1.0
            base_idx += len(SOLDIER_BEHAVES) + 1
            # Config ID group
            cid = soldier.config_id
            for j, ids in enumerate(SOLDIER_CONFIG_IDS):
                if cid in ids:
                    prefix[base_idx + j] = 1.0
                    break
            base_idx += len(SOLDIER_CONFIG_IDS)
            # Tower flags
            if opposed_sub_tower is not None:
                if math.dist(soldier.position, opposed_sub_tower.position) <= opposed_sub_tower.attack_range:
                    prefix[base_idx] = 1.0
                if soldier.id == opposed_sub_tower.attack_target:
                    prefix[base_idx + 1] = 1.0
            x_soldiers += prefix + self.process_unit(soldier)
            mask[i] = 1.0
        # Pad
        x_soldiers += [0.0] * (DIM_SOLDIER * SOLDIER_MAX_NUM - len(x_soldiers))
        return x_soldiers, mask

    def process_organ(self, sub_tower, cake, is_enemy: bool) -> list:
        if sub_tower is None:
            return [0.0] * DIM_ORGAN
        x_target = [0.0] * 5  # none/hero/soldier/unknown/unknown
        if sub_tower is not None:
            tgt = sub_tower.attack_target
            if tgt == 0:
                x_target[0] = 1.0
            else:
                tgt_type = self.info.id2type.get(tgt, None)
                if tgt_type == "hero":
                    x_target[1] = 1.0
                elif tgt_type == "soldier":
                    x_target[2] = 1.0
                else:
                    x_target[3] = 1.0
        x_cake = [float(cake is not None), 0.0]
        is_enemy_int = int(is_enemy)
        if cake is not None:
            self.next_cake_frame[is_enemy_int] = self.n_frame + 76 * 30
        else:
            x_cake[1] = min(
                (self.next_cake_frame[is_enemy_int] - self.n_frame) / (75 * 30), 1
            )
        return x_target + x_cake + self.process_unit(sub_tower)

    def process_bullet(self, bullet: BulletInfo) -> list:
        x_slot = [0.0] * len(BULLET_SLOTS)
        if bullet.slot_type in BULLET_SLOTS:
            x_slot[BULLET_SLOTS.index(bullet.slot_type)] = 1.0
        return x_slot + self.process_position(bullet.position)

    def process_bullets(self) -> Tuple[list, list]:
        hero_bullets = self.info.bullets_enemy.hero
        hero_bullets = sorted(
            hero_bullets, key=lambda b: math.dist(b.position, self.pos)
        )
        hero_bullets = hero_bullets[:BULLET_MAX_NUM - 1]
        x_bullets = []
        masks = [0.0] * BULLET_MAX_NUM
        for i, b in enumerate(hero_bullets):
            x_bullets += self.process_bullet(b)
            masks[i] = 1.0
        # Pad missing hero bullets
        x_bullets += [0.0] * ((BULLET_MAX_NUM - 1) * DIM_BULLET - len(x_bullets))
        # One enemy organ bullet
        organ_bullets = self.info.bullets_enemy.organ
        if organ_bullets:
            organ_bullets = sorted(
                organ_bullets, key=lambda b: math.dist(b.position, self.pos)
            )
            x_bullets += self.process_bullet(organ_bullets[0])
            masks[-1] = 1.0
        else:
            x_bullets += [0.0] * DIM_BULLET
        return x_bullets, masks

    def build_observation(self, info: Info) -> np.ndarray:
        """Build the full feature vector from Info."""
        self.info = info
        self.n_frame = info.n_frame
        self.pos = info.hero_our.info.position
        if self.last_money[0] is None:
            self.last_money = [info.hero_our.money_total, info.hero_enemy.money_total]

        # Heroes
        x_hero_our = self.process_hero(info.hero_our, False, info.organ_enemy.sub_tower)
        x_hero_enemy = self.process_hero(info.hero_enemy, True, info.organ_our.sub_tower)

        # Soldiers
        x_soldier_our, _ = self.process_soldier(info.soldiers_our.merge, info.organ_enemy.sub_tower)
        x_soldier_enemy, _ = self.process_soldier(info.soldiers_enemy.merge, info.organ_our.sub_tower)

        # Organs
        x_organ_our = self.process_organ(info.organ_our.sub_tower, info.cake_our, False)
        x_organ_enemy = self.process_organ(info.organ_enemy.sub_tower, info.cake_enemy, True)

        # Bullets
        x_bullet, _ = self.process_bullets()

        x = np.array(
            x_hero_our + x_hero_enemy
            + x_soldier_our + x_soldier_enemy
            + x_organ_our + x_organ_enemy
            + x_bullet,
            np.float32,
        )
        return x


OBS_DIMS = {
    "DIM_POSITION": DIM_POSITION,
    "DIM_HP": DIM_HP,
    "DIM_CD": DIM_CD,
    "DIM_EP": DIM_EP,
    "DIM_MONEY": DIM_MONEY,
    "DIM_UNIT": DIM_UNIT,
    "DIM_HERO": DIM_HERO,
    "DIM_SOLDIER": DIM_SOLDIER,
    "DIM_ORGAN": DIM_ORGAN,
    "DIM_BULLET": DIM_BULLET,
    "DIM_ALL": DIM_ALL,
}

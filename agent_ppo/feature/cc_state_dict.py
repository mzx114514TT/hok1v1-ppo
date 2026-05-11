#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Protocol adaptation — wraps raw env observation dicts into typed Info objects.

================================================================================
The Kaiwu env returns Python dicts whose field names may differ between protocol
versions (e.g. old baseline uses `skill_slot_list` / `cool_down`, new protocol
uses `skill_state.slot_states` / `cooldown`).  This module provides:

  _sf(dict, *candidate_keys)  — safe-first accessor: returns the first non-None
                                value for any of the candidate key names.
  _type_str / _sub_str / _camp_int / _behave_str  — type converters that
                                handle both int enum and string formats.

  Info / HeroInfo / ActorInfo / SkillInfo / SlotInfo / OrganInfo /
  SoldierInfo / BulletInfo / BuffInfo / CakeInfo / BulletsInfo —
                                typed wrappers with consistent attribute names,
                                used by ObsBuilder and DebugAgent.

Key compatibility mappings:
  old field name        →  new protocol name      →  cc_state_dict attribute
  ─────────────────────────────────────────────────────────────────
  skill_slot_list       skill_state.slot_states    SkillInfo(slots)
  cool_down / max_cd    cooldown / cooldown_max    SlotInfo.cd / .cd_max
  enable / can_cast     usable                    SlotInfo.usable
  kill_count            kill_cnt                  HeroInfo.kill_cnt
  mp / max_mp           ep / max_ep               ActorInfo.ep / .ep_max
  camp (string)         camp (int 1=blue,2=red)   ActorInfo.camp (int 0/1)
  actor_type (string)   actor_type (int)          ActorInfo.type (list[str])
  UNSEEN_PADDING        UNSEEN_PADDING            position=[100000, 100000]
================================================================================
"""

import math
from typing import List, Optional

# ---- helpers ----

UNSEEN_PADDING = 100000


def _sf(d: dict, *keys):
    """Safe-first: return first matching key's value from dict, or None."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _cvt_pos(d: dict) -> list:
    """Convert location dict to [x, z] list. Handle unseen padding."""
    x = d.get("x", 0) if isinstance(d, dict) else 0
    z = d.get("z", 0) if isinstance(d, dict) else 0
    if x == UNSEEN_PADDING or x == 100000:
        return [UNSEEN_PADDING, UNSEEN_PADDING]
    return [x, z]


def _pad_direction(d: dict) -> list:
    """Convert forward dict to [x, z] list. Default [0,0] if missing."""
    if not d or not isinstance(d, dict):
        return [0, 0]
    return [d.get("x", 0), d.get("z", 0)]


def _camp_int(camp_val) -> int:
    """Resolve camp to 0/1 (blue=0, red=1)."""
    if isinstance(camp_val, int):
        # 1=blue, 2=red
        return camp_val - 1
    if isinstance(camp_val, str):
        if "PLAYERCAMP" in camp_val:
            return int(camp_val[-1]) - 1
        return 0
    return 0


def _camp_str(camp_val) -> str:
    """Resolve camp to 'PLAYERCAMP_1' (blue) / 'PLAYERCAMP_2' (red)."""
    if isinstance(camp_val, str):
        if "PLAYERCAMP" in camp_val:
            return camp_val
        try:
            camp_val = int(camp_val)
        except (ValueError, TypeError):
            return "PLAYERCAMP_1"
    # int: 0→PLAYERCAMP_1(blue), 1→PLAYERCAMP_2(red), raw 1→blue, raw 2→red
    if camp_val in (0, 1):
        return f"PLAYERCAMP_{camp_val + 1}"
    return f"PLAYERCAMP_{camp_val}"


_ACTOR_TYPE_HERO = "ACTOR_TYPE_HERO"
_ACTOR_TYPE_ORGAN = "ACTOR_TYPE_ORGAN"
_ACTOR_TYPE_MONSTER = "ACTOR_TYPE_MONSTER"
_ACTOR_TYPE_BULLET = "ACTOR_TYPE_BULLET"
_ACTOR_SUB_SOLDIER = "ACTOR_SUB_SOLDIER"
_ACTOR_SUB_TOWER = "ACTOR_SUB_TOWER"
_ACTOR_SUB_CRYSTAL = "ACTOR_SUB_CRYSTAL"
_ACTOR_SUB_TOWER_SPRING = "ACTOR_SUB_TOWER_SPRING"

# Map int-based actor_type (CC protocol) to string (wty-yy compatible)
_ATYPE_MAP = {
    1: _ACTOR_TYPE_HERO,
    2: _ACTOR_TYPE_MONSTER,
    3: _ACTOR_TYPE_ORGAN,
    4: _ACTOR_TYPE_BULLET,
    5: "ACTOR_TYPE_SHENFU",
}
_ASUB_MAP = {
    1: _ACTOR_SUB_SOLDIER,
    2: _ACTOR_SUB_TOWER_SPRING,
    3: _ACTOR_SUB_TOWER,
    4: _ACTOR_SUB_CRYSTAL,
}


def _type_str(actor_type) -> str:
    """Resolve actor_type to string."""
    if isinstance(actor_type, str):
        return actor_type
    return _ATYPE_MAP.get(actor_type, str(actor_type))


def _sub_str(sub_type) -> str:
    """Resolve sub_type to string."""
    if isinstance(sub_type, str):
        return sub_type
    return _ASUB_MAP.get(sub_type, str(sub_type))


def _behave_str(bm) -> str:
    """Resolve behav_mode to string."""
    if isinstance(bm, str):
        return bm
    if bm is None:
        return "State_Idle"
    behave_map = {
        0: "State_Dead",
        1: "State_Idle",
        2: "Direction_Move",
        3: "Normal_Attack",
        4: "State_Revive",
        5: "UseSkill_1",
        6: "UseSkill_2",
        7: "UseSkill_3",
    }
    return behave_map.get(bm, f"behav_{bm}")


# ---- Info classes ----

class BuffSkillInfo:
    def __init__(self, s: dict):
        self.id = _sf(s, "configId", "config_id", "buff_id")
        self.start_time = _sf(s, "startTime", "start_time", 0)
        self.times = _sf(s, "times", 0)
        self.buff_skill_id = self.id


class BuffMarkInfo:
    def __init__(self, s: dict):
        self.actor_id = _sf(s, "origin_actorId", "origin_actorId", "caster_id")
        self.id = _sf(s, "configId", "config_id", "mark_id")
        self.layer = _sf(s, "layer", 0)
        self.buff_mark_id = self.id


class BuffInfo:
    def __init__(self, buff_state: dict):
        self.skills: List[BuffSkillInfo] = []
        self.skill_ids: List[int] = []
        self.marks: List[BuffMarkInfo] = []
        self.marks_ids: List[int] = []
        self.marks_layers: List[int] = []
        if not buff_state or not isinstance(buff_state, dict):
            return
        for s in buff_state.get("buff_skills", []):
            self.skills.append(BuffSkillInfo(s))
            self.skill_ids.append(self.skills[-1].id)
        for s in buff_state.get("buff_marks", []):
            self.marks.append(BuffMarkInfo(s))
            self.marks_ids.append(self.marks[-1].id)
            self.marks_layers.append(self.marks[-1].layer)


class HitTargetInfo:
    def __init__(self, info: dict):
        self.hit_target = _sf(info, "hit_target", "target_id", 0)
        self.skill_id = _sf(info, "skill_id", 0)
        self.slot_type = _sf(info, "slot_type", "")


class ActorInfo:
    """Core unit info extracted from a Hero or NPC state dict."""

    def __init__(self, s: dict):
        self.config_id: int = _sf(s, "config_id", "configId", 0) or 0
        self.id: int = _sf(s, "runtime_id", "runtimeId", 0) or 0
        atype = _sf(s, "actor_type", "actorType", "")
        stype = _sf(s, "sub_type", "subType", "")
        self.type: List[str] = [_type_str(atype), _sub_str(stype)]
        self.camp = _camp_int(_sf(s, "camp", "camp_type", 0))
        self.behave: str = _behave_str(_sf(s, "behav_mode", "behave", 1))
        self.position = _cvt_pos(_sf(s, "location", "pos", {}))
        self.forward = _pad_direction(_sf(s, "forward", "face_dir", {}))
        self.hp = _sf(s, "hp", 0) or 0
        self.hp_max = _sf(s, "max_hp", "maxHp", 1) or 1

        # CC has these flat; wty-yy reads from nested 'values'
        self.ep = _sf(s, "ep", "mp", 0) or 0
        self.ep_max = _sf(s, "max_ep", "max_mp", "maxMp", 1) or 1
        self.hp_recover = _sf(s, "hp_recover", 0) or 0
        self.ep_recover = _sf(s, "ep_recover", 0) or 0
        self.attack_range = _sf(s, "attack_range", 0) or 0
        self.attack_target = _sf(s, "attack_target", 0) or 0
        self.kill_bonus = _sf(s, "kill_income", 0) or 0
        self.sight_range = _sf(s, "sight_area", "sight_range", 0) or 0
        self.hit_target_infos = [
            HitTargetInfo(x)
            for x in (_sf(s, "hit_target_info", "hit_target_infos", []) or [])
        ]
        self.buff_state = _sf(s, "buff_state", {}) or {}
        self.buff = BuffInfo(self.buff_state)


class SlotInfo:
    """Slot state adapted for CC skill_slot_list or skill_state.slot_states."""

    def __init__(self, s: dict):
        self.config_id = _sf(s, "configId", "skill_id", "skill_config_id", 0) or 0
        self.slot_type = _sf(s, "slot_type", "slot_index", "")
        self.level = _sf(s, "level", "skill_level", 0) or 0
        self.usable = _sf(s, "usable", "can_cast", "enable", True)
        if isinstance(self.usable, str):
            self.usable = self.usable in ("True", "true", "1")
        self.cd = _sf(s, "cooldown", "cool_down", 0) or 0
        self.cd_max = _sf(s, "cooldown_max", "max_cool_down", "maxCoolDown", 1) or 1
        self.count = _sf(s, "usedTimes", "used_times", 0) or 0
        self.hit_hero_count = _sf(s, "hitHeroTimes", "hit_hero_times", 0) or 0
        self.flag_used = _sf(s, "succUsedInFrame", "succ_used_in_frame", 0) or 0


class SkillInfo:
    """Skill state extracted from skill_state.slot_states or skill_slot_list.

    Slots are indexed: 0=normal_attack, 1=skill1, 2=skill2, 3=skill3,
    4=recover, 5=flash, 6=back. Missing slots get a dummy SlotInfo.
    """

    _DUMMY_SLOT = SlotInfo({})

    def __init__(self, hero: dict):
        # Try new format first: skill_state.slot_states
        skill_state = _sf(hero, "skill_state", {}) or {}
        slots = _sf(skill_state, "slot_states", None)

        # Fallback to old CC format: skill_slot_list
        if slots is None:
            slots = _sf(hero, "skill_slot_list", [])

        if slots is None:
            slots = []

        # Build index-based lookup
        slot_by_index = {}
        for s in slots:
            si = _sf(s, "configId", None)
            if si is None:
                idx = _sf(s, "slot_index", -1)
                if idx >= 0:
                    slot_by_index[idx] = s
            else:
                # slot_states format: ordered by type. Guess mapping.
                st = _sf(s, "slot_type", "")
                if isinstance(st, str):
                    if "0" in st or "1" in st or "2" in st or "3" in st:
                        pass  # slot_type names, use position

        # If we got slot_states (new format), map by position
        if not slot_by_index and len(slots) >= 7:
            # slot_states are typically ordered: [normal, skill1, skill2, skill3, recover, flash, back]
            self.normal_attack = SlotInfo(slots[0]) if len(slots) > 0 else self._DUMMY_SLOT
            self.first = SlotInfo(slots[1]) if len(slots) > 1 else self._DUMMY_SLOT
            self.second = SlotInfo(slots[2]) if len(slots) > 2 else self._DUMMY_SLOT
            self.thrid = SlotInfo(slots[3]) if len(slots) > 3 else self._DUMMY_SLOT
            self.recover = SlotInfo(slots[4]) if len(slots) > 4 else self._DUMMY_SLOT
            self.flash = SlotInfo(slots[5]) if len(slots) > 5 else self._DUMMY_SLOT
            self.back = SlotInfo(slots[6]) if len(slots) > 6 else self._DUMMY_SLOT
            return

        # Old format with slot_index
        self.normal_attack = SlotInfo(slot_by_index.get(0, {}))
        self.first = SlotInfo(slot_by_index.get(1, {}))
        self.second = SlotInfo(slot_by_index.get(2, {}))
        self.thrid = SlotInfo(slot_by_index.get(3, {}))
        self.recover = SlotInfo(slot_by_index.get(4, {}))
        self.flash = SlotInfo(slot_by_index.get(5, {}))
        self.back = SlotInfo(slot_by_index.get(6, {}))


class HeroInfo:
    """Hero state wrapper."""

    def __init__(self, hero_state: dict):
        s = hero_state
        self.player_id: int = _sf(s, "player_id", "playerId", 0) or 0
        self.info = ActorInfo(s)
        self.skill = SkillInfo(s)
        self.equip_state = _sf(s, "equip_state", {}) or {}
        self.level = _sf(s, "level", 1) or 1
        self.exp = _sf(s, "exp", 0) or 0
        self.money = _sf(s, "money", 0) or 0
        self.money_total = _sf(s, "money_cnt", "moneyCnt", 0)
        if self.money_total == 0:
            self.money_total = self.money
        self.revive_time = _sf(s, "revive_time", 0) or 0
        self.kill_cnt = _sf(s, "kill_cnt", "kill_count", "killCnt", 0) or 0
        self.dead_cnt = _sf(s, "dead_cnt", "dead_count", "deadCnt", 0) or 0
        self.assist_cnt = _sf(s, "assist_cnt", "assist_count", "assistCnt", 0) or 0
        self.kda = [self.kill_cnt, self.dead_cnt, self.assist_cnt]
        self.hurt_total = _sf(s, "total_hurt", "totalHurt", 0) or 0
        self.hurt_hero_total = _sf(s, "total_hurt_to_hero", "totalHurtToHero", 0) or 0
        self.be_hurt_total = _sf(s, "total_be_hurt_by_hero", "totalBeHurtByHero", 0) or 0
        self.flag_in_grass = _sf(s, "is_in_grass", "isInGrass", False) or False
        self.flag_buy_equip = _sf(s, "canBuyEquip", False) or False
        self.passive_skill = _sf(s, "passive_skill", "passive_skill_list", None)


class OrganInfo:
    """Extract towers from npc_states for a given camp."""

    def __init__(self, npc_states: list, camp):
        camp_str = _camp_str(camp)
        self.sub_tower: Optional[ActorInfo] = None
        self.crystal: Optional[ActorInfo] = None
        self.spring: Optional[ActorInfo] = None
        for s in npc_states:
            atype = _type_str(_sf(s, "actor_type", ""))
            if atype != _ACTOR_TYPE_ORGAN:
                continue
            scamp = _camp_str(_sf(s, "camp", 0))
            if scamp != camp_str:
                continue
            stype = _sub_str(_sf(s, "sub_type", ""))
            if stype == _ACTOR_SUB_TOWER:
                self.sub_tower = ActorInfo(s)
            elif stype == _ACTOR_SUB_CRYSTAL:
                self.crystal = ActorInfo(s)
            elif stype == _ACTOR_SUB_TOWER_SPRING:
                self.spring = ActorInfo(s)


class SoldierInfo:
    """Extract minions/soldiers from npc_states."""

    # CC soldier config_ids need calibration. Using wty-yy defaults as starting point.
    # Red: 6801 close, 6800 remote, 6802 cannon. Blue: 6804 close, 6803 remote, 6805 cannon.
    # For CC 2026, these might differ.
    CLOSE_IDS = {6801, 6804}
    REMOTE_IDS = {6800, 6803}
    CAR_IDS = {6802, 6805}

    def __init__(self, npc_states: list, camp_str: str):
        self.close: List[ActorInfo] = []
        self.remote: List[ActorInfo] = []
        self.car: List[ActorInfo] = []
        self.merge: List[ActorInfo] = []
        for s in npc_states:
            atype = _type_str(_sf(s, "actor_type", ""))
            if atype != _ACTOR_TYPE_MONSTER:
                continue
            stype = _sub_str(_sf(s, "sub_type", ""))
            if stype != _ACTOR_SUB_SOLDIER:
                continue
            scamp = _camp_str(_sf(s, "camp", 0))
            if scamp != camp_str:
                continue
            cid = _sf(s, "config_id", "configId", 0)
            ai = ActorInfo(s)
            if cid in self.CLOSE_IDS:
                self.close.append(ai)
            elif cid in self.REMOTE_IDS:
                self.remote.append(ai)
            elif cid in self.CAR_IDS:
                self.car.append(ai)
            else:
                # Unknown soldier type — add to merge anyway
                self.remote.append(ai)  # treat as generic
        self.merge = self.close + self.remote + self.car


class CakeInfo:
    """Extract cakes for a given camp."""

    def __init__(self, cakes, camp):
        self.position = None
        self.camp = None
        if cakes is None:
            return
        for cake in cakes:
            collider = cake.get("collider", {})
            loc = collider.get("location", {}) if isinstance(collider, dict) else {}
            pos_x = loc.get("x", 0) if isinstance(loc, dict) else 0
            cake_camp = int(pos_x > 0)  # rough heuristic
            if camp == cake_camp:
                self.position = _cvt_pos(loc)
                self.camp = cake_camp


class BulletInfo:
    """Bullet info."""

    def __init__(self, bullet: dict, btype: str):
        self.id = _sf(bullet, "runtime_id", 0) or 0
        self.source_id = _sf(bullet, "source_actor", 0) or 0
        self.camp = _camp_int(_sf(bullet, "camp", 0))
        self.slot_type = _sf(bullet, "slot_type", "")
        self.skill_id = _sf(bullet, "skill_id", 0) or 0
        self.position = _cvt_pos(_sf(bullet, "location", "pos", {}))
        self.type = btype


class BulletsInfo:
    """Bullets grouped by source type."""

    def __init__(self, bullets, camp, id2type: dict):
        self.id2type = id2type
        self.soldier: List[BulletInfo] = []
        self.hero: List[BulletInfo] = []
        self.organ: List[BulletInfo] = []
        self.merge: list = []
        if bullets is None:
            return
        camp_str = _camp_str(camp)
        for b in bullets:
            bc = _camp_str(_sf(b, "camp", 0))
            if bc != camp_str:
                continue
            src = _sf(b, "source_actor", 0)
            if src not in id2type:
                continue
            btype = id2type[src]
            bi = BulletInfo(b, btype)
            if btype == "soldier":
                self.soldier.append(bi)
            elif btype == "hero":
                self.hero.append(bi)
            elif btype == "organ":
                self.organ.append(bi)
        self.merge = self.soldier + self.hero + self.organ


class Info:
    """Top-level info extracted from a single camp's observation."""

    UNSEEN_PADDING = UNSEEN_PADDING

    def __init__(self, state_dict: dict = None):
        self.id2type = {}
        self.state_dict = state_dict
        if state_dict is not None:
            self.update(state_dict)

    def update(self, obs: dict):
        s = self.state_dict = obs
        self.player_id = _sf(s, "player_id", 0) or 0
        self.player_camp = _camp_int(_sf(s, "player_camp", "camp", 0))
        self.camp_str = _camp_str(_sf(s, "player_camp", "camp", 0))
        self.game_id = _sf(s, "env_id", "game_id", "")
        self.legal_action = _sf(s, "legal_action", []) or []
        self.sub_action_mask = _sf(s, "sub_action_mask", []) or []
        self.frame_state = fs = _sf(s, "frame_state", {}) or {}
        self.n_frame = _sf(fs, "frame_no", "frameNo", 0) or 0
        self.map_state = _sf(fs, "map_state", False) or False

        hero_states = fs.get("hero_states", [])
        self.hero_our = HeroInfo(
            _find_hero(hero_states, self.player_camp)
        )
        self.hero_enemy = HeroInfo(
            _find_hero(hero_states, 1 - self.player_camp)
        )
        for h in [self.hero_our, self.hero_enemy]:
            self.id2type[h.info.id] = "hero"

        npc_states = fs.get("npc_states", [])
        self.organ_our = OrganInfo(npc_states, self.player_camp)
        self.organ_enemy = OrganInfo(npc_states, 1 - self.player_camp)
        for o in [self.organ_our, self.organ_enemy]:
            if o.sub_tower:
                self.id2type[o.sub_tower.id] = "organ"
            if o.crystal:
                self.id2type[o.crystal.id] = "organ"
            if o.spring:
                self.id2type[o.spring.id] = "organ"

        self.soldiers_our = SoldierInfo(npc_states, self.camp_str)
        self.soldiers_enemy = SoldierInfo(npc_states, _camp_str(1 - self.player_camp))
        for soldiers in [self.soldiers_our, self.soldiers_enemy]:
            for lst in [soldiers.close, soldiers.remote, soldiers.car]:
                for si in lst:
                    self.id2type[si.id] = "soldier"

        # River crab
        self.river_crab = None
        for s in npc_states:
            cid = _sf(s, "config_id", "configId", 0)
            if cid == 6827:
                self.river_crab = ActorInfo(s)
                self.id2type[self.river_crab.id] = "river_crab"

        # Cakes
        cakes = fs.get("cakes")
        self.cake_our = CakeInfo(cakes, self.player_camp)
        self.cake_enemy = CakeInfo(cakes, 1 - self.player_camp)
        if self.cake_our and self.cake_our.position is None:
            self.cake_our = None
        if self.cake_enemy and self.cake_enemy.position is None:
            self.cake_enemy = None

        # Bullets
        bullets = fs.get("bullets")
        self.bullets_our = BulletsInfo(bullets, self.player_camp, self.id2type)
        self.bullets_enemy = BulletsInfo(bullets, 1 - self.player_camp, self.id2type)

        # Dead actions
        fa = fs.get("frame_action", {}) or {}
        self.deads = fa.get("dead_action", [])


def _find_hero(hero_states: list, camp_int: int) -> dict:
    """Find hero by camp int in hero_states list."""
    for h in hero_states:
        hc = _sf(h, "player_camp", "camp", 0)
        hc_int = _camp_int(hc)
        if hc_int == camp_int:
            return h
    # Fallback: check config_id
    for h in hero_states:
        cid = _sf(h, "config_id", "configId", 0)
        if cid in (112, 133):
            return h
    return hero_states[0] if hero_states else {}

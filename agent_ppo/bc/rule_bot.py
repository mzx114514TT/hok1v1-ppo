import math


class RuleBot:
    """Simple rule-based bot for BC warmup.

    Generates actions from observation so the network can learn basic
    behaviors (move forward, attack enemies, use skills, recall at low HP)
    via supervised cross-entropy loss before switching to PPO.
    """

    # Button indices (12 total)
    B_NONE = 0
    B_NOOP = 1
    B_MOVE = 2
    B_ATTACK = 3
    B_SKILL1 = 4
    B_SKILL2 = 5
    B_SKILL3 = 6
    B_RECALL = 9

    # Target indices (9 total)
    T_NONE = 0
    T_ENEMY = 1

    # Direction: 16 buckets, 0 = right(0 deg), CCW, 22.5 deg per step
    NULL_DIR = 15

    COMBAT_RANGE = 8000.0
    RECALL_HP = 0.25

    def __init__(self, camp):
        self.camp = camp

    def act(self, observation):
        frame_state = observation["frame_state"]

        main_hero = None
        enemy_hero = None
        enemy_tower = None

        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == self.camp:
                main_hero = hero
            else:
                enemy_hero = hero

        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == 21 and npc["camp"] != self.camp:
                enemy_tower = npc

        if not main_hero or main_hero.get("hp", 0) <= 0:
            return [self.B_NOOP, self.NULL_DIR, self.NULL_DIR, self.NULL_DIR, self.NULL_DIR, self.T_NONE]

        hp = main_hero["hp"]
        max_hp = max(main_hero["max_hp"], 1)
        hp_ratio = hp / max_hp
        my_pos = (main_hero["location"]["x"], main_hero["location"]["z"])

        enemy_pos = None
        if enemy_hero and enemy_hero.get("hp", 0) > 0:
            enemy_pos = (enemy_hero["location"]["x"], enemy_hero["location"]["z"])

        dist_to_enemy = math.dist(my_pos, enemy_pos) if enemy_pos else 99999.0

        # Critically low HP → recall
        if hp_ratio < self.RECALL_HP and dist_to_enemy > 3000:
            return [self.B_RECALL, self.NULL_DIR, self.NULL_DIR, self.NULL_DIR, self.NULL_DIR, self.T_NONE]

        # Enemy in range → fight: skill first, then attack
        if dist_to_enemy < self.COMBAT_RANGE and enemy_pos:
            available = self._skills_available(main_hero)
            if available:
                slot = available[0]
                btn = self.B_SKILL1 + slot
                d = self._dir_index(my_pos, enemy_pos)
                return [btn, self.NULL_DIR, self.NULL_DIR, d, d, self.T_ENEMY]
            return [self.B_ATTACK, self._dir_index(my_pos, enemy_pos), self.NULL_DIR, self.NULL_DIR, self.NULL_DIR, self.T_ENEMY]

        # No enemy → push toward enemy tower
        if enemy_tower:
            target = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        else:
            target = (my_pos[0] + 10000, my_pos[1])

        d = self._dir_index(my_pos, target)
        return [self.B_MOVE, d, d, self.NULL_DIR, self.NULL_DIR, self.T_NONE]

    def _skills_available(self, hero):
        out = []
        for s in hero.get("skill_slot_list", []):
            idx = s.get("slot_index")
            if idx in (0, 1, 2):
                cd = s.get("cool_down", 0)
                mcd = s.get("max_cool_down", 1)
                if cd == 0 and mcd > 0:
                    out.append(idx)
        return out

    def _dir_index(self, fr, to):
        dx = to[0] - fr[0]
        dz = to[1] - fr[1]
        deg = math.degrees(math.atan2(dz, dx))
        if deg < 0:
            deg += 360.0
        return int(round(deg / 22.5)) % 16

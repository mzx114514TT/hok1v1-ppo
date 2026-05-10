#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Debug: dump raw observation dicts to JSON files at selected frames.

Usage: set `GameConfig.DEBUG_DUMP_FRAMES = True` and run train_test.
Each episode saves frames at configurable intervals to
  D:/Tencent-No2-cc/experiments/debug_frames/ep{episode_id}_f{frame_no}.json
"""

import json
import os
import time


class FrameDumper:
    """Dumps raw observation dicts to disk for offline inspection.

    Controlled by:
      GameConfig.DEBUG_DUMP_FRAMES: bool (master switch)
      GameConfig.DEBUG_DUMP_DIR: str (output root)
      GameConfig.DEBUG_DUMP_INTERVAL: int (save every N frames, default 1000)
      GameConfig.DEBUG_DUMP_MAX_FRAMES: int (cap per episode, default 10)
      GameConfig.DEBUG_DUMP_EPISODES: list[int] (which episodes to dump, empty=all)
    """

    def __init__(self, dump_dir="D:/Tencent-No2-cc/experiments/debug_frames"):
        self.dump_dir = dump_dir
        self.episode_id = 0
        self.frame_count = 0
        self.saved_this_episode = 0
        os.makedirs(dump_dir, exist_ok=True)

    def reset_episode(self, episode_id: int):
        self.episode_id = episode_id
        self.frame_count = 0
        self.saved_this_episode = 0

    def maybe_dump(self, observation: dict, frame_no: int, config) -> bool:
        """Dump observation if conditions are met. Returns True if saved."""
        from agent_ppo.conf.conf import GameConfig

        if not GameConfig.DEBUG_DUMP_FRAMES:
            return False

        interval = getattr(GameConfig, "DEBUG_DUMP_INTERVAL", 1000)
        max_per_episode = getattr(GameConfig, "DEBUG_DUMP_MAX_FRAMES", 10)
        allowed_episodes = getattr(GameConfig, "DEBUG_DUMP_EPISODES", None)

        if allowed_episodes and self.episode_id not in allowed_episodes:
            return False

        if self.saved_this_episode >= max_per_episode:
            return False

        if self.frame_count % interval != 0:
            self.frame_count += 1
            return False

        self.frame_count += 1
        self.saved_this_episode += 1

        fname = f"ep{self.episode_id}_f{frame_no}.json"
        fpath = os.path.join(self.dump_dir, fname)

        def _serialize(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            if isinstance(obj, (list, tuple)):
                return [_serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            return str(obj)

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(_serialize(observation), f, indent=2, ensure_ascii=False)

        return True


# Singleton
_dumper: FrameDumper = None


def get_dumper() -> FrameDumper:
    global _dumper
    if _dumper is None:
        _dumper = FrameDumper()
    return _dumper

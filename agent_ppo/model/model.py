#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Model — PPO actor-critic with Entity Attention + Target Attention.
====================================================================

Architecture pipeline:
  3578-dim flat feature vector
    ├─ Main pathway:  FC(3578→1024) → ResBlock×2 → LayerNorm → 1024
    ├─ Entity pathway: 12 entity slices → per-type MLPs → TransformerEncoder
    │    (128d, 2-layer, sparse mask) → self_hero queries entities → 128
    └─ Fusion:  concat(1024, 128) → FC→512 → TemporalSelfAttn → LSTM(512)
         → Policy/Value split → 6 output heads (5 action + target + value)

Key modules:
  - ResBlock:       Pre-Norm residual block for stable feature extraction.
  - TemporalSelfAttention: Multi-head attention over T=16 frames before LSTM.
                         Degrades to identity at inference (T=1).
  - Entity Attention: Adapted from 齐梓桐 SCAN.  Extracts 12 typed entity
                      tokens from feature vector gaps and runs structured
                      self-attention with sparse mask (same-side soldiers
                      blocked from mutual attention).
  - Target Attention: Adapted from 齐梓桐 SCAN.  LSTM output dot-products
                      against per-type target embeddings from attended
                      entity tokens to produce 9-dim target logits.
  - Dual-clip PPO:   AAAI 2020 — when advantage<0, extra clip to c·A to
                     bound negative gradient in off-policy training.
  - Policy/Value separation: value_fc uses detach() to prevent policy
                             gradients from destabilizing the critic.

Training modes:
  - Train:  T=16, uses TemporalSelfAttention + EntityAttention.
  - BC warmup (first 200 steps): RuleBot targets → cross-entropy loss.
  - Inference: T=1, attention paths degrade to identity/single-token.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config


class Model(nn.Module):
    """PPO Actor-Critic with Entity + Temporal + Target Attention.

    Input:  [feature_vec (BT, 3578), lstm_hidden_init (B, 512), lstm_cell_init (B, 512)]
    Output: train mode → list[6 head logits + value (BT, 1)]
            inference mode → [flattened_logits (BT, 85), value (BT, 1), cell, hidden]
    """

    def __init__(self):
        super(Model, self).__init__()
        # ── Framework-facing config ──────────────────────────────
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS  # 16 train, 1 eval
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE    # 512
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE  # [(3578,), (85,)]
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START   # 3e-4
        self.m_var_beta = Config.BETA_START         # 0.1
        self.log_epsilon = Config.LOG_EPSILON       # 1e-6
        self.label_size_list = Config.LABEL_SIZE_LIST       # [12,16,16,16,16,9]
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY          # 1e-5 for numerical stability
        self.clip_param = Config.CLIP_PARAM          # 0.2
        self.dual_clip_c = Config.DUAL_CLIP_C        # 3.0 (AAAI 2020)
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM  # 64
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]  # 3578
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE  # 512
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])

        # ── Main pathway: flat feature → shared representation ──
        # FC(3578→1024) + 2× Pre-Norm ResBlock → stable 1024-dim encoding
        self.input_proj = make_fc_layer(self.feature_dim, 1024)
        self.input_act = nn.ReLU()
        self.res_block1 = ResBlock(1024)
        self.res_block2 = ResBlock(1024)
        self.concat_norm = nn.LayerNorm(1024)

        # Project from ResBlock output (1024) to LSTM input dim (512)
        self.pre_lstm_proj = make_fc_layer(1024, self.lstm_unit_size)
        self.pre_lstm_act = nn.ReLU()

        # Multi-head self-attention over T=16 time steps (identity at T=1)
        self.temporal_attn = TemporalSelfAttention(
            d_model=self.lstm_unit_size, num_heads=4, dropout=0.1
        )

        # Policy/value branch separation layers (256-dim)
        self.policy_fc = MLP([512, 256], "policy_fc", non_linearity_last=True)
        self.policy_norm = nn.LayerNorm(256)
        self.value_fc = MLP([512, 256], "value_fc", non_linearity_last=True)
        self.value_norm = nn.LayerNorm(256)

        # 1-layer LSTM, no internal dropout (external dropout after)
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1, bias=True, batch_first=True,
            dropout=0, bidirectional=False,
        )
        self.lstm_dropout = nn.Dropout(p=0.1)

        # 6 action heads (button, move_x, move_z, skill_x, skill_z, target)
        self.label_mlp = ModuleDict({
            f"hero_label{i}_mlp": MLP(
                [256, 256, self.label_size_list[i]],
                f"hero_label{i}_mlp",
            )
            for i in range(len(self.label_size_list))
        })

        # ── Entity Attention (齐梓桐 SCAN port) ──────────────────
        # Extracts 12 typed entity tokens from the 3578-dim gap vector,
        # runs structured self-attention with a sparse mask, then lets
        # self_hero query all entities to produce a key_info summary.
        from agent_ppo.conf.conf import Config as Cfg
        self.entity_dim = Cfg.ENTITY_DIM         # 128
        self.num_entities = Cfg.NUM_ENTITIES     # 12

        # Per-entity-type projection MLPs: entity_slice → 256 → 128
        from agent_ppo.feature.cc_obs_builder import DIM_HERO, DIM_SOLDIER, DIM_ORGAN, DIM_RIVER_CRAB
        self.entity_proj_self_hero = nn.Sequential(
            make_fc_layer(DIM_HERO, 256), nn.ReLU(),
            make_fc_layer(256, self.entity_dim), nn.ReLU(),
        )
        self.entity_proj_enemy_hero = nn.Sequential(
            make_fc_layer(DIM_HERO, 256), nn.ReLU(),
            make_fc_layer(256, self.entity_dim), nn.ReLU(),
        )
        self.entity_proj_soldier = nn.Sequential(
            make_fc_layer(DIM_SOLDIER, 256), nn.ReLU(),
            make_fc_layer(256, self.entity_dim), nn.ReLU(),
        )
        self.entity_proj_tower = nn.Sequential(
            make_fc_layer(DIM_ORGAN, 256), nn.ReLU(),
            make_fc_layer(256, self.entity_dim), nn.ReLU(),
        )
        self.entity_proj_crab = nn.Sequential(
            make_fc_layer(DIM_RIVER_CRAB, 256), nn.ReLU(),
            make_fc_layer(256, self.entity_dim), nn.ReLU(),
        )

        # 2-layer TransformerEncoder: 13 tokens attend to each other
        # with a sparse mask blocking same-side soldier internal attention
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.entity_dim, nhead=Cfg.NUM_ENTITY_HEADS,
            dim_feedforward=self.entity_dim * 2, dropout=0.0,
            batch_first=True, activation="relu",
        )
        self.entity_attn = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.register_buffer("entity_sparse_mask", _build_entity_sparse_mask())

        # self_hero (index 0) cross-attends all entities to extract key info
        self.entity_query_attn = nn.MultiheadAttention(
            embed_dim=self.entity_dim, num_heads=Cfg.NUM_ENTITY_HEADS,
            batch_first=True,
        )

        # Fuse ResBlock output (1024) + entity key_info (128) → LSTM input (512)
        self.entity_fusion = make_fc_layer(1024 + self.entity_dim, self.lstm_unit_size)

        # ── Target Attention (齐梓桐 SCAN port) ──────────────────
        # Instead of a flat MLP predicting 9-dim target logits, we encode
        # each target candidate from attended entity tokens, then let the
        # LSTM output query them via dot-product attention.
        self.tar_proj_hero = make_fc_layer(self.entity_dim, self.target_embed_dim)
        self.tar_proj_tower = make_fc_layer(self.entity_dim, self.target_embed_dim)
        self.tar_proj_soldier = make_fc_layer(self.entity_dim, self.target_embed_dim)
        self.tar_proj_crab = make_fc_layer(self.entity_dim, self.target_embed_dim)
        self.tar_query = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)

        # ── Value head ────────────────────────────────────────────
        self.value_mlp = MLP([256, 256, 1], "hero_value_mlp")

    # ---- Entity token extraction ----

    def _extract_entity_tokens(self, feature_vec):
        """Slice 12 entity tokens from the 3578-dim feature vector gaps.

        Feature layout (from cc_obs_builder.ObsBuilder.build_observation):
          offset 0-289:     self_hero   (DIM_HERO=290)
          offset 290-579:   enemy_hero  (DIM_HERO=290)
          offset 580-1259:  our_soldiers ×4 (DIM_SOLDIER=170 each)
          offset 1260-1939: enemy_soldiers ×4
          offset 1940-2108: our_tower   (DIM_ORGAN=169)
          offset 2109-2277: enemy_tower (DIM_ORGAN=169)
          offset 2278-3577: bullets ×10 (not extracted)

        Returns: (B, 12, entity_dim) tensor.
        """
        from agent_ppo.feature.cc_obs_builder import (
            DIM_HERO, DIM_SOLDIER, DIM_ORGAN, DIM_RIVER_CRAB, SOLDIER_MAX_NUM,
        )
        S = SOLDIER_MAX_NUM  # 4
        tokens = []
        # self_hero: offset 0
        tokens.append(self.entity_proj_self_hero(feature_vec[:, :DIM_HERO]))
        # enemy_hero: offset DIM_HERO
        off = DIM_HERO
        tokens.append(self.entity_proj_enemy_hero(feature_vec[:, off:off + DIM_HERO]))
        # our soldiers ×4
        off = DIM_HERO * 2
        for _ in range(S):
            tokens.append(self.entity_proj_soldier(feature_vec[:, off:off + DIM_SOLDIER]))
            off += DIM_SOLDIER
        # enemy soldiers ×4
        for _ in range(S):
            tokens.append(self.entity_proj_soldier(feature_vec[:, off:off + DIM_SOLDIER]))
            off += DIM_SOLDIER
        # our tower
        tokens.append(self.entity_proj_tower(feature_vec[:, off:off + DIM_ORGAN]))
        off += DIM_ORGAN
        # enemy tower
        tokens.append(self.entity_proj_tower(feature_vec[:, off:off + DIM_ORGAN]))
        off += DIM_ORGAN
        # river crab
        tokens.append(self.entity_proj_crab(feature_vec[:, off:off + DIM_RIVER_CRAB]))
        return torch.stack(tokens, dim=1)  # (B, 13, entity_dim)

    # ---- Forward pass ----

    def forward(self, data_list, inference=False):
        """Main forward pass with dual-pathway architecture.

        data_list = [feature_vec, lstm_hidden_init, lstm_cell_init]
          feature_vec:      (BT, 3578) flat feature vector
          lstm_hidden_init: (B, 512)  initial LSTM hidden state
          lstm_cell_init:   (B, 512)  initial LSTM cell state
        """
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        # ── Main pathway: FC → ResBlocks → LayerNorm ────────
        fc_public_result = self.input_act(self.input_proj(feature_vec))
        fc_public_result = self.res_block1(fc_public_result)
        fc_public_result = self.res_block2(fc_public_result)
        fc_public_result = self.concat_norm(fc_public_result)

        # ── Entity Attention pathway ─────────────────────────
        # Extract 12 entity tokens → TransformerEncoder (sparse mask)
        # → self_hero queries entities → key_info summary (128-dim)
        entity_tokens = self._extract_entity_tokens(feature_vec)
        entity_tokens = self.entity_attn(
            entity_tokens, mask=self.entity_sparse_mask
        )
        self_query = entity_tokens[:, :1, :]  # self_hero @ index 0
        key_info, _ = self.entity_query_attn(
            query=self_query, key=entity_tokens, value=entity_tokens
        )
        key_info = key_info.squeeze(1)  # (B, entity_dim)

        # ── Fusion: concat ResBlock output + entity context ──
        # (B, 1024+128) → FC(1152→512) → ReLU
        fused = torch.cat([fc_public_result, key_info], dim=-1)
        fused = self.pre_lstm_act(self.entity_fusion(fused))

        # ── LSTM with temporal attention ─────────────────────
        # Train: T=16, frames packed in batch dim. Eval: T=1.
        T = self.lstm_time_steps
        BT = fused.shape[0]
        B = BT // T
        lstm_input = fused.view(B, T, -1)
        lstm_input = self.temporal_attn(lstm_input)

        h_0 = lstm_hidden_init.view(B, -1).unsqueeze(0).contiguous()
        c_0 = lstm_cell_init.view(B, -1).unsqueeze(0).contiguous()
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h_0, c_0))
        # Save final states for next call (agent passes them back)
        self.lstm_hidden_output = h_n
        self.lstm_cell_output = c_n
        lstm_features = lstm_out.contiguous().view(BT, -1)
        lstm_features = self.lstm_dropout(lstm_features)

        # ── Policy/Value branch separation ───────────────────
        # policy_fc: LSTM output → 256-dim (trained by policy loss)
        # value_fc:  LSTM output.detach() → 256-dim (trained by value loss)
        # detach() prevents policy gradient from flowing into the critic
        policy_feature = self.policy_fc(lstm_features)
        policy_feature = self.policy_norm(policy_feature)
        value_feature = self.value_fc(lstm_features.detach())
        value_feature = self.value_norm(value_feature)

        # ── 5 action heads (button, move_x, move_z, skill_x, skill_z) ──
        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            result_list.append(
                self.label_mlp[f"hero_label{label_index}_mlp"](policy_feature)
            )

        # ── Target Attention head (label index 5, 9-dim output) ──
        target_logits = self._compute_target_attention(
            entity_tokens, lstm_features
        )
        result_list.append(target_logits)

        # ── Value head ───────────────────────────────────────
        value_result = self.value_mlp(value_feature)
        result_list.append(value_result)

        logits = torch.cat(result_list[:-1], 1)
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list

    # ---- Target attention ----

    def _compute_target_attention(self, entity_tokens, lstm_features):
        """Dot-product target attention using attended entity embeddings.

        Builds 9 target candidate embeddings from entity tokens:
          0: padding       4: enemy soldier 1      8: padding
          1: enemy hero    5: enemy soldier 2
          2: enemy tower   6: enemy soldier 3
          3: enemy soldier 0  7: crab (padding, no crab in CC)

        LSTM output is projected to a query vector (64-dim), then
        dot-products with all 9 target embeddings to produce logits.
        """
        from agent_ppo.conf.conf import Config as Cfg
        E_HERO = Cfg.ENTITY_ENEMY_HERO        # 1
        E_SOL = Cfg.ENTITY_ENEMY_SOLDIERS      # (6, 10)
        E_TOWER = Cfg.ENTITY_ENEMY_TOWER       # 11
        E_CRAB = Cfg.ENTITY_RIVER_CRAB         # 12

        # Project entity tokens to target embedding space (64-dim)
        hero_emb = self.tar_proj_hero(entity_tokens[:, E_HERO, :])
        tower_emb = self.tar_proj_tower(entity_tokens[:, E_TOWER, :])
        soldier_embs = self.tar_proj_soldier(entity_tokens[:, E_SOL[0]:E_SOL[1], :])
        crab_emb = self.tar_proj_crab(entity_tokens[:, E_CRAB, :])

        # Stack 9 target candidates: pad slots use fixed small embedding
        pad_emb = torch.full_like(hero_emb, 0.1)
        target_embs = torch.stack([
            pad_emb,                    # 0: padding
            hero_emb,                   # 1: enemy hero
            tower_emb,                  # 2: enemy tower
            soldier_embs[:, 0, :],      # 3: enemy soldier 0
            soldier_embs[:, 1, :],      # 4: enemy soldier 1
            soldier_embs[:, 2, :],      # 5: enemy soldier 2
            soldier_embs[:, 3, :],      # 6: enemy soldier 3
            crab_emb,                   # 7: river crab
            pad_emb,                    # 8: padding
        ], dim=1)  # (B, 9, 64)

        # LSTM output → query → dot-product → unnormalized logits
        query = self.tar_query(lstm_features).unsqueeze(1)  # (B, 1, 64)
        logits = torch.bmm(query, target_embs.transpose(1, 2)).squeeze(1)
        return logits

    # ---- Loss computation ----

    def compute_loss(self, data_list, rst_list):
        """PPO loss with dual-clip + Huber value loss + entropy bonus."""
        # Unpack and reshape data from the reverb buffer
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3:3 + len(self.label_size_list)]
        for i in range(len(self.label_size_list)):
            usq_label_list[i] = (
                usq_label_list[i].reshape(-1, self.data_split_shape[3 + i]).long()
            )

        old_label_probability_list = data_list[
            3 + len(self.label_size_list):3 + 2 * len(self.label_size_list)
        ]
        for i in range(len(self.label_size_list)):
            old_label_probability_list[i] = old_label_probability_list[i].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + i]
            )

        usq_weight_list = data_list[
            3 + 2 * len(self.label_size_list):3 + 3 * len(self.label_size_list)
        ]
        for i in range(len(self.label_size_list)):
            usq_weight_list[i] = usq_weight_list[i].reshape(
                -1, self.data_split_shape[3 + 2 * len(self.label_size_list) + i],
            )

        # Squeeze trailing dimensions
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [w.squeeze(dim=1) for w in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        # Split feature vector from legal action mask
        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        fla_shape = list(self.seri_vec_split_shape[1])
        fla_shape.insert(0, -1)
        legal_action_flag_list = torch.split(
            split_feature_legal_action.reshape(fla_shape), self.label_size_list, dim=1
        )

        # Value loss: Huber (smooth L1) — less sensitive to outlier frames than MSE
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = F.smooth_l1_loss(fc2_value_result_squeezed, reward)
        new_advantage = reward - fc2_value_result_squeezed

        label_probability_list = []
        epsilon = 1e-5

        # Policy loss: dual-clip PPO (AAAI 2020)
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue

            final_log_p = torch.tensor(0.0)
            boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
            one_hot = nn.functional.one_hot(
                label_list[task_index].long(), self.label_size_list[task_index]
            )

            # Legal action masking: subtract huge value for illegal actions
            legal_mask = (1 - legal_action_flag_list[task_index]) * boundary
            logits_clamped = torch.clamp(
                label_result[task_index]
                - torch.max(label_result[task_index] - legal_mask, dim=1, keepdim=True).values,
                -boundary, 1,
            )
            exp_logits = (
                legal_action_flag_list[task_index] * torch.exp(logits_clamped) + self.min_policy
            )
            label_probability = exp_logits / exp_logits.sum(1, keepdim=True)
            label_probability_list.append(label_probability)

            # Probability ratio: π_new / π_old
            policy_p = (one_hot * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (one_hot * old_label_probability_list[task_index] + epsilon).sum(1)
            old_policy_log_p = torch.log(old_policy_p)
            final_log_p = final_log_p + policy_log_p - old_policy_log_p
            ratio = torch.exp(final_log_p)  # importance sampling ratio

            # Dual-clip: when advantage < 0, extra lower bound c·A
            surr1 = ratio * advantage
            surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            surr_min = torch.minimum(surr1, surr2)
            dual_clipped = torch.where(
                advantage < 0,
                torch.maximum(surr_min, self.dual_clip_c * advantage),
                surr_min,
            )
            temp_policy_loss = -torch.sum(
                dual_clipped * weight_list[task_index].float() * frame_is_train
            ) / torch.maximum(
                torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0)
            )
            self.policy_cost = self.policy_cost + temp_policy_loss

        # Entropy loss: H = -Σ p·log(p), masked to legal actions only
        # entropy_cost is negative (more negative = more entropy / exploration)
        entropy_idx = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy = -torch.sum(
                    label_probability_list[entropy_idx]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[entropy_idx] + epsilon),
                    dim=1,
                )
                temp_entropy = -torch.sum(
                    temp_entropy * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0)
                )
                entropy_loss_list.append(temp_entropy)
                entropy_idx += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0))

        self.entropy_cost = sum(entropy_loss_list)
        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def compute_bc_loss(self, data_list, rst_list):
        """Behavior Cloning loss for warmup phase (first BC_WARMUP_STEPS=200).

        Uses cross-entropy on RuleBot action labels + Huber value loss.
        No policy gradient — pure supervised learning to bootstrap.
        """
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])

        usq_label_list = data_list[3:3 + len(self.label_size_list)]
        for i in range(len(self.label_size_list)):
            usq_label_list[i] = (
                usq_label_list[i].reshape(-1, self.data_split_shape[3 + i]).long()
            )

        reward = usq_reward.squeeze(dim=1)
        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        # Cross-entropy on each action head (compared to RuleBot's actions)
        bc_loss = torch.tensor(0.0)
        for i in range(len(self.label_size_list)):
            labels = usq_label_list[i].squeeze(dim=1)
            bc_loss = bc_loss + F.cross_entropy(label_result[i], labels)

        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = F.smooth_l1_loss(fc2_value_result_squeezed, reward)

        self.loss = bc_loss + self.value_cost
        return self.loss, [self.loss, [self.value_cost, bc_loss, torch.tensor(0.0)]]

    # ---- Mode switching ----

    def set_train_mode(self):
        """Switch to training: T=16 frames per batch, dropout active."""
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        """Switch to inference: T=1 frame, dropout disabled, attention=identity."""
        self.lstm_time_steps = 1
        self.eval()


# ====================================================================
#  Layer constructors
# ====================================================================

def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Orthogonal-init Linear layer with optional zero bias."""
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


# ====================================================================
#  Building-block modules
# ====================================================================

class ResBlock(nn.Module):
    """Pre-Norm Residual Block.

    Input → LayerNorm → Linear → ReLU → Dropout → Linear → +input (skip).
    Pre-norm design (LayerNorm before FC) improves training stability.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = make_fc_layer(dim, dim)
        self.fc2 = make_fc_layer(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out + residual


class TemporalSelfAttention(nn.Module):
    """Self-attention over the T time steps before LSTM.

    Shape: [B, T, d_model] → [B, T, d_model].
    When T=1 (inference), returns identity (no attention overhead).
    Allows each frame to attend to all other frames in the 16-frame window,
    capturing global temporal dependencies before recurrence.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x  # T=1: skip attention
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


def _build_entity_sparse_mask():
    """12×12 sparse attention mask blocking same-type-same-side internal attention.

    Token layout (matching Config.ENTITY_* constants):
      0:  self_hero             6-9:  enemy_soldiers (4)
      1:  enemy_hero            10:   our_tower
      2-5: our_soldiers (4)     11:   enemy_tower

    Blocked pairs (-inf = no attention):
      - our soldiers (2-5) don't attend to each other
      - enemy soldiers (6-9) don't attend to each other
    All other pairs are allowed (can cross-attend between camps/types).
    """
    N = 13
    mask = torch.zeros(N, N)
    mask[2:6, 2:6] = float('-inf')   # our soldiers internal
    mask[6:10, 6:10] = float('-inf')  # enemy soldiers internal
    # crab (12) can freely attend to/be attended by all entities
    return mask


class MLP(nn.Module):
    """Stacked Linear + ReLU with orthogonal init.

    fc_feat_dim_list[i] → fc_feat_dim_list[i+1].
    If non_linearity_last=True, final layer also gets activation (for
    intermediate representations). Otherwise final layer is linear.
    """

    def __init__(
        self, fc_feat_dim_list: List[int], name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i+1}", fc)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i+1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

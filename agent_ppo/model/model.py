#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # NETWORK DIM
        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])

        # 共享特征提取层：MLP + 残差块 [feature_dim -> 1024]
        self.input_proj = make_fc_layer(self.feature_dim, 1024)
        self.input_act = nn.ReLU()
        self.res_block1 = ResBlock(1024)
        self.res_block2 = ResBlock(1024)
        self.concat_norm = nn.LayerNorm(1024)

        # 1024 -> LSTM input projection (LSTM stays 512 for memory efficiency)
        self.pre_lstm_proj = make_fc_layer(1024, self.lstm_unit_size)
        self.pre_lstm_act = nn.ReLU()

        # 时间轴自注意力（在 LSTM 之前，捕捉 16 帧间全局依赖）
        # Temporal self-attention before LSTM (T=16 frames, d_model=512)
        self.temporal_attn = TemporalSelfAttention(d_model=self.lstm_unit_size, num_heads=4, dropout=0.1)

        # 策略和价值分支分离
        self.policy_fc = MLP([512, 256], "policy_fc", non_linearity_last=True)
        self.policy_norm = nn.LayerNorm(256)
        self.value_fc = MLP([512, 256], "value_fc", non_linearity_last=True)
        self.value_norm = nn.LayerNorm(256)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        # Dropout after LSTM to prevent overfitting on selfplay opponents
        self.lstm_dropout = nn.Dropout(p=0.1)

        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [256, 256, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list))
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)

        self.value_mlp = MLP([256, 256, 1], "hero_value_mlp")

        self.target_embed_mlp = make_fc_layer(self.target_embed_dim, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        result_list = []

        # public concat -> ResBlocks -> project -> TemporalAttn -> LSTM
        # 特征提取：输入投射 -> 两个残差块 -> LayerNorm -> 投射到 LSTM 维度 -> 时间注意力
        fc_public_result = self.input_act(self.input_proj(feature_vec))
        fc_public_result = self.res_block1(fc_public_result)
        fc_public_result = self.res_block2(fc_public_result)
        fc_public_result = self.concat_norm(fc_public_result)
        fc_public_result = self.pre_lstm_act(self.pre_lstm_proj(fc_public_result))

        # LSTM: reshape [B*T, 512] -> [B, T, 512], temporal attention, run, reshape back
        # 训练时 T=lstm_time_steps=16, 推理时 T=1
        T = self.lstm_time_steps
        BT = fc_public_result.shape[0]
        B = BT // T
        lstm_input = fc_public_result.view(B, T, -1)

        # Temporal self-attention over T frames before LSTM
        # 时间轴自注意力：让每帧看到所有帧的上下文（训练时有效，推理时 T=1 退化为恒等）
        lstm_input = self.temporal_attn(lstm_input)

        # Initial hidden/cell state shape: [num_layers=1, B, lstm_unit_size]
        # 初始 hidden/cell 来自 batch 的第一帧状态
        h_0 = lstm_hidden_init.view(B, -1).unsqueeze(0).contiguous()
        c_0 = lstm_cell_init.view(B, -1).unsqueeze(0).contiguous()

        lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h_0, c_0))

        # Save final LSTM state for inference (transported back to agent)
        # 保存 LSTM 最终状态，供推理时透传给 agent
        self.lstm_hidden_output = h_n
        self.lstm_cell_output = c_n

        # Flatten time dim back: [B, T, 512] -> [B*T, 512]
        lstm_features = lstm_out.contiguous().view(BT, -1)
        lstm_features = self.lstm_dropout(lstm_features)

        # 策略分支和价值分支分离（value detach 防止策略梯度干扰价值网络）
        # Policy and value branches separation (value detaches to stabilize training)
        policy_feature = self.policy_fc(lstm_features)
        policy_feature = self.policy_norm(policy_feature)

        value_feature = self.value_fc(lstm_features.detach())
        value_feature = self.value_norm(value_feature)

        # output label
        # 输出标签
        for label_index, label_dim in enumerate(self.label_size_list[:]):
            label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](policy_feature)
            result_list.append(label_mlp_out)

        # output value
        # 输出价值
        value_result = self.value_mlp(value_feature)
        result_list.append(value_result)

        # prepare for infer graph
        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net (Huber for robustness to outlier frames)
        # 值网络的损失（Huber 对异常帧不敏感）
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = F.smooth_l1_loss(fc2_value_result_squeezed, reward)
        new_advantage = reward - fc2_value_result_squeezed

        # for entropy loss calculate
        # 用于熵损失计算
        label_probability_list = []

        epsilon = 1e-5

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * frame_is_train
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * frame_is_train), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * frame_is_train)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def compute_bc_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        reward = usq_reward.squeeze(dim=1)
        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        # Cross-entropy BC loss for each action head
        bc_loss = torch.tensor(0.0)
        for i in range(len(self.label_size_list)):
            labels = usq_label_list[i].squeeze(dim=1)
            bc_loss = bc_loss + F.cross_entropy(label_result[i], labels)

        # Value loss (Huber)
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = F.smooth_l1_loss(fc2_value_result_squeezed, reward)

        self.loss = bc_loss + self.value_cost
        return self.loss, [self.loss, [self.value_cost, bc_loss, torch.tensor(0.0)]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class ResBlock(nn.Module):
    """Pre-Norm Residual Block: LN -> Linear -> ReLU -> Linear + skip."""

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
    """Multi-head self-attention over the T time steps of an LSTM sequence.

    Input/output shape: [B, T, d_model]
    Falls back to identity when T=1 (inference).
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        if x.shape[1] == 1:
            # T=1 at inference: attention is trivial, skip for speed
            return x
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

#!/usr/bin/env python3
# coding=utf-8
"""RNNoise 整网（7 层）的类封装 —— 从 `examples/rnn_unit.py: rnnoise_demo()` 抽出来的。

层实现都在别处：`examples/linear.py`（Conv1D / DenseLayer）、`examples/gru.py`（GRUScratch）。
本文件只负责按 `src/rnn.c: compute_rnn()` 的顺序把它们串起来：

    conv1(65->128) -> conv2(128->384) -> gru1 -> gru2 -> gru3 -> dense_out / vad_dense

用例来自 `scripts/unit_cases/*.json`（真实模型规模的 //16）。

★ 现在只是「能跑」：它**还不是 nn.Module**、隐状态存在实例属性里、forward 里还在打协议行。
  要给 `torch.onnx.export` 用，这几条必须先解决（见 examples/onnx_demo.py 与后续讨论）。
"""

import sys

import torch
from torch import nn as nn
from gru import GRUScratch
from linear import Conv1D, DenseLayer
from utils import emit


class RNNoiseMo(nn.Module):
    """整网：conv1 / conv2 / gru1..3 / dense_out / vad_dense。

    一次吃一帧（`forward(frame, data)`）：conv 的 mem 和三层 GRU 的隐状态都存在
    各自的子对象里，所以**同一个实例必须按帧顺序连续调用**，不能跳帧或复用。

    cases:         examples/loader.py 读出来的用例 {层名: case}
    low_accuracy:  激活档次（True -> 多项式近似 = C 默认；False -> torch 自带）
    out:           协议行输出目标（None 表示静默）
    """

    def __init__(self, cases, low_accuracy=True, out=sys.stdout):
        super().__init__()
        self.low_accuracy = low_accuracy
        self.out = out

        # conv1 / conv2：每帧输入宽 = 上一层的输出宽
        input_features = cases['conv1']['nb_in']
        nb_out = cases['conv1']['nb_out']

        self.conv1 = Conv1D(input_features * 3, nb_out, cases['conv1'], input_features,
                            low_accuracy=low_accuracy)
        self.conv2 = Conv1D(nb_out * 3, nb_out * 3, cases['conv2'], nb_out,
                            low_accuracy=low_accuracy)

        # gruN：每帧输入宽 = conv2 输出宽 = 隐状态宽 N，门输出宽 = 3N
        # ['layer', 'input_weights_float', 'input_bias', 'recurrent_weights_float',
        #  'recurrent_bias', 'state', 'inputs']
        hidden_size = nb_out * 3
        gru_out_size = hidden_size * 3
        self.gru1 = GRUScratch(hidden_size, hidden_size, gru_out_size,
                               cases['gru1'], layer=cases['gru1']['layer'], detail=False)
        self.gru2 = GRUScratch(hidden_size, hidden_size, gru_out_size,
                               cases['gru2'], layer=cases['gru2']['layer'], detail=False)
        self.gru3 = GRUScratch(hidden_size, hidden_size, gru_out_size,
                               cases['gru3'], layer=cases['gru3']['layer'], detail=False)

        # 输出层：输入宽 = conv2 输出 + 3 层 gru 的隐状态 = 4 x (nb_out*3)
        dense_in = hidden_size * 3 + nb_out * 3
        self.dense = DenseLayer(dense_in, int(dense_in / (1536 / 32)), cases['dense_out'],
                                low_accuracy=low_accuracy)
        self.vad_dense = DenseLayer(dense_in, 1, cases['vad_dense'],
                                    low_accuracy=low_accuracy)

    def forward(self, frame, data):
        """跑一帧。

        frame: 帧号 —— **只用于协议行**，不参与任何计算
        data:  这一帧的输入，长度 = conv1 的输入宽

        返回 (conv1_out, conv2_out, gru1_state, gru2_state, gru3_state,
              dense_in, gains, vad) 八项，顺序和协议行一致。
        注意 gruN.state 是 GRUScratch 自己打出来的（detail=False），
        返回值里那三个 state 只是顺手带出来给调用方用。
        """
        conv1_output = self.conv1(data)

        conv2_output = self.conv2(conv1_output.flatten())

        gru1_state = self.gru1(conv2_output.flatten(), low_accuracy=self.low_accuracy,
                               out=self.out, frame=frame)

        gru2_state = self.gru2(gru1_state, low_accuracy=self.low_accuracy,
                               out=self.out, frame=frame)

        gru3_state = self.gru3(gru2_state, low_accuracy=self.low_accuracy,
                               out=self.out, frame=frame)

        dense_in = torch.cat((conv2_output.flatten(), gru1_state.flatten(),
                              gru2_state.flatten(), gru3_state.flatten()), dim=0)

        # dense
        gains = self.dense(dense_in)

        # vad
        vad = self.vad_dense(dense_in)

        return (conv1_output, conv2_output, gru1_state, gru2_state, gru3_state,
                dense_in, gains, vad)

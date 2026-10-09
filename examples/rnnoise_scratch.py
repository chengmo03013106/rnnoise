#!/usr/bin/env python3
# coding=utf-8
"""RNNoise 整网（7 层）的类封装 —— 从 `examples/rnn_unit.py: rnnoise_demo()` 抽出来的。

层实现都在别处：`examples/linear.py`（Conv1D / DenseLayer）、`examples/gru.py`（GRUMo）。
本文件只负责按 `src/rnn.c: compute_rnn()` 的顺序把它们串起来：

    conv1(65->128) -> conv2(128->384) -> gru1 -> gru2 -> gru3 -> dense_out / vad_dense

用例来自 `scripts/unit_cases/*.json`（真实模型规模的 //16）。

打印：本类**完全不打印**（2026-10-08 起 `GRUMo` 也不再打印）——
`forward` 把 8 项全部返回，协议行由调用方去打，
见 `examples/rnn_unit.py: rnnoise_demo_2()`。这也是能被 trace / 导出的前提。

★ 离 `torch.onnx.export` 的现状（2026-10-09 **逐条实测过**，不是推测）：
  ① （已解决）权重进参数表了：`GRUMo` 的 4 组权重改成 `register_buffer`
     （`W_xzrh` / `b_xzrh` / `W_hzrh` / `b_hzrh`，实测 `state_dict()` 里能看到它们）；
     7 个子层都注册成功（`named_children()` = conv1/conv2/gru1..3/dense/vad_dense），
     `eval()` / `.to()` 也都正常。
  ② 唯一还差的一件事：状态还没进图 —— `Conv1D` / `GRUMo` 本身**已经**是"状态参数进出"
     （`out, mem = net(data, mem)`、`h_out, detail = net(x, h_in)`），但本类又替它们把状态
     存回 `self.conv1_mem` / `self.conv2_mem` / `self.gru1_state..gru3_state`。
     静态图里状态必须显式进出（同 LLM 的 KV cache）：把这 5 个状态一起做成
     `forward` 的参数与返回值即可。
  （已解决·2026-10-08：打印副作用已消除 —— `GRUMo` 不再打印，本类也不再持有 `out`。）
"""

import torch
from torch import nn as nn
from gru import GRUMo
from linear import Conv1D, DenseLayer


class RNNoiseMo(nn.Module):
    """整网：conv1 / conv2 / gru1..3 / dense_out / vad_dense。

    一次吃一帧（`forward(frame, data)`）：conv 的 mem 和三层 GRU 的隐状态都存在
    各自的子对象里，所以**同一个实例必须按帧顺序连续调用**，不能跳帧或复用。

    cases:         examples/loader.py 读出来的用例 {层名: case}
    low_accuracy:  激活档次（True -> 多项式近似 = C 默认；False -> torch 自带）
    """

    def __init__(self, cases, low_accuracy=True):
        super().__init__()
        self.low_accuracy = low_accuracy

        # conv1 / conv2：每帧输入宽 = 上一层的输出宽
        input_features = cases['conv1']['nb_in']
        nb_out = cases['conv1']['nb_out']

        self.conv1 = Conv1D(input_features * 3, nb_out, cases['conv1'], input_features,
                            low_accuracy=low_accuracy)
        self.conv2 = Conv1D(nb_out * 3, nb_out * 3, cases['conv2'], nb_out,
                            low_accuracy=low_accuracy)

        # Conv1D 不再自带状态：mem 由调用方持有，长度 = nb_input - input_size = 2 帧宽。
        # ★ 现在还放在实例属性里 —— 要导出 ONNX 得把它们改成 forward 的输入/输出。
        self.conv1_mem = torch.zeros(input_features * 2, dtype=torch.float32)
        self.conv2_mem = torch.zeros(nb_out * 2, dtype=torch.float32)

        # gruN：每帧输入宽 = conv2 输出宽 = 隐状态宽 N，门输出宽 = 3N
        # ['layer', 'input_weights_float', 'input_bias', 'recurrent_weights_float',
        #  'recurrent_bias', 'state', 'inputs']
        hidden_size = nb_out * 3
        gru_out_size = hidden_size * 3
        # low_accuracy 走构造函数（GRUMo 在 __init__ 里快照 self.sigmoid / self.tanh）
        self.gru1 = GRUMo(hidden_size, hidden_size, gru_out_size, cases['gru1'],
                          low_accuracy=low_accuracy)
        self.gru2 = GRUMo(hidden_size, hidden_size, gru_out_size, cases['gru2'],
                          low_accuracy=low_accuracy)
        self.gru3 = GRUMo(hidden_size, hidden_size, gru_out_size, cases['gru3'],
                          low_accuracy=low_accuracy)

        # 同理，GRUMo 的隐状态也由调用方持有（forward(x, hidden)），本类替它存着。
        # ★ 现在还放在实例属性里 —— 导出 ONNX 时要和三块 mem 一起改成 forward 的输入/输出。
        self.gru1_state = torch.zeros(hidden_size, dtype=torch.float32)
        self.gru2_state = torch.zeros(hidden_size, dtype=torch.float32)
        self.gru3_state = torch.zeros(hidden_size, dtype=torch.float32)

        # 输出层：输入宽 = conv2 输出 + 3 层 gru 的隐状态 = 4 x (nb_out*3)
        dense_in = hidden_size * 3 + nb_out * 3
        self.dense = DenseLayer(dense_in, int(dense_in / (1536 / 32)), cases['dense_out'],
                                low_accuracy=low_accuracy)
        self.vad_dense = DenseLayer(dense_in, 1, cases['vad_dense'],
                                    low_accuracy=low_accuracy)

    def forward(self, frame, data):
        """跑一帧。

        frame: 帧号 —— **只用于日志/定位**，不参与任何计算
        data:  这一帧的输入，长度 = conv1 的输入宽

        返回 (conv1_out, conv2_out, gru1_state, gru2_state, gru3_state,
              dense_in, gains, vad) 八项，顺序和协议行一致。
        **本方法不打印任何东西**：协议行由调用方按这份返回值去打
        （见 `examples/rnn_unit.py: rnnoise_demo_2()`）。
        """
        conv1_output, self.conv1_mem = self.conv1(data, self.conv1_mem)

        conv2_output, self.conv2_mem = self.conv2(conv1_output.flatten(), self.conv2_mem)

        # forward 返回 (新隐状态, 中间量 dict)：这里只取隐状态
        self.gru1_state, _ = self.gru1(conv2_output.flatten(), self.gru1_state)

        self.gru2_state, _ = self.gru2(self.gru1_state, self.gru2_state)

        self.gru3_state, _ = self.gru3(self.gru2_state, self.gru3_state)

        dense_in = torch.cat((conv2_output.flatten(), self.gru1_state.flatten(),
                              self.gru2_state.flatten(), self.gru3_state.flatten()), dim=0)

        # dense
        gains = self.dense(dense_in)

        # vad
        vad = self.vad_dense(dense_in)

        return (conv1_output, conv2_output, self.gru1_state, self.gru2_state,
                self.gru3_state, dense_in, gains, vad)

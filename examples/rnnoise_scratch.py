#!/usr/bin/env python3
# coding=utf-8
"""RNNoise 整网（7 层）的类封装 —— 从 `examples/rnn_unit.py: rnnoise_demo()` 抽出来的。

层实现都在别处：`examples/linear.py`（Conv1D / DenseLayer）、`examples/gru.py`（GRUMo）。
本文件只负责按 `src/rnn.c: compute_rnn()` 的顺序把它们串起来：

    conv1(65->128) -> conv2(128->384) -> gru1 -> gru2 -> gru3 -> dense_out / vad_dense

用例来自 `scripts/unit_cases/*.json`（真实模型规模的 //16）。

打印：本类**完全不打印**（2026-10-08 起 `GRUMo` 也不再打印）——
`forward` 只返回结果，协议行由调用方去打，
见 `examples/rnn_unit.py: rnnoise_demo_2()`。这也是能被 trace / 导出的前提。

★ 状态（Explicit State）—— 2026-10-09 已参数化：
  5 个跨帧状态（conv1/conv2 的 mem + 三层 GRU 的 hidden）全部
  ① 用 `register_buffer` 注册：进 `state_dict()`、`.to()` / `.eval()` 管得住，
     同时是「全 0 初值」的**唯一出处**；
  ② **打包成一个 `state` 参数**在 `forward` 进、出一个新 `state`：`forward` 里不写
     `self.*`，只用传进来的参数，算完把新状态一起返回，由调用方在下一帧喂回。

  静态图没有「对象属性」这个概念（同 LLM 的 KV cache），状态必须显式进出；
  这样 `torch.onnx.export` 才能把它变成 graph 的 inputs / outputs。

  调用形态（`state` 顺序 = conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state）：
      state = net.init_state()                  # 初值（buffer 的 clone）
      state, (gains, vad) = net(data, state)    # 每帧都是同一个写法

  打包成一个参数（而不是 5 个位置参数），是为了不让「顺序」散落到每个调用点 ——
  顺序只在 `init_state()` / `forward` 里各定义一次，改了一处就全对。

  签名里**只有张量**（`data` + 5 个状态）：帧号是"只用于日志的 int"，
  `torch.onnx.export` 不能把它当 graph input，所以它留在调用方的循环里（`enumerate`），
  不进 `forward`。
"""

import torch
from torch import nn as nn
from gru import GRUMo
from linear import Conv1D, DenseLayer


class RNNoiseMo(nn.Module):
    """整网：conv1 / conv2 / gru1..3 / dense_out / vad_dense。

    一次吃一帧，5 个跨帧状态（conv1_mem / conv2_mem / gru1_state / gru2_state /
    gru3_state）**打包成一个 `state` 参数显式进出**：由调用方持有，每帧喂进来、
    再把新的拿回去。初值用 `init_state()` 取（那 5 个 register_buffer 的 clone），
    对象里不存活动状态，所以同一个实例可以跳帧 / 复用。

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
        # register_buffer：既进 state_dict（.to()/.eval() 管得住），又是「全 0 初值」的出处；
        # 真正参与计算的是 forward 的 state 参数里的对应分量（见 forward / init_state）。
        self.register_buffer('conv1_mem', torch.zeros(input_features * 2, dtype=torch.float32))
        self.register_buffer('conv2_mem', torch.zeros(nb_out * 2, dtype=torch.float32))

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

        # 同理，GRUMo 的隐状态也由调用方持有（forward(x, hidden)）。
        # 和两块 conv mem 一样：register_buffer 只负责「初值 + 进 state_dict」，
        # 计算时用 forward 的 state 参数里传进来的那一个。
        self.register_buffer('gru1_state', torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer('gru2_state', torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer('gru3_state', torch.zeros(hidden_size, dtype=torch.float32))

        # 输出层：输入宽 = conv2 输出 + 3 层 gru 的隐状态 = 4 x (nb_out*3)
        dense_in = hidden_size * 3 + nb_out * 3
        self.dense = DenseLayer(dense_in, int(dense_in / (1536 / 32)), cases['dense_out'],
                                low_accuracy=low_accuracy)
        self.vad_dense = DenseLayer(dense_in, 1, cases['vad_dense'],
                                    low_accuracy=low_accuracy)

    def init_state(self):
        """取 5 个状态的初值 —— 顺序与 `forward` 的 `state` 参数、返回值完全一致：

            (conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state)

        唯一出处是那几个 register_buffer（构造时全 0）。这里 **clone 一份**再返回：
        调用方拿到的是自己的副本，之后就算原地改也污染不到 buffer 的初值
        （对张量 `a = b` 只是再绑一个名字；`.detach()` / `.view()` / 切片都不隔离内存）。
        """
        return (self.conv1_mem.clone(), self.conv2_mem.clone(),
                self.gru1_state.clone(), self.gru2_state.clone(),
                self.gru3_state.clone())

    def forward(self,data, state):
        """跑一帧 —— **状态显式进出**，本方法不读写 self 上的任何状态。

        data:  这一帧的输入，长度 = conv1 的输入宽
        state: 上一帧的 5 个状态，顺序
               (conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state)；
               初值用 `init_state()` 取。

        返回 `(state, (gains, vad))`：
            state        更新后的 5 元组，顺序与传进来的完全一致
            (gains, vad) 两个最终输出

        这 7 个张量正好对应 ONNX 图的 outputs（5 个状态 + 2 个输出）；
        conv1_out / conv2_out / dense_in 是纯调试量，不再返回
        （要对比就去看内联版 `all` 单元）。
        **本方法不打印任何东西**：协议行由调用方按这份返回值去打
        （见 `examples/rnn_unit.py: rnnoise_demo_2()`）。
        """
        conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state = state
        conv1_output, conv1_mem = self.conv1(data, conv1_mem)

        conv2_output, conv2_mem = self.conv2(conv1_output.flatten(), conv2_mem)

        # forward 只返回新隐状态（GRUMo 现在不再返回中间量 detail）
        gru1_state = self.gru1(conv2_output.flatten(), gru1_state)

        gru2_state = self.gru2(gru1_state, gru2_state)

        gru3_state = self.gru3(gru2_state, gru3_state)

        dense_in = torch.cat((conv2_output.flatten(), gru1_state.flatten(),
                              gru2_state.flatten(), gru3_state.flatten()), dim=0)

        # dense
        gains = self.dense(dense_in)

        # vad
        vad = self.vad_dense(dense_in)

        return (conv1_mem, conv2_mem, 
            gru1_state, gru2_state, gru3_state), (gains, vad)

#!/usr/bin/env python3
# coding=utf-8
"""线性层 / 卷积层：把 src/nnet.c 里那几个「矩阵乘 + 激活」的算子搬成 PyTorch。

    Linear      一层全连接      y = x @ W + b       （对应 compute_linear()）
    DenseLayer  全连接 + sigmoid                    （整网输出层 dense_out / vad_dense）
    Conv1D      带一帧记忆的全连接 + tanh            （对应 compute_generic_conv1d()）

这三个层被 conv1d / all 两个单元共用（gru.py 不用它们，GRU 的 [z|r|h] 布局
是它自己的事），所以从 rnn_unit.py 抽出来放在这里，只留一份。
本文件另外还装着 conv1d 单元（conv1d_demo）。

★ 唯一**不是单元**的东西是 linear_demo：它没有 C 侧对手、不打协议行、
  scripts/unit_check.py 也不跑它 —— 所以它没进 examples/rnn_unit.py 的 UNITS，
  只能单独跑：

      uv run examples/linear.py [case.json]

同目录的模块之间用**裸 import**：`from utils import emit`、`from loader import CONV_LAYERS`。
"""

import os
import sys

import torch
from torch import nn as nn

from loader import CONV_LAYERS, CaseError, load_unit_cases
from rnnoise_activation import pick     # 高/低精度只换激活函数
from utils import case_tensor, emit


class Linear(nn.Module):
    """一层全连接：y = x @ W + b。对应 C 的 compute_linear()。

    C 的权重布局是 weights[输入 j][输出 i]（sgemv 的 col_stride = nb_outputs），
    和 nn.Linear 的 [输出, 输入] 正好相反 —— 所以这里不套 nn.Linear，
    直接把用例里的扁平数组 reshape 成 [输入, 输出]，forward 里写 x @ W，不做转置。

    推理用不到梯度，所以用 register_buffer 而不是 Parameter（也就不需要 requires_grad）。
    """

    def __init__(self, nb_input, nb_output, case):
        super().__init__()
        self.register_buffer(
            "weights",
            case_tensor(case, "float_weights", nb_input * nb_output, (nb_input, nb_output)),
        )
        self.register_buffer("bias", case_tensor(case, "bias", nb_output))

    def forward(self, x):
        # 一帧本来就是一维；reshape 成 (1, nb_input) 让 @ 当成一个样本算
        x = x.reshape(-1, self.weights.shape[0])
        return x @ self.weights + self.bias


def linear_demo(cases, out=sys.stdout):
    """单层线性层的手工实验：借 conv1 那一层的参数，喂一个手写输入跑一次。

    对应 src/nnet.c: compute_linear()（float_weights 非空时走 sgemv）。
    ``out`` 只影响往哪儿写 —— 这里打的是张量的字符串，**不是协议行**。
    """
    layer, in_size, nb_out = CONV_LAYERS[0]
    net = Linear(in_size, nb_out, cases[layer])
    in_data = torch.tensor([1.0, 0.5, 0.5, 1.0])
    out_data = net(in_data)
    out.write("%s\n" % out_data)


class DenseLayer(nn.Module):
    """整网输出层：线性 + sigmoid（dense_out 出 gains，vad_dense 出 VAD 概率）。

    高/低精度只换激活函数：
        low_accuracy=True  -> rnnoise_activation 的多项式 sigmoid（= C 默认）
        low_accuracy=False -> torch.sigmoid（= C 的 -DHIGH_ACCURACY）

    它自己没有参数（参数都在 self.linear 里），所以不是 nn.Module，只是个可调用对象。
    """

    def __init__(self, nb_input, nb_output, case, low_accuracy=True):
        super().__init__()
        self.linear = Linear(nb_input, nb_output, case)
        self.active = pick(low_accuracy)[0]

    def forward():
        return self.active(self.linear(data))

    def __call__(self, data): # 没有继承 nn.Module 使用这个函数
        return self.active(self.linear(data))


class Conv1D(nn.Module):
    """带记忆的一维卷积：tmp = [mem, input] -> 线性 -> 激活 -> 把 tmp 尾部存回 mem。

    对应 C 的 compute_generic_conv1d()：mem 长度 = nb_input - input_size
    （即 (kernel_size - 1) 帧），每帧把最旧的一帧挤出去。

    高/低精度只换激活函数：
        low_accuracy=True  -> rnnoise_activation 的多项式 tanh（= C 默认）
        low_accuracy=False -> torch.tanh（= C 的 -DHIGH_ACCURACY）

    ★ 必须是 nn.Module 才可调用（`net(x)` 走 nn.Module.__call__ -> forward）。
      裸类只写 forward 是**不能调用**的（会 TypeError: object is not callable）——
      DenseLayer 那种裸类是显式写了 __call__ 才行。
    """

    def __init__(self, nb_input, nb_output, case, input_size, low_accuracy=True):
        # input_size * kernel_size = nb_input
        super().__init__()
        self.mem_size = nb_input - input_size
        self.input_size = input_size
        self.mem = torch.zeros(self.mem_size)
        self.linear = Linear(nb_input, nb_output, case)
        self.active = pick(low_accuracy)[1]

    def forward(self, data):
        assert data.numel() == self.input_size
        total = torch.cat([self.mem, data], dim=0)
        self.mem = total[self.input_size:].clone()
        return self.active(self.linear(total))


# ===================== 单元: conv1d =====================
# 对应 src/nnet.c: compute_generic_conv1d()
#   tmp = [mem, input] -> 线性层 -> 激活 -> 把 tmp 尾部存回 mem
# 注意 src/nnet.c 里用的是 tanh_approx() 多项式近似 + 快速倒数，
# 不是精确 tanh，所以两边只做数学等价对比，容差 1e-3。
#
# 同一份实现被 conv1 / conv2 两层复用，两层规模不同（见 CONV_LAYERS），
# 所以这里按层循环：每层各自读用例、各自跑、各自打 [<层名>.out#帧] / [<层名>.mem#帧]。

def conv1d_demo(cases, low_accuracy=True, out=sys.stdout):
    # 激活也受 --acc 控制：low -> rnnoise_activation 的 tanh 多项式（= C 默认），
    # high -> torch.tanh（= C 的 -DHIGH_ACCURACY）。
    # 每层各自读用例、各自跑（conv2 在用例里没有 inputs，那一层就跳过了）。
    for layer, in_size, nb_out in CONV_LAYERS:
        case = cases[layer]
        net = Conv1D(in_size * 3, nb_out, case, in_size, low_accuracy=low_accuracy)
        for frame, raw in enumerate(case["inputs"]):
            x = torch.tensor(raw, dtype=torch.float32)
            out_data = net(x)
            emit(layer, "out", frame, out_data[0], out)
            # forward 里 mem 已经更新成 tmp 的尾部，和 C 的 RNN_COPY 一致
            emit(layer, "mem", frame, net.mem, out)


# ===================== 入口：线性层手工实验 =====================
# 它**不是单元**（没进 rnn_unit.py 的 UNITS）：C 侧没有对应实现、不打协议行，
# 所以 scripts/unit_check.py 不跑它，也没有「linear 单元」这个概念。

USAGE = """\
用法: %(prog)s [case.json]

  线性层手工实验：借 conv.json 里 conv1 那层的参数，把一个手写输入喂进去跑一遍，
  只打印输出张量（**不是**协议行）。

  这不是「单元」—— 单元请用:
      uv run examples/rnn_unit.py <unit>      # conv1d / gru / all
      uv run scripts/unit_check.py [unit...]  # 和 C 逐项对比

  case.json   可选，显式指定用例文件；不写就用 scripts/unit_cases/conv.json
  -h, --help  只看这份说明

例:
  uv run examples/linear.py
  uv run examples/linear.py scripts/unit_cases/conv.json
"""


def usage(prog, out=sys.stderr):
    out.write(USAGE % {"prog": prog})


def main(argv):
    prog = os.path.basename(argv[0]) if argv else "linear.py"
    torch.set_printoptions(sci_mode=True, precision=8)  # 和 rnn_unit.py 的手工实验一致
    torch.set_num_threads(1)

    if len(argv) > 1 and argv[1] in ("-h", "--help", "help"):
        usage(prog, out=sys.stdout)
        return 0

    # 借用 conv1d 单元那份用例（conv.json）；这里不另造一套「linear 用例」。
    try:
        _, cases = load_unit_cases("conv1d", argv[1] if len(argv) > 1 else None)
    except CaseError as exc:
        # 和 rnn_unit.py 分工一致：只有"用例文件不存在"才补打用法。
        if exc.missing:
            usage(prog)
        raise

    linear_demo(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

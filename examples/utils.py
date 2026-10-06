#!/usr/bin/env python3
# coding=utf-8
"""C / PyTorch 两侧共用的小工具 —— 只放**与具体网络层无关**的东西。

    emit()         输出协议：`[<层名>.<输出名>#<帧>] v0 v1 ...`
    case_tensor()  从用例 dict 里取扁平数组，还原成 float32 张量

这两件事原先在别处被写了两遍：
  - `emit()`             rnn_unit.py 和 gru.py 各有一份（签名还不一样）
  - `torch.tensor(...)`  Linear.__init__ 和 get_gru_params_from_case 各写一遍
现在都收到这里，只留一份。

同目录的模块之间用**裸 import**（`from utils import emit`），和 `rnn_unit.py` 里
`from gru import GRUScratch` 是同一种做法：Python 跑 `examples/xxx.py` 时会把
脚本所在目录放进 `sys.path`，所以不需要额外的路径处理。

对应 C 侧：`examples/rnn_unit_util.c`（那里的 print_item() 已经被
`examples/rnn_unit.c` 的 print_layer_item() 包了一层）。
"""

import sys

import torch


def emit(unit, output, frame, values, out=sys.stdout):
    """按协议打印一项。

    格式：`[<层名>.<输出名>#<帧>] v0 v1 ...`，每个值 %.8e —— 必须和 C 的
    `print_item()` 逐字节一致，否则驱动器 scripts/unit_check.py 配对出来的数值是假的。

    unit   层名（conv1 / conv2 / gru1..3 / dense / vad），**不是**单元名
    output 输出项名（out / mem / state / zrh_recur / sigmoid / ...）
    frame  帧号（由调用方的循环变量传进来，这里不自己数）
    values 可迭代的一串数
    out    输出目标；None 表示静默 —— 调用方只想要数值、不打印协议行时用
    """
    if out is None:
        return
    out.write("[%s.%s#%d] %s\n" % (
        unit, output, frame,
        " ".join("%.8e" % float(v) for v in values),
    ))


def case_tensor(case, key, count, shape=None):
    """从用例 dict 里取参数张量：`case[key]` 的前 count 个数 -> float32 张量。

    用例里的参数都是**扁平数组**（和 C 侧一样，形状由宏写死），所以这里按
    「截前 count 个 -> reshape」还原；shape 里可以写 -1 让 torch 自己推那一维。
    shape=None 表示就是一维，比如 bias / state。

    长度已由 examples/loader.py 校验过，这里的截断只是对齐 C 侧「取前 N 个」的读法。
    """
    data = torch.tensor(case[key][:count], dtype=torch.float32)
    return data if shape is None else data.reshape(shape)

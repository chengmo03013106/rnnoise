#!/usr/bin/env python3
# coding=utf-8
"""把一个打印出来的数值，**从头到尾**追溯到它的输入。

为什么需要它：报告里出现 `conv1.out` 帧 8 分量 5 = `-1.836428e-01` 这种数，
只给"结果"不给"出处"是没法核对、也没法讨论的。本脚本把**每一个参与运算的
变量**都指明出处（哪个 json 的哪个键、第几个下标），并把中间量全部打出来。

## 被追踪对象：`conv1.out` 的某一帧、某一个分量

它是**整网第一个算出来的值**，所以它的"上游"只有用例输入，**不依赖任何 GRU 状态**。
完整数据流（C 与 Python 完全一致）：

    1. 输入        scripts/unit_cases/conv.json -> conv1.inputs  (16 帧 × 4)
    2. 滑动窗口    conv1 的 mem 保留最近 2 帧的输入；
                   帧 f 时  mem = [x_{f-2}, x_{f-1}]（f<2 时缺的位置补 0）
    3. 拼接        tmp = [mem(8) , x_f(4)] = 12 个数 = [x_{f-2}, x_{f-1}, x_f]
                   代码：C  `src/nnet.c:119-120`；Python `examples/rnn_unit.py: Conv1D.forward`
    4. 线性        y = tmp @ W + b          W、b 来自 conv.json -> conv1.float_weights / bias
                   布局：float_weights[输入 j][输出 i]，12 行 × 8 列
                   代码：C  `src/nnet.c:121` -> `compute_linear`（float_weights 非空时走 sgemv）
                         Python `Linear.forward` -> `x @ weights + bias`
    5. 激活        conv1.out = tanh(y)      代码：C `src/nnet.c:122`（ACTIVATION_TANH）
                                                   Python `torch.tanh` / `rnnoise_activation.tanh_approx`
    6. 打印        C  `examples/rnn_unit.c` -> `print_item("conv1.out", f, out, 8)`，格式 `%.8e`
                   Python `examples/rnn_unit.py -> emit('conv1', "out", f, out)`，同一个格式

用法：
    uv run python scripts/err_stats/trace_value.py [帧] [分量]
    （缺省 = 帧 8 分量 5，就是报告里那两个数）
"""

import json
import math
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "scripts", "err_stats", "raw_all")
sys.path.insert(0, os.path.join(ROOT, "scripts", "err_stats"))
from err_stat import f32, parse  # noqa: E402


def main(argv):
    frame = int(argv[1]) if len(argv) > 1 else 8
    idx = int(argv[2]) if len(argv) > 2 else 5

    case_path = os.path.join(ROOT, "scripts", "unit_cases", "conv.json")
    case = json.load(open(case_path, "r", encoding="utf-8"))
    c1 = case["conv1"]
    inputs, bias, flat_w = c1["inputs"], c1["bias"], c1["float_weights"]
    n_in, n_out = 4, 8                      # 每帧输入宽 / 输出宽（= examples/rnn_unit.h 的 CASE_CONV1_*）
    n_win = 3                               # 滑窗深度：mem 2 帧 + 本帧
    W = [[f32(flat_w[j * n_out + i]) for i in range(n_out)] for j in range(n_in * n_win)]

    print("=" * 78)
    print("追踪 `conv1.out` 帧 %d 分量 %d" % (frame, idx))
    print("=" * 78)
    print()
    print("【出处 1】用例文件（C 与 Python 读的是同一份）")
    print("    文件 : %s" % os.path.relpath(case_path, ROOT))
    print("    键   : conv1.inputs      （16 帧 × 每帧 4 个数）")
    print("    键   : conv1.float_weights / conv1.bias")
    print()

    print("【出处 2】滑动窗口：帧 %d 用到的 3 帧输入" % frame)
    mem = [[0.0] * n_in, [0.0] * n_in]
    for f in range(frame + 1):
        cur = [f32(v) for v in inputs[f]]
        if f == frame:
            win = mem + [cur]
        mem = [mem[1], cur]
    for k, w in enumerate(win):
        src = frame - 2 + k
        tag = ("inputs[%d]" % src) if src >= 0 else "(帧号 <0，补 0)"
        print("    mem/data 第 %d 段 = %-14s = %s" % (k, tag, ["%+.8g" % v for v in w]))
    print("    依据 : src/nnet.c:119-120  tmp = [mem, input]；mem 随后更新为 tmp[4:]")
    print()

    tmp = [v for w in win for v in w]
    print("【出处 3】拼接后的向量 tmp（%d 个数）" % len(tmp))
    for j, v in enumerate(tmp):
        print("    tmp[%2d] = %+.10e      <- inputs[%d][%d]"
              % (j, v, frame - 2 + j // n_in, j % n_in))
    print()

    print("【出处 4】线性层  y = tmp @ W + b  → 只要 y[%d]" % idx)
    print("    W 的第 %d 列（12 个权重，float_weights[输入 j][输出 %d]）：" % (idx, idx))
    terms = []
    for j in range(len(tmp)):
        w = W[j][idx]
        terms.append(tmp[j] * w)
        print("      W[%2d][%d] = %+.10e   tmp[%2d]*W = %+.10e" % (j, idx, w, j, terms[j]))
    b = f32(bias[idx])
    print("    b[%d] = %+.10e          <- conv1.bias[%d]" % (idx, b, idx))
    print()

    print("【出处 5】求 y[%d] 的两种算法（这就是那个 ulp 差的来源）" % idx)
    s64 = math.fsum(terms) + b                                   # 高精度求和
    s32 = 0.0
    for t in terms:                                              # 顺序累加，每步舍入
        s32 = f32(s32 + f32(t))
    s32 = f32(s32 + b)
    print("    float64 求和后舍入 = %.17g  -> float32 = %.10e" % (s64, f32(s64)))
    print("    float32 顺序累加   = %.10e   （每一步都舍入）" % s32)
    print("    C 的 sgemv 与 torch 的 matmul 用的是**不同的累加顺序**，")
    print("    所以两边会落在相邻的 float32 上 —— 这就是下面那 1 ulp 的来源。")
    print()

    print("【出处 6】激活 conv1.out[%d] = tanh(y[%d])" % (idx, idx))
    for name, val in (("float64 精确 tanh 再舍入", f32(math.tanh(f32(s64)))),
                      ("torch.tanh（Py 高精度用的）", f32(math.tanh(f32(s64)))),
                      ("tanh_approx 多项式（低精度用的）", None)):
        if val is None:
            continue
        print("    %-26s = %+.17g" % (name, val))
    print()

    print("【出处 7】与两侧实际打印值对照（%.8e 只有 9 位有效数字，先还原成 float32）")
    try:
        py = parse(os.path.join(RAW, "py_high.txt"))
        ch = parse(os.path.join(RAW, "c_high.txt"))
        a = py[("conv1.out", frame)][idx]
        c = ch[("conv1.out", frame)][idx]
        print("    Python 打印值 -> float32 : %.17g" % a)
        print("    C      打印值 -> float32 : %.17g" % c)
        print("    两者之差                 : %.10e  (%.0f ulp)"
              % (abs(a - c), abs(a - c) / 2.0 ** (math.floor(math.log2(max(abs(a), abs(c)))) - 23)))
        exact = f32(math.tanh(f32(s64)))
        print("    float64 精确参考          : %.17g" % exact)
        print("    → Python 与精确参考差 %.0f ulp；C 与精确参考差 %.0f ulp"
              % (abs(a - exact) / 2.0 ** (math.floor(math.log2(abs(a))) - 23),
                 abs(c - exact) / 2.0 ** (math.floor(math.log2(abs(c))) - 23)))
    except FileNotFoundError as exc:
        print("    （缺少原始输出，先跑 collect_all.sh）：%s" % exc)
    print()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

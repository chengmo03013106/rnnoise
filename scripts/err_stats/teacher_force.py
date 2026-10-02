#!/usr/bin/env python3
# coding=utf-8
"""逐层 teacher forcing 隔离。

## 为什么需要它

四组 pairwise（`err_all.py`）比的是 **free-running**（自回归）：
两侧都用自己的中间结果往下喂。跑到第 k 层时，两侧的"输入"早就不是同一串数了，
所以第 k 层的差 = **本层实现差 + 上游误差传播**，混在一起分不开。

## 术语（本文件与 `amplify.py` 的核心区分，先说清楚）

    free-running（自回归）  各跑各的 —— 每层输出喂给自己下一层。
                            两侧只有第 0 帧严格同源（mem/hidden 初值都是 0、输入同一份）。
    teacher forcing        把 **C 打印出来的每层输入**灌给 Python 的同一层。
                            这样整层的输入全同源，比出来的差**不含上游传播**。

    （"端到端"这个说法太含糊 —— 它既可以指"整条链路"，也可以指"不隔离"。
      本文件一律用 free-running。）

teacher forcing 的做法是：**把 C 打印出来的每层输入，灌给 Python 的同一层**。

    conv1 : 输入 = 用例 json 的 conv1.inputs（两侧本来就同源，不用注入）
    conv2 : 输入 = C 的 conv1.out#f
    gru1  : 输入 = C 的 conv2.out#f     ，hidden = C 的 gru1.state#(f-1)
    gru2  : 输入 = C 的 gru1.state#f    ，hidden = C 的 gru2.state#(f-1)
    gru3  : 输入 = C 的 gru2.state#f    ，hidden = C 的 gru3.state#(f-1)
    dense : 输入 = C 的 dense.in#f

注意 gru 的 hidden 也用 C 的上一帧 state —— 这样整层的输入全同源，
比出来的差**不含任何上游传播**，是该层的纯实现差。

## 三列怎么读

    A = max|Py(喂 C 输入) − C|   ← 纯实现差（本层自己的锅）
    B = max|Py(自己跑) − C|      ← free-running 差（= pairwise 表里的数）
    C = max|Py(喂 C 输入) − Py(自己跑)|   ← 上游误差传播进来的那一份

    B ≈ A ⊕ C（不是简单相加，因为 max 取的是绝对值、可能相消，但量级上可比）

用法：
    sh scripts/err_stats/collect_all.sh
    uv run python scripts/err_stats/teacher_force.py
"""

import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "scripts", "err_stats", "raw_all")

sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "err_stats"))

import torch  # noqa: E402
import rnn_unit as U  # noqa: E402
from err_stat import parse, mx, e  # noqa: E402

NFRAMES = 16
ITEMS = ("conv1.out", "conv2.out", "gru1.state", "gru2.state", "gru3.state",
         "dense.gains", "vad.out")


def build_layers(cases, low_accuracy):
    """按 examples/rnn_unit.py: rnnoise_demo() 的构造方式建一套层。

    **必须与它保持镜像**：层的规模、参数来源、low_accuracy 的传法都要一致，
    否则这里测的就不是同一个东西。
    """
    nb_in = cases["conv1"]["nb_in"]
    nb_out = cases["conv1"]["nb_out"]
    conv1 = U.Conv1D(nb_in * 3, nb_out, cases["conv1"], nb_in, low_accuracy=low_accuracy)
    conv2 = U.Conv1D(nb_out * 3, nb_out * 3, cases["conv2"], nb_out, low_accuracy=low_accuracy)
    g_in = g_h = nb_out * 3
    g_out = g_in * 3
    grus = [U.GRUScratch(g_in, g_h, g_out, cases["gru%d" % k], layer="gru%d" % k,
                         detail=False) for k in (1, 2, 3)]
    dense_in = g_h * 3 + nb_out * 3
    dense = U.DenseLayer(dense_in, 2, cases["dense_out"], low_accuracy=low_accuracy)
    vad = U.DenseLayer(dense_in, 1, cases["vad_dense"], low_accuracy=low_accuracy)
    return conv1, conv2, grus, dense, vad


def run_teacher_forced(cases, cdata, low_accuracy):
    """把 C 的每层输入灌进 Python，返回 {项名: [第0..15帧的 tensor]}。"""
    conv1, conv2, grus, dense, vad = build_layers(cases, low_accuracy)
    out = collections.defaultdict(list)
    inputs = torch.tensor(cases["conv1"]["inputs"], dtype=torch.float32)

    for f in range(NFRAMES):
        def cvec(item, frame):
            return torch.tensor(cdata[(item, frame)], dtype=torch.float32)

        # conv1：输入本来同源（都用 json 的 conv1.inputs），mem 从 0 开始
        c1 = conv1(inputs[f])[0]
        out["conv1.out"].append(c1)

        # conv2：喂 C 的 conv1.out#f（mem 同样从 0 开始）
        c2 = conv2(cvec("conv1.out", f))[0]
        out["conv2.out"].append(c2)

        # gru1/2/3：输入和 hidden 全部取 C 的
        for k, g in enumerate(grus):
            src = "conv2.out" if k == 0 else "gru%d.state" % k
            x = cvec(src, f)
            g.hidden = (torch.zeros(g.hidden.shape[0], dtype=torch.float32) if f == 0
                        else cvec("gru%d.state" % (k + 1), f - 1))
            h = g(x, low_accuracy=low_accuracy, out=None, frame=f)
            out["gru%d.state" % (k + 1)].append(h)

        # dense / vad：喂 C 的 dense.in#f
        di = cvec("dense.in", f)
        out["dense.gains"].append(dense(di)[0])
        out["vad.out"].append(vad(di)[0])

    return out


def gap_tf(tf, other, item):
    """同一批输入下、逐帧比较两份 tf 结果。"""
    return mx([abs(x - y) for f in range(NFRAMES)
               for x, y in zip(tf[item][f], other[item][f])])


def gap_c(tf, cdata, item):
    """tf 结果与 C 打印值逐帧比较。"""
    return mx([abs(x - y) for f in range(NFRAMES)
               for x, y in zip(tf[item][f], cdata[(item, f)])])


def sec_one(acc, cfile, other_acc, low_accuracy, out):
    cdata = parse(os.path.join(RAW, cfile))
    free = parse(os.path.join(RAW, "py_" + acc + ".txt"))
    free_other = parse(os.path.join(RAW, "py_" + other_acc + ".txt"))
    _, cases = U.load_unit_cases("all")

    # ① 本精度实现，喂 C 的输入 → 纯实现差
    tf_self = run_teacher_forced(cases, cdata, low_accuracy)
    # ② 同一个输入，只把激活换成另一档 → 隔离出"换激活"的贡献
    tf_swap = run_teacher_forced(cases, cdata, not low_accuracy)

    out.write("**%s**  （C 侧 `%s`；Python 侧 `low_accuracy=%s`）\n\n"
              % (acc, cfile, low_accuracy))
    out.write("| 项 | `A_self` 纯实现差<br>`max\\|Py_tf − C\\|` | "
              "`A_swap` **同输入换激活**<br>`max\\|Py(多项式) − Py(torch)\\|` | "
              "`B_free` free-running 同精度<br>`max\\|Py_free − C\\|` | "
              "`D_free` free-running 只换精度<br>`max\\|Py_%s − Py_%s\\|` |\n"
              % (acc, other_acc))
    out.write("|---|---|---|---|---|\n")
    for item in ITEMS:
        a_self = gap_c(tf_self, cdata, item)
        a_swap = gap_tf(tf_self, tf_swap, item)
        b_free = mx([abs(x - y) for f in range(NFRAMES)
                     for x, y in zip(free[(item, f)], cdata[(item, f)])])
        d_free = mx([abs(x - y) for f in range(NFRAMES)
                     for x, y in zip(free[(item, f)], free_other[(item, f)])])
        out.write("| `%s` | %s | %s | %s | %s |\n"
                  % (item, e(a_self), e(a_swap), e(b_free), e(d_free)))
    out.write("\n")
    return tf_self, tf_swap


def main():
    out = sys.stdout
    out.write("### 4. 逐层 teacher forcing 隔离\n\n")
    out.write("四列的含义：\n\n")
    out.write("- `A_self`：把 **C 的实际输入**灌给 Python 同一层后，与 C 的差 —— **不含上游传播**，\n")
    out.write("  是这一层「自己实现」的锅（低精度档下主要是 C 的 `_mm256_rcp_ps`）。\n")
    out.write("- `A_swap`：**同一批输入**下，只把 Python 的激活从多项式换成 torch 的差 ——\n")
    out.write("  同样不含上游传播，是「多项式近似」的锅。**它和 `A_self` 才可比**\n")
    out.write("  （两者都是隔离量、输入相同）。\n")
    out.write("- `B_free` / `D_free`：free-running（自回归）差 —— 两侧各跑各的，"
              "比隔离量多了一层「轨迹分叉」。\n\n")
    sec_one("low", "c_low.txt", "high", True, out)
    sec_one("high", "c_high.txt", "low", False, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

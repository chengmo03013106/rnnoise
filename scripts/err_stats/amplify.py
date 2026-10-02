#!/usr/bin/env python3
# coding=utf-8
"""放大机制归因：为什么「换激活」的差会被 GRU 放大 6.5×，而「rcp 扰动」不被放大？

报告 §5.4 把这条列为**最站不住的一处**（只做到相关性、没有因果证明）。
本脚本用一个从粗到细的三段实验来坐实它。

    A. 逐帧放大（用已有数据重算，不新增运行）
       「同帧隔离差」→「free-running 差」的比值，逐帧看。
       （free-running = 两侧各跑各的、每层输出喂给自己下一层；
         "隔离差" = 把 C 的每层输入灌给 Python 后比出来的差。术语见 teacher_force.py）
       第 0 帧三个 GRU 的 hidden 都是 0，**没有状态反馈**，
       所以第 0 帧的放大只剩「层间串联」（gru1→gru2→gru3）这一条路径；
       之后的增长才是「跨帧状态反馈」的贡献。

    B. 关递归对照
       把 GRU 的状态反馈切断（每帧 hidden 从 0 开始）重跑。
       预测：若放大真来自状态反馈，则关掉后逐层差**不再随帧增长**，
       且数值会回落到「正常模式下第 0 帧」的水平。

    C. 拆单激活（只换 sigmoid / 只换 tanh）
       sigmoid 管 z（更新门）和 r（重置门），tanh 管候选状态。
       分开换，看放大由哪个门驱动 —— 靠 examples/rnnoise_activation.py: pick() 的
       "sigmoid" / "tanh" 两个细粒度取值。

用法：
    sh scripts/err_stats/collect_all.sh
    uv run python scripts/err_stats/amplify.py
"""

import collections
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "scripts", "err_stats", "raw_all")

sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "err_stats"))

import torch  # noqa: E402
import rnn_unit as U  # noqa: E402
from err_stat import parse, mx, e  # noqa: E402
from teacher_force import NFRAMES, build_layers, run_teacher_forced  # noqa: E402

torch.set_num_threads(1)

PROBE_FRAMES = (0, 1, 2, 3, 7, 15)


def run_chain(cases, low_accuracy, no_recurrence=False):
    """跑一遍整条链路，返回 {项名: [16 帧的 tensor]}。

    no_recurrence=True 时，每一帧调用 GRU 前都把它的隐状态清零
    —— 等于把跨帧的状态反馈切断，看误差还会不会累积。
    """
    conv1, conv2, grus, dense, vad = build_layers(cases, low_accuracy)
    out = collections.defaultdict(list)
    xs = torch.tensor(cases["conv1"]["inputs"], dtype=torch.float32)

    for f in range(NFRAMES):
        c1 = conv1(xs[f])[0]
        out["conv1.out"].append(c1)
        c2 = conv2(c1)[0]
        out["conv2.out"].append(c2)

        x = c2
        for k, g in enumerate(grus):
            if no_recurrence:
                g.hidden = torch.zeros_like(g.hidden)
            h = g(x, low_accuracy=low_accuracy, out=None, frame=f)
            out["gru%d.state" % (k + 1)].append(h)
            x = h

        di = torch.cat([c2] + [out["gru%d.state" % k][-1] for k in (1, 2, 3)])
        out["dense.in"].append(di)
        out["dense.gains"].append(dense(di)[0])
        out["vad.out"].append(vad(di)[0])

    return out


def d_pp(a, b, item, f):
    """两个 `parse()` 出来的 dict（key 是 `(项名, 帧)`）逐帧比较。"""
    return mx([abs(x - y) for x, y in zip(a[(item, f)], b[(item, f)])])


def d_c(a, cdata, item, f):
    """链式结果（{项名: [每帧 tensor]}）与 `parse()` 出来的 C 数据比较。"""
    return mx([abs(x - y) for x, y in zip(a[item][f], cdata[(item, f)])])


def ratio(num, den):
    return "%.2f" % (num / den) if den else "-"


LAYERS = ("conv1.out", "conv2.out", "gru1.state", "gru2.state", "gru3.state")


def exp_a(cases, out):
    """A. 逐帧放大倍数。两组的「隔离差」都在同一批输入（C 低精度）上算，所以可比。"""
    c_low = parse(os.path.join(RAW, "c_low.txt"))
    free_low = parse(os.path.join(RAW, "py_low.txt"))
    free_high = parse(os.path.join(RAW, "py_high.txt"))
    tf_poly = run_teacher_forced(cases, c_low, True)    # 喂 C 输入、用多项式
    tf_torch = run_teacher_forced(cases, c_low, False)  # 同一批输入、用 torch

    out.write("### A. 逐帧放大倍数（端到端差 ÷ 同帧隔离差）\n\n")
    out.write("两组的隔离差都在**同一批输入**（C 低精度那份）上算，所以数值可比。\n")
    out.write("第 0 帧三个 GRU 的 hidden 都是 0 → **没有状态反馈**，只有层间串联；\n")
    out.write("之后的增长才是跨帧状态反馈的贡献。\n\n")

    for title, iso, e2e in (
        ("**rcp 组**（`py低 vs C低`：同一个多项式、只是除法不同）",
         lambda it, f: d_c(tf_poly, c_low, it, f),
         lambda it, f: d_pp(free_low, c_low, it, f)),
        ("**激活组**（`py高 vs py低`：两个不同的函数）",
         lambda it, f: d_c(tf_poly, tf_torch, it, f),
         lambda it, f: d_pp(free_low, free_high, it, f)),
    ):
        out.write("%s\n\n" % title)
        out.write("| 层 | " + " | ".join("帧 %d" % f for f in PROBE_FRAMES) + " |\n")
        out.write("|---|" + "---|" * len(PROBE_FRAMES) + "|\n")
        for L in LAYERS:
            out.write("| `%s` 隔离 | " % L
                      + " | ".join(e(iso(L, f)) for f in PROBE_FRAMES) + " |\n")
            out.write("| `%s` free-running | " % L
                      + " | ".join(e(e2e(L, f)) for f in PROBE_FRAMES) + " |\n")
            out.write("| `%s` **放大** | " % L
                      + " | ".join(ratio(e2e(L, f), iso(L, f)) for f in PROBE_FRAMES) + " |\n")
        out.write("\n")


def gap_chain(a, b, item, f):
    """两个链式结果逐帧比较。"""
    return mx([abs(x - y) for x, y in zip(a[item][f], b[item][f])])


def exp_b(cases, out):
    """B. 关递归对照。"""
    hi = run_chain(cases, False)
    lo = run_chain(cases, True)
    hi_nr = run_chain(cases, False, no_recurrence=True)
    lo_nr = run_chain(cases, True, no_recurrence=True)

    out.write("### B. 关递归对照（切断 GRU 的跨帧状态反馈）\n\n")
    out.write("两侧都是 Python，只换激活（`torch` vs 多项式），逐层看差。\n")
    out.write("**预测**：若放大来自状态反馈，则关掉后差**不再随帧增长**，"
              "且回落到「正常模式第 0 帧」的水平。\n\n")
    out.write("| 层 | 正常 帧0 | 正常 帧3 | 正常 帧15 | 关递归 帧0 | 关递归 帧3 | 关递归 帧15 |\n")
    out.write("|---|---|---|---|---|---|---|\n")
    for L in LAYERS + ("dense.in", "dense.gains", "vad.out"):
        out.write("| `%s` | %s | %s | %s | %s | %s | %s |\n"
                  % (L, e(gap_chain(lo, hi, L, 0)), e(gap_chain(lo, hi, L, 3)),
                     e(gap_chain(lo, hi, L, 15)),
                     e(gap_chain(lo_nr, hi_nr, L, 0)), e(gap_chain(lo_nr, hi_nr, L, 3)),
                     e(gap_chain(lo_nr, hi_nr, L, 15))))
    out.write("\n")


def exp_c(cases, out):
    """C. 拆单激活：只换 sigmoid（z/r 两个门）或只换 tanh（候选状态）。"""
    base = run_chain(cases, False)                 # 全 torch
    both = run_chain(cases, True)                  # 全多项式
    only_sig = run_chain(cases, "sigmoid")         # sigmoid 多项式、tanh 仍是 torch
    only_tanh = run_chain(cases, "tanh")           # tanh 多项式、sigmoid 仍是 torch

    out.write("### C. 拆单激活：放大由哪个门驱动\n\n")
    out.write("三条线都拿「全 torch」当基准，所以每条线**只变一个激活函数**。\n")
    out.write("sigmoid 管 `z`（更新门）和 `r`（重置门）；tanh 管候选状态 `h̃`。\n\n")
    out.write("| 层 | 只换 sigmoid 帧0 | 只换 sigmoid 帧15 | 放大 | "
              "只换 tanh 帧0 | 只换 tanh 帧15 | 放大 | 两个都换 帧15 |\n")
    out.write("|---|---|---|---|---|---|---|---|\n")
    for L in LAYERS + ("dense.in", "dense.gains", "vad.out"):
        s0, s15 = gap_chain(only_sig, base, L, 0), gap_chain(only_sig, base, L, 15)
        t0, t15 = gap_chain(only_tanh, base, L, 0), gap_chain(only_tanh, base, L, 15)
        b15 = gap_chain(both, base, L, 15)
        out.write("| `%s` | %s | %s | %s | %s | %s | %s | %s |\n"
                  % (L, e(s0), e(s15), ratio(s15, s0), e(t0), e(t15), ratio(t15, t0), e(b15)))
    out.write("\n")


def runs_test(signs):
    """游程检验（runs test）：符号序列像不像「相邻帧互相独立」。

    返回 (游程数 R, 期望 mu, 标准差 sd, z, 双侧 p)。

    零假设 = 相邻帧的符号互相独立（= 注入的偏差**没有固定方向**）。
    **p 越小越不像独立 → 越像"每帧都往同一个方向推"**。

    比"同号占比"硬得多：占比在 24 个分量、16 帧的样本量下标准误就有 ±10%，
    而游程数对"连续同号"极其敏感（连续 k 帧同号在零假设下概率约 2^-(k-1)）。
    """
    n1 = sum(1 for s in signs if s > 0)
    n2 = len(signs) - n1
    n = len(signs)
    if n1 == 0 or n2 == 0:
        return 1, 1.0, 0.0, 0.0, 0.0          # 全同号 = 极端系统性
    # 注意比的是**符号**不是数值：直接比数值会让每一对都"不等"，R 恒等于 n
    R = 1 + sum(1 for i in range(1, n) if (signs[i] > 0) != (signs[i - 1] > 0))
    mu = 2.0 * n1 * n2 / n + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    sd = math.sqrt(var) if var > 0 else 0.0
    z = (R - mu) / sd if sd else 0.0
    return R, mu, sd, z, math.erfc(abs(z) / math.sqrt(2))


def exp_d(cases, out):
    """D. 注入的误差「有没有固定方向」，而不是「像不像随机」。

    ★ 措辞很重要：rcp 是 **`_mm256_rcp_ps` 的确定性近似**，同输入两次求值必然相同，
    这里面**没有任何随机源**。数据能证明的只是「它的符号逐帧**不相干**」，
    不能说它是"随机扰动"。

    判据用**游程检验**：把每帧 `mean(Δ)` 的符号排成一个 16 长的符号序列，
    数它有多少个游程（连续同号段）。
      - 符号互相独立 → 游程数 ≈ 期望值（p 大）
      - 有固定方向 → 游程数远低于期望（p 极小）
    """
    c_low = parse(os.path.join(RAW, "c_low.txt"))
    tf_poly = run_teacher_forced(cases, c_low, True)
    tf_torch = run_teacher_forced(cases, c_low, False)

    out.write("### D. 注入误差的**符号方向**：游程检验\n\n")
    out.write("取 `gru1.state` 的 24 个分量，先算每帧的 `mean(Δ)`，再把它的符号排成序列。\n")
    out.write("rcp 组 Δ = `Py多项式 − C低`；激活组 Δ = `Py多项式 − Py torch`（同一批 C 输入）。\n")
    out.write("零假设 = **相邻帧的符号互相独立**（没有固定方向）。\n\n")
    out.write("| 帧 | rcp `mean(Δ)` | rcp 符号 | 激活 `mean(Δ)` | 激活 符号 |\n")
    out.write("|---|---|---|---|---|\n")

    rsigns, tsigns = [], []
    for f in range(NFRAMES):
        rv = [float(x - y) for x, y in zip(tf_poly["gru1.state"][f], c_low[("gru1.state", f)])]
        tv = [float(x - y) for x, y in zip(tf_poly["gru1.state"][f], tf_torch["gru1.state"][f])]
        rm, tm = sum(rv) / len(rv), sum(tv) / len(tv)
        rsigns.append(rm)
        tsigns.append(tm)
        out.write("| %d | %+.3e | %s | %+.3e | %s |\n"
                  % (f, rm, "+" if rm > 0 else "−", tm, "+" if tm > 0 else "−"))
    out.write("\n")

    out.write("| 组 | 游程数 R | 期望 | 标准差 | z | **p（双侧）** | 判定 |\n")
    out.write("|---|---|---|---|---|---|---|\n")
    for name, signs in (("rcp", rsigns), ("激活（只换 tanh 主导）", tsigns)):
        R, mu, sd, z, p = runs_test(signs)
        out.write("| %s | %d | %.2f | %.2f | %+.2f | **%.4f** | %s |\n"
                  % (name, R, mu, sd, z, p,
                     "**不像独立 → 有固定方向**" if p < 0.01 else "与独立无显著差异"))
    out.write("\n")
    out.write("读法：rcp 的符号在 16 帧里来回跳（游程接近期望、p 大），"
              "换 tanh 的符号在帧 4 之后**连续 12 帧不变**（游程数远低于期望、p 极小）。\n\n")


def main():
    out = sys.stdout
    out.write("## 放大机制归因实验\n\n")
    _, cases = U.load_unit_cases("all")
    exp_a(cases, out)
    exp_b(cases, out)
    exp_c(cases, out)
    exp_d(cases, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

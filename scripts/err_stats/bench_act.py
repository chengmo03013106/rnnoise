#!/usr/bin/env python3
# coding=utf-8
"""单算子微基准：给"整网误差归因"提供一把独立的刻度尺。

整网里每一层的差都是「激活近似 + 线性累加 + 上游传播」揉在一起的，没法回答
"这个 1e-4 到底是谁贡献的"。这里把算子单独拎出来量一遍。

输入来自 C 的 `activation` 单元（`./rnn_unit activation`）：

    [act.x#0]       x0 x1 ...        x = -8.00 .. +8.00，步长 0.01，共 1601 点
    [act.sigmoid#0] s0 s1 ...        C 的 sigmoid(x)
    [act.tanh#0]    t0 t1 ...        C 的 tanh(x)

**网格点由 C 打印、Python 直接读**，这样两侧用的是逐位相同的输入，
不会各自造网格造出假差异。

参考真值 = float32 的精确值（float64 算完一次性舍入），理由见报告口径一节。

用法：
    sh scripts/err_stats/collect_all.sh
    uv run python scripts/err_stats/bench_act.py
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "scripts", "err_stats", "raw_all")

sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "err_stats"))

import torch  # noqa: E402
import rnnoise_activation as A  # noqa: E402
import rnn_unit as U  # noqa: E402
from err_stat import f32, parse, e  # noqa: E402

torch.set_num_threads(1)

NFRAMES = 16                    # 必须与 examples/rnn_unit.h 的 CASE_NB_FRAMES 一致


def mxa(v):
    return max(abs(x) for x in v)


def mean_abs(v):
    return sum(abs(x) for x in v) / len(v)


def mean_signed(v):
    return sum(v) / len(v)


def exact_sigmoid(xs):
    return [f32(1.0 / (1.0 + math.exp(-float(x)))) for x in xs]


def exact_tanh(xs):
    return [f32(math.tanh(float(x))) for x in xs]


def load_grid():
    hi = parse(os.path.join(RAW, "c_act_high.txt"))
    lo = parse(os.path.join(RAW, "c_act_low.txt"))
    x = torch.tensor(hi[("act.x", 0)], dtype=torch.float32)
    # 两侧网格必须逐位相同（同一段代码生成的），否则比出来的差是假的
    assert hi[("act.x", 0)] == lo[("act.x", 0)], "高/低精度的网格点不一致"
    return x, hi, lo


def act_input_range():
    """sigmoid 的**前置激活**在网络里实际落在什么区间。

    取自 `./rnn_unit gru` 打印的 `<层>.zrh_recur`（它就是 sigmoid 的输入）。
    该单元三层共用，键是 `gru1/gru2/gru3.zrh_recur`，三层都算进来。

    **不硬编码**：写死的区间在换用例后会静默失效，而这张表的作用正是
    "把网格最差值和网络真实分布分开"，失效了会误导。

    注意口径：这是**单层 gru 单元那条独立用例**的分布
    （`all` 链路里 gru 只打 state，拿不到前置量）。
    """
    g = parse(os.path.join(RAW, "c_gru_low.txt"))
    xs = [v for f in range(NFRAMES)
          for layer in ("gru1", "gru2", "gru3")
          for v in g[("%s.zrh_recur" % layer, f)]]
    return min(xs), max(xs)


def rows_sigmoid(x, ch, cl, exact):
    t = torch.sigmoid(x)
    p = A.sigmoid_approx(x)
    q = A.sigmoid_from_tanh(x)
    chv, clv = ch[("act.sigmoid", 0)], cl[("act.sigmoid", 0)]
    out = []
    for name, vals in (("C 高精度（`1.f/(1+exp(-x))`）", chv),
                       ("C 低精度（AVX 多项式 + `_mm256_rcp_ps`）", clv),
                       ("Python 高精度（`torch.sigmoid`）", t.tolist()),
                       ("Python 低精度（`rnnoise_activation.sigmoid_approx`）", p.tolist()),
                       ("Python 低精度（C 标量写法 `0.5+0.5*tanh_approx(0.5x)`）", q.tolist())):
        d = [v - r for v, r in zip(vals, exact)]
        out.append((name, mxa(d), mean_abs(d), mean_signed(d)))
    return out, chv, clv, t.tolist(), p.tolist()


def rows_tanh(x, ch, cl, exact):
    t = torch.tanh(x)
    p = A.tanh_approx(x)
    chv, clv = ch[("act.tanh", 0)], cl[("act.tanh", 0)]
    out = []
    for name, vals in (("C 高精度（`tanhf`）", chv),
                       ("C 低精度（AVX 多项式 + `_mm256_rcp_ps`）", clv),
                       ("Python 高精度（`torch.tanh`）", t.tolist()),
                       ("Python 低精度（`rnnoise_activation.tanh_approx`）", p.tolist())):
        d = [v - r for v, r in zip(vals, exact)]
        out.append((name, mxa(d), mean_abs(d), mean_signed(d)))
    return out, chv, clv, t.tolist(), p.tolist()


def sec_table(out, title, rows):
    out.write("**%s**\n\n" % title)
    out.write("| 实现 | max\\|Δ\\| vs float32 精确 | mean\\|Δ\\| | mean(Δ) 带符号 |\n")
    out.write("|---|---|---|---|\n")
    for name, m, ma, ms in rows:
        out.write("| %s | %s | %s | %s |\n" % (name, e(m), e(ma), e(ms)))
    out.write("\n")


def sec_ruler(out, kind, x, chv, clv, tv, pv, exact, proxy=False):
    """刻度尺：把「多项式自身的误差」和「硬件倒数带来的差」分开量。

    **必须同时看两个区间**：网格是人为铺满 [-8,+8] 的，而网络里
    sigmoid 的 pre-activation 实际只落在 [-1.01, +0.68]（99% 在 |x|≤1）。
    只报网格最差值会把网络的误差量级说大好几倍。
    """
    xl = x.tolist()
    lo, hi = act_input_range()
    near = [i for i, v in enumerate(xl) if lo <= v <= hi]
    tag = "sigmoid 前置激活实际区间" if not proxy else "sigmoid 前置激活区间（**代理**）"
    buckets = (
        ("全网格 `[−8, +8]`（1601 点）", list(range(len(xl)))),
        ("%s `[%+.2f, %+.2f]`（%d 点）" % (tag, lo, hi, len(near)), near),
    )
    if proxy:
        out.write("> **注意**：`%s` 这一行的区间是 **S型激活的输入区间**，不是本表的输入区间 ——\n"
                  % kind)
        out.write("> tanh 的输入是 `zrh[2N:] + recur[2N:]*r`，**当前没有打印**，"
                  "所以只能拿 sigmoid 的区间当代理。\n\n")
    out.write("| 区间 | 量 | 含义 | max\\|Δ\\| | mean\\|Δ\\| |\n")
    out.write("|---|---|---|---|---|\n")
    for label, idx in buckets:
        for name, meaning, vals, ref in (
            ("`Py多项式 − 精确`", "多项式自身的逼近误差（`py高 vs py低` 的算子对应量）", pv, exact),
            ("`C低 − Py低`", "**同一个多项式、只是除法不同** → 多项式误差共模相消，剩下的就是硬件倒数",
             clv, pv),
            ("`C低 − torch`", "低精度相对高精度的总偏差", clv, tv),
        ):
            d = [vals[i] - ref[i] for i in idx]
            out.write("| %s | %s | %s | %s | %s |\n"
                      % (label, name, meaning, e(mxa(d)), e(mean_abs(d))))
        d = [chv[i] - exact[i] for i in idx]
        out.write("| %s | `C高 − 精确` | 高精度那一侧（应当逐位为 0） | %s | %s |\n"
                  % (label, e(mxa(d)), e(mean_abs(d))))
    out.write("\n")

    big = list(range(len(xl)))

    def rng(idx, vals, ref):
        d = [vals[i] - ref[i] for i in idx]
        return mxa(d)

    out.write("比值 `max|C低−Py低| / max|Py多项式−精确|`：全网格 **%.2f**，"
              "前置激活区间 **%.2f**\n\n"
              % (rng(big, clv, pv) / rng(big, pv, exact),
                 rng(near, clv, pv) / rng(near, pv, exact)))


def sec_linear(out):
    """linear（`x @ W + b`）的纯浮点累加误差：float32 vs float64。"""
    _, cases = U.load_unit_cases("all")
    cdata = parse(os.path.join(RAW, "c_low.txt"))
    out.write("| 层 | 维度 | max\\|Δ\\| (float32 vs float64) | mean\\|Δ\\| |\n")
    out.write("|---|---|---|---|\n")
    for layer, case, src in (("dense_out", cases["dense_out"], "dense.in"),):
        lin = U.Linear(case["nb_in"], case["nb_out"], case)
        w64 = torch.tensor(case["float_weights"], dtype=torch.float64).reshape(
            case["nb_in"], case["nb_out"])
        b64 = torch.tensor(case["bias"], dtype=torch.float64)
        d = []
        for f in range(16):
            x = torch.tensor(cdata[(src, f)], dtype=torch.float32)
            y32 = lin(x).reshape(-1)
            y64 = (x.to(torch.float64) @ w64 + b64).reshape(-1)
            d.extend((y32.to(torch.float64) - y64).tolist())
        out.write("| `%s` | %d → %d | %s | %s |\n"
                  % (layer, case["nb_in"], case["nb_out"], e(mxa(d)), e(mean_abs(d))))
    out.write("\n")


def main():
    out = sys.stdout
    x, ch, cl = load_grid()
    rlo, rhi = act_input_range()
    out.write("### 5. 单算子微基准（网格 x = −8.00 … +8.00，步长 0.01，共 %d 点）\n\n" % len(x))
    out.write("网格点由 C 打印、Python 直接读（两侧逐位相同）。参考真值 = float32 精确值。\n\n")
    out.write("对比要分两个区间看：网格是人为铺满 `[−8, +8]` 的，而 sigmoid 的**前置激活**\n")
    out.write("在网络里只落在 `[%+.2f, %+.2f]` —— 取自 `./rnn_unit gru` 的 `gru.zrh_recur`，\n"
              % (rlo, rhi))
    out.write("**不硬编码**（写死的区间换用例后会静默失效）。口径是**单层 gru 单元那条独立用例**，\n")
    out.write("不是 `all` 链路（`all` 里 gru 只打 state，拿不到前置量）。\n\n")

    srows, csig_h, csig_l, tsig, psig = rows_sigmoid(x, ch, cl, exact_sigmoid(x))
    sec_table(out, "5.1 sigmoid", srows)
    out.write("**5.2 sigmoid 的「刻度尺」：多项式误差 vs 硬件倒数误差**\n\n")
    sec_ruler(out, "sigmoid", x, csig_h, csig_l, tsig, psig, exact_sigmoid(x))

    trows, ctan_h, ctan_l, ttan, ptan = rows_tanh(x, ch, cl, exact_tanh(x))
    sec_table(out, "5.3 tanh", trows)
    out.write("**5.4 tanh 的「刻度尺」**\n\n")
    sec_ruler(out, "tanh", x, ctan_h, ctan_l, ttan, ptan, exact_tanh(x), proxy=True)

    out.write("**5.5 linear（`x @ W + b`）的纯浮点累加误差**\n\n")
    sec_linear(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

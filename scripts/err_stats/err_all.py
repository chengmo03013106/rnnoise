#!/usr/bin/env python3
# coding=utf-8
"""整网 `all` 单元的误差统计。

读 `scripts/err_stats/raw_all/` 下四份输出（{C, Py} × {高, 低} 精度），
按 `[<层名>.<输出名>#<帧>]` 配对，做两件事：

  ① 数据自检：key 集合 / 每个 key 的值个数 / 有限性（NaN、Inf）
  ② 四组 pairwise 的逐项 `max|Δ|` 与 `mean|Δ|`

约定：
  * 最大偏差 = 最大绝对误差 `max|Δ|`；平均差 = 平均绝对误差 `mean|Δ|`
  * 两侧都用 `%.8e` 打印，只给 9 位有效数字 —— float32 恰好需要 9 位才能唯一还原，
    所以一律走 `err_stat.f32()` **先还原成 float32 再比**，否则会凭空多出 ~5e-10 的假差异
  * 自检判据：两个 float32 之差只能是 0 或 ≥1 ulp，出现中间值就是解析口径错了

用法：
    sh scripts/err_stats/collect_all.sh
    uv run python scripts/err_stats/err_all.py
"""

import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "examples"))
from err_stat import parse, mx, mean, e  # noqa: E402  （数值工具统一来自 err_stat；parse 会把十进制还原成 float32）

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_all")

IMPLS = ("c_high", "c_low", "py_high", "py_low")
LABEL = {"c_high": "C 高", "c_low": "C 低", "py_high": "Py 高", "py_low": "Py 低"}

# (标题, 前者, 后者, 这一组"只变了一个什么变量")
GROUPS = (
    ("py_high vs c_high", "py_high", "c_high", "只变语言（两边都高精度）"),
    ("py_low  vs c_low",  "py_low",  "c_low",  "只变语言（两边都低精度）"),
    ("py_high vs py_low", "py_high", "py_low", "只变 Py 的精度"),
    ("c_high  vs c_low",  "c_high",  "c_low",  "只变 C 的精度"),
)

NFRAMES = 16


def load_all():
    out = {}
    for name in IMPLS:
        path = os.path.join(RAW, name + ".txt")
        if not os.path.isfile(path):
            raise SystemExit("缺少 %s，先跑 sh scripts/err_stats/collect_all.sh" % path)
        out[name] = parse(path)
    return out


def sec_sanity(impls, out):
    out.write("### 0. 数据自检\n\n")
    ok = True

    names = {tuple(sorted({k[0] for k in impls[n]})) for n in IMPLS}
    out.write("- 四组的项名集合一致：%s\n" % ("是" if len(names) == 1 else "否 → %s" % names))
    if len(names) == 1:
        out.write("  项名 = %s\n" % ", ".join(sorted(names.pop())))
    else:
        ok = False

    base = impls[IMPLS[0]]
    want = {(nm, f) for nm, f in base for f in range(NFRAMES)}
    got = set(base)
    out.write("- 每个 key 都有 16 帧、无重复：%s（%s 共 %d 个 key）\n"
              % ("是" if got == want else "否，缺 %s / 多 %s"
                 % (sorted(want - got)[:5], sorted(got - want)[:5]),
                 IMPLS[0], len(base)))
    if got != want:
        ok = False

    bad_len = []
    for n in IMPLS:
        for k, v in impls[n].items():
            if len(v) != len(base[k]):
                bad_len.append((n, k))
    out.write("- 同一个 key 的**值个数**四组一致：%s\n"
              % ("是" if not bad_len else "否 → %s" % bad_len[:5]))
    if bad_len:
        ok = False

    nonfinite = [(n, k) for n in IMPLS for k, v in impls[n].items()
                 for x in v if not math.isfinite(x)]
    out.write("- 无 NaN / Inf：%s\n" % ("是" if not nonfinite else "否 → %s" % nonfinite[:5]))
    if nonfinite:
        ok = False

    out.write("- 每帧 8 项 × 16 帧 = %d 个 key\n" % (8 * NFRAMES))

    # 回归护栏：高精度下两侧的 conv1.out 必须落在 1 ulp 量级。
    # 这一条专门抓"用例输入装填错位"那一类 bug —— 它曾经真的发生过：
    # padding_rn 的行步长不匹配，让 C 侧 inputs[1] 拿到第 0 帧的填充区（恒 0），
    # 于是所有帧的输出全错，但 key 集合、长度、帧数**全都是对的**。
    guard = mx([abs(x - y) for f in range(NFRAMES)
                for x, y in zip(impls["py_high"][("conv1.out", f)],
                                impls["c_high"][("conv1.out", f)])])
    out.write("- 回归护栏（高精度 `conv1.out` 应 ≈1 ulp，现 %s）：%s\n"
              % (e(guard), "通过" if guard <= 1e-6 else "**不通过 → 用例装填可能错位**"))
    if guard > 1e-6:
        ok = False

    out.write("- 自检结论：**%s**\n\n" % ("通过" if ok else "不通过"))
    return ok


def ulp32(v):
    """float32 在 |v| 附近的间距（ulp）。

    两个 float32 相减，结果必然是 0 或者 ≥ 1 ulp —— 所以把 `|Δ|` 换算成
    "几个 ulp"，比看绝对数值有意义得多：**能用同一个刻度跨层比较**。

    只看正规数（本任务的量级都远离下溢区），次正规区间直接返回 2^-149。
    """
    v = abs(float(v))
    if v == 0.0:
        return 5e-324
    if v < 2.0 ** -126:                       # 次正规，罕见，给个保守值
        return 2.0 ** -149
    return 2.0 ** (math.floor(math.log2(v)) - 23)


def sec_pairs(impls, out):
    out.write("### 1. 四组 pairwise（`Δ = 前者 − 后者`）\n\n")
    out.write("**每一项的算法**：对每一帧 f、每个分量 i，算 `|A[f][i] − B[f][i]|`，\n")
    out.write("把这些数**全部堆进一个列表**，列表长度 = `帧数(16) × 维度`，然后对这个列表取 `max` / `mean`。\n")
    out.write("表里的 `max|Δ| ÷ ulp` 是把最大差换算成「几个 float32 的最小间距」——\n")
    out.write("同一个差值在不同量级的分量上含义完全不同，换算成 ulp 才能跨层比较。\n\n")
    items = sorted({k[0] for k in impls[IMPLS[0]]})
    for title, a, b, why in GROUPS:
        out.write("**%s** —— %s\n\n" % (title, why))
        out.write("| 打印项 | 维度 | 样本数 | max\\|Δ\\| | max\\|Δ\\| ÷ ulp | 最大值出现在 "
                  "| mean\\|Δ\\| |\n")
        out.write("|---|---|---|---|---|---|---|\n")
        for item in items:
            best = (-1.0, -1, -1, 0.0, 0.0)       # (|Δ|, frame, idx, A, B)
            acc = []
            n_elem = len(impls[a][(item, 0)])
            for f in range(NFRAMES):
                for i, (x, y) in enumerate(zip(impls[a][(item, f)], impls[b][(item, f)])):
                    d = abs(x - y)
                    acc.append(d)
                    if d > best[0]:
                        best = (d, f, i, x, y)
            _d, bf, bi, bx, by = best
            nulp = _d / ulp32(max(abs(bx), abs(by))) if _d else 0.0
            out.write("| `%s` | %d | %d | %s | **%.0f ulp** | 帧%d 分量%d | %s |\n"
                      % (item, n_elem, len(acc), e(mx(acc)), nulp, bf, bi, e(mean(acc))))
        out.write("\n")


def sec_compress(impls, out):
    """最后一层（dense）到底把误差压缩了多少 —— **实测，不靠推断**。

    为什么必须实测：`dense.in` 是 **96 维**、`dense.gains` 是 **2 维**，
    两边取 `max` **根本不是同一个统计量**；而且 `dense.in` 的最大差那一个分量，
    乘进 gains 时还要过它自己的那一列权重。所以"最后一层把误差压小了"这句话
    必须拆成两段分别量：

        dense.in 的 96 维差  --(线性层 x@W+b)-->  过 sigmoid 前的差  --(sigmoid)-->  gains 的差
    """
    import torch
    import rnn_unit as U

    _, cases = U.load_unit_cases("all")
    out.write("### 3.6 最后一层到底压缩了多少（实测）\n\n")
    out.write("`dense.in`(96 维) → `x@W+b`(2 维) → `sigmoid`(2 维)，逐段量差。对比组 `py_low vs c_low`。\n\n")
    case = cases["dense_out"]
    lin = U.Linear(case["nb_in"], case["nb_out"], case)
    d_in, d_pre, d_post = [], [], []
    for f in range(NFRAMES):
        cvals, pvals = impls["c_low"][("dense.in", f)], impls["py_low"][("dense.in", f)]
        d_in.append(mx([abs(x - y) for x, y in zip(cvals, pvals)]))
        ya = lin(torch.tensor(cvals, dtype=torch.float32)).reshape(-1)
        yb = lin(torch.tensor(pvals, dtype=torch.float32)).reshape(-1)
        d_pre.append(mx([abs(float(x) - float(y)) for x, y in zip(ya, yb)]))
        d_post.append(mx([abs(float(x) - float(y))
                          for x, y in zip(torch.sigmoid(ya), torch.sigmoid(yb))]))
    out.write("| 段 | 维度 | max\\|Δ\\| | 相对上一段 | 这一步在做什么 |\n")
    out.write("|---|---|---|---|---|\n")
    out.write("| `dense.in` | %d | %s | — | 上一层传进来的误差 |\n"
              % (case["nb_in"], e(max(d_in))))
    out.write("| 过 sigmoid 前 `x@W+b` | %d | %s | **%.2f×** | 96 项加权求和（有**平均效应**） |\n"
              % (case["nb_out"], e(max(d_pre)), max(d_pre) / max(d_in)))
    out.write("| 过 sigmoid 后 = `dense.gains` | %d | %s | **%.2f×** | sigmoid（导数值 ≤ 0.25） |\n"
              % (case["nb_out"], e(max(d_post)), max(d_post) / max(d_pre)))
    out.write("\n")
    out.write("→ 总压缩 **%.0f 倍**，其中线性层占 **%.1f 倍**、sigmoid 占 **%.1f 倍**。\n\n"
              % (max(d_in) / max(d_post), max(d_in) / max(d_pre), max(d_pre) / max(d_post)))


def sec_dense_in_trend(impls, out):
    """`dense.in` 的**完整 16 帧**逐帧差 + 跳变检测。

    为什么单看第 0 / 第 15 帧不够：如果中间某帧突增（例如某个分量跨过了某个非线性区），
    只报端点会把它藏起来。
    """
    out.write("### 3.5 `dense.in` 完整逐帧（跳变检测）\n\n")
    out.write("`dense.in` 是 96 维拼接，**每帧有 96 个分量**。下表是每帧 96 个 `|Δ|` 里的最大值。\n\n")
    out.write("| 帧 | " + " | ".join(t.split(" vs ")[0].replace("py_", "py") + "vs"
                                       + t.split(" vs ")[1].replace("_", "") for t, *_ in GROUPS)
              + " |\n")
    out.write("|---|" + "---|" * len(GROUPS) + "\n")
    series = {}
    for title, a, b, why in GROUPS:
        vals = [mx([abs(x - y) for x, y in zip(impls[a][("dense.in", f)],
                                               impls[b][("dense.in", f)])])
                for f in range(NFRAMES)]
        series[title] = vals
    for f in range(NFRAMES):
        out.write("| %d | %s |\n"
                  % (f, " | ".join(e(series[t][f]) for t, *_ in GROUPS)))
    out.write("\n")

    out.write("**跳变检测**：相邻帧的比值（比值 >2 或 <0.5 记为一次跳变）\n\n")
    out.write("| 组 | 最大单帧跳升 | 出现在 | 最大单帧跳降 | 出现在 | 非单调帧数 |\n")
    out.write("|---|---|---|---|---|---|\n")
    for title, a, b, why in GROUPS:
        v = series[title]
        ups = [(v[f] / v[f - 1], f) for f in range(1, NFRAMES) if v[f - 1] > 0]
        downs = [(v[f - 1] / v[f], f) for f in range(1, NFRAMES) if v[f] > 0]
        nonmono = sum(1 for f in range(1, NFRAMES) if v[f] < v[f - 1])
        u = max(ups)
        d = max(downs)
        out.write("| %s | %.2f× | 帧%d | %.2f× | 帧%d | %d/15 |\n"
                  % (title, u[0], u[1], d[0], d[1], nonmono))
    out.write("\n")


def sec_per_frame(impls, out):
    """逐帧 max|Δ|：用来区分"这一帧算错了"和"前几帧攒下来的轨迹分叉"。"""
    out.write("### 2. 逐帧 `max|Δ|`（判断是轨迹分叉还是单帧算错）\n\n")
    out.write("第 0 帧两侧输入严格同源（都从 conv1 的输入开始、mem/hidden 都是 0），"
              "所以第 0 帧的差就是**纯实现差**；之后逐帧增长的部分是**轨迹分叉**。\n\n")
    items = sorted({k[0] for k in impls[IMPLS[0]]})
    for title, a, b, why in GROUPS:
        out.write("**%s**\n\n" % title)
        out.write("| 帧 | " + " | ".join("`%s`" % i for i in items) + " |\n")
        out.write("|---|" + "---|" * len(items) + "\n")
        for f in range(NFRAMES):
            cells = []
            for item in items:
                d = [abs(x - y) for x, y in zip(impls[a][(item, f)], impls[b][(item, f)])]
                cells.append(e(mx(d)))
            out.write("| %d | %s |\n" % (f, " | ".join(cells)))
        out.write("\n")


def sec_dense_in_check(impls, out):
    """核实"dense.in 的 max|Δ| 逐位等于 gru3.state 的"这条说法。"""
    out.write("### 3. 交叉校验：`dense.in` 的最大差是不是就是 `gru3.state` 那一项\n\n")
    out.write("`dense.in` = `conv2.out`(24) + `gru1/2/3.state`(各 24) 的拼接，"
              "所以它的 `max|Δ|` 应当等于四个分量里**最大**的那个。\n\n")
    out.write("| 组 | 帧 | `dense.in` | `conv2.out` | `gru1` | `gru2` | `gru3` | 四者最大 | 一致? |\n")
    out.write("|---|---|---|---|---|---|---|---|---|\n")
    for title, a, b, why in GROUPS:
        for f in (0, 1, 7, 15):
            def m(item):
                return mx([abs(x - y) for x, y in
                           zip(impls[a][(item, f)], impls[b][(item, f)])])
            parts = [m("conv2.out"), m("gru1.state"), m("gru2.state"), m("gru3.state")]
            out.write("| %s | %d | %s | %s | %s | %s | %s | %s | %s |\n"
                      % (title, f, e(m("dense.in")), e(parts[0]), e(parts[1]),
                         e(parts[2]), e(parts[3]), e(max(parts)),
                         "是" if abs(m("dense.in") - max(parts)) < 1e-20 else "**否**"))
    out.write("\n")


def main():
    impls = load_all()
    out = sys.stdout
    ok = sec_sanity(impls, out)
    sec_pairs(impls, out)
    sec_per_frame(impls, out)
    sec_dense_in_check(impls, out)
    sec_dense_in_trend(impls, out)
    sec_compress(impls, out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

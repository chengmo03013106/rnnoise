#!/usr/bin/env python3
# coding=utf-8
"""业务层验收：这点数值误差会不会真的影响输出？

前面所有分析都在问「差多少、从哪来」。这里问的是**用户能不能感知**。

RNNoise 真正对外的东西只有两个：`dense.gains`（频带增益）和 `vad.out`（VAD 概率）。
本任务里它们的误差分别是 **1.3e-05 / 1.9e-05**。这个量级要紧吗？

用**代码里的真实判据**来量（都是读出来的硬事实，不是估计）：

  ① VAD 判决门限 = **0.5**
     `src/dump_rnn_io.c:173` `vad[SEQUENCE_LENGTH-1] = curr > .5;`
     而且它是在**序列级 Viterbi 平滑之后**才判决的，第 181-186 行还会把判决
     向两侧蔓延（`if (vad[i+1]) vad[i] = 1;`）→ 单帧的微小扰动很容易被平滑掉。

  ② 增益的**限速台阶**：`src/denoise.c:483` `g[i] = MAX16(g[i], .6f*st->lastg[i])`
     这是个 `max` 台阶。两个实现的 `g` 若分居 `0.6*lastg` 两侧，
     会不会一个被抬上去、一个没被抬，从而把误差**放大**？→ 本脚本给上界。

  ③ 输出量化：对外是 16 bit PCM → LSB = 1/32768 ≈ **3.05e-05**（相对满量程）。

用法：
    sh scripts/err_stats/collect_all.sh
    uv run python scripts/err_stats/business_check.py
"""

import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "scripts", "err_stats", "raw_all")

sys.path.insert(0, os.path.join(ROOT, "scripts", "err_stats"))

from err_stat import parse, e  # noqa: E402

IMPLS = ("c_high", "c_low", "py_high", "py_low")
NFRAMES = 16
VAD_THRESHOLD = 0.5
LSB16 = 1.0 / 32768.0


def sec_vad(impls, out):
    """① VAD 判决门限。"""
    out.write("### ① VAD 判决门限（`> 0.5`）\n\n")
    out.write("| 帧 | " + " | ".join("`%s`" % n for n in IMPLS)
              + " | 四者极差 | 离 0.5 的裕量 |\n")
    out.write("|---|" + "---|" * (len(IMPLS) + 2) + "\n")
    worst = 1.0
    flip = []
    for f in range(NFRAMES):
        vs = [impls[n][("vad.out", f)][0] for n in IMPLS]
        spread = max(vs) - min(vs)
        margin = min(abs(v - VAD_THRESHOLD) for v in vs)
        worst = min(worst, margin)
        # 判决是否一致（实际用 0.5 判）
        decided = {v > VAD_THRESHOLD for v in vs}
        if len(decided) > 1:
            flip.append(f)
        out.write("| %d | %s | %s | %s |\n"
                  % (f, " | ".join("%.9f" % v for v in vs), e(spread), e(margin)))
    out.write("\n")
    out.write("- 16 帧里 `vad_prob` 离门限 0.5 的**最小裕量** = **%s**\n" % e(worst))
    out.write("- 四组判决不一致的帧：%s\n" % (flip if flip else "**无**"))
    out.write("- 而本任务实测的最大误差是 **1.9e-05**，比最小裕量小 **%.0f 倍**\n\n"
              % (worst / 1.9e-05 if worst else float("inf")))


def sec_vad_real(out):
    """用仓库里真实音频的 dump（500 帧）看 vac 分布离门限有多近。"""
    path = os.path.join(ROOT, "src", "default_int8.txt")
    if not os.path.isfile(path):
        out.write("（没有 `src/default_int8.txt`，跳过真实数据分布）\n\n")
        return
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = re.search(r"vad:\s*([0-9.]+)", line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        out.write("（`src/default_int8.txt` 里没解析出 vad 值，跳过）\n\n")
        return
    near = [v for v in vals if abs(v - VAD_THRESHOLD) <= 1.9e-05]
    out.write("**真实数据参照**（`src/default_int8.txt`，仓库里真实音频跑出来的 %d 帧）：\n\n" % len(vals))
    out.write("- `vad_prob` 范围 `[%.6f, %.6f]`，离 0.5 最近的一帧是 **%s**\n"
              % (min(vals), max(vals), e(min(abs(v - VAD_THRESHOLD) for v in vals))))
    out.write("- 落在门限 ±1.9e-05 之内的帧数：**%d / %d（%.1f%%）**\n\n"
              % (len(near), len(vals), 100.0 * len(near) / len(vals)))


def sec_gain_stair(impls, out):
    """② 增益的限速台阶会不会放大误差。"""
    out.write("### ② 增益的限速台阶 `g = MAX16(g, .6*lastg)` 会不会放大误差\n\n")
    out.write("设两个实现对同一频带算出 `g1`、`g2`（`|g1-g2| = δ`），台阶值 `t = .6*lastg`。\n\n")
    out.write("| 情形 | 结果 | 差 |\n|---|---|---|\n")
    out.write("| 两者都在台阶之上（`g1,g2 > t`） | 都不被抬 | `δ` |\n")
    out.write("| 两者都在台阶之下（`g1,g2 < t`） | 都被抬成 `t` | `0` |\n")
    out.write("| 一个上一个下 | 一个被抬、一个不动 | `\\|g2 − t\\| ≤ δ`（因为 `t` 夹在两者之间） |\n\n")
    out.write("→ **台阶不会把差放大**，任何情形下 `max|Δ| ≤ δ`。"
              "所以 `MAX16` 这一步**不是**误差放大器。\n\n")

    d = [abs(impls["py_low"][("dense.gains", f)][i] - impls["c_low"][("dense.gains", f)][i])
         for f in range(NFRAMES) for i in (0, 1)]
    g = [v for f in range(NFRAMES) for v in impls["c_low"][("dense.gains", f)]]
    rel = max(d) / (sum(abs(v) for v in g) / len(g))
    out.write("**实测**：`py低 vs C低` 的 `dense.gains` 最大绝对误差 **%s**；"
              "C 低精度增益的均值 %.4f → **相对误差 %s**\n\n" % (e(max(d)), sum(g) / len(g), e(rel)))
    return max(d), rel


def sec_quant(out, gain_abs, gain_rel):
    """③ 16 bit 输出的量化对照。数值全部来自 ② 的实测，不硬编码。"""
    out.write("### ③ 16 bit 输出的量化对照\n\n")
    out.write("| 量 | 值 |\n|---|---|\n")
    out.write("| 增益最大绝对误差（实测） | %s |\n" % e(gain_abs))
    out.write("| 增益相对误差（实测） | %s |\n" % e(gain_rel))
    out.write("| 16 bit LSB（相对满量程） | %s |\n" % e(LSB16))
    out.write("| 相对误差 ÷ LSB | %.2f LSB |\n" % (gain_rel / LSB16))
    out.write("| 相对误差换算成电平 | **%.1f dBFS**（= `20*log10(%s)`） |\n\n"
              % (20 * math.log10(gain_rel), e(gain_rel)))
    out.write("→ 增益的相对误差只有约 **%.1f 个 16-bit LSB**，落到输出上是"
              "**最低有效位级别的抖动（≈ %.0f dBFS）**，比 16 bit 的量化噪声底还低，"
              "属于听不出来的量级。\n\n"
              % (gain_rel / LSB16, 20 * math.log10(gain_rel)))


def sec_gru_state_note(out):
    """GRU 隐状态误差虽然没有直接对外，但它是增益误差的上游。"""
    out.write("### ④ 网络内部误差（`gruN.state` 1e-04）要不要紧\n\n")
    out.write("`gruN.state` 的 1e-04 是**中间量**，不直接对外。它要紧了只有两种路径：\n\n")
    out.write("1. **沿链路传到输出**：`dense.in` 1.1e-04 → `dense.gains` **1.3e-05**、"
              "`vad.out` **1.9e-05**，被最后一层 sigmoid **压缩了约 8 倍**；\n")
    out.write("2. **跨帧累积**：`MAX16(g, .6*lastg)` 的台阶不放大误差（见 ②），"
              "而 `lastg` 只由 `g` 和输入能量决定（输入能量两侧完全相同）。\n\n")
    out.write("→ 所以 **内部 1e-04 / 1e-05 的量级，到对外输出只剩 1e-05**，"
              "再经过限速台阶与序列级 VAD 平滑，最终不可感知。\n\n")


def main():
    impls = {n: parse(os.path.join(RAW, n + ".txt")) for n in IMPLS}
    out = sys.stdout
    out.write("## 业务层验收（阈值 / 台阶层面）\n\n")
    out.write("问的是：**这点数值误差会不会真的影响输出**。判据全部取自源码，不是估计。\n\n")
    sec_vad(impls, out)
    sec_vad_real(out)
    gain_abs, gain_rel = sec_gain_stair(impls, out)
    sec_quant(out, gain_abs, gain_rel)
    sec_gru_state_note(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

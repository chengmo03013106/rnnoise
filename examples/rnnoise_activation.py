#!/usr/bin/env python3
# coding=utf-8
"""RNNoise 的激活函数近似 —— 从 `examples/gru.py` 拆出来的独立模块。

对应关系（关键是**两边"低精度"落在哪个算法上**）：

    C 侧                                 本模块
    ------------------------------------  ---------------------------
    `src/vec.h:337`   tanh_approx(x)      tanh_approx(x)
    `src/vec_avx.h:398` tanh8_approx(x)   同上（同一个有理式）
    `src/vec.h:353`   sigmoid_approx(x)   sigmoid_from_tanh(x)   = 0.5 + 0.5*tanh_approx(0.5x)
    `src/vec_avx.h:426` sigmoid8_approx(x) sigmoid_approx(x)     = 直接四阶有理式
    `src/nnet_arch.h:83` HIGH_ACCURACY      torch.tanh / torch.sigmoid

**`sigmoid_from_tanh` 和 `sigmoid_approx` 是同一个函数**：把 u = x/2 代进前者、
分子分母同乘 16，得到的系数正好是后者系数的 **64 倍**，在 `num/den` 里约掉
（`src/vec_avx.h:418-424` 的注释写的就是这个推导）。保留两种写法是为了让
"同一函数、不同求值顺序"的浮点舍入差异可测。

两点必须记住：

1. 这些多项式本身的逼近误差：在 x = −8…+8 的网格上，`sigmoid` 最大 2.933e-05、
   `tanh` 最大 5.966e-05（实测见 `scripts/err_stats/bench_act.py`）。
2. C 侧真正跑的时候还会再过一层 `_mm256_rcp_ps` / `_mm_rcp_ps` 硬件倒数
   （相对误差上界 1.5×2⁻¹²≈3.66e-04），而本模块用的是精确除法 ——
   所以**同样的多项式，C 低精度与 Python 低精度之间仍有 ~1e-04 的差**
   （实测 sigmoid 9.722e-05 / tanh 2.599e-04）。

整网级别的测量见 `doc/RNN_UNIT整网误差分析.md`。

原名对照（从 `gru_scratch.py` 搬过来时改的名）：
    `_rnnoise_tanh_approx`    -> `tanh_approx`
    `_rnnoise_sigmoid_approx` -> `sigmoid_approx`
    `_check_sigmoid_diff`     -> `check_sigmoid_diff`（签名从 `(zrh, hidden_size, recur)`
                                 改成直接收 pre-activation；原版内部调用了并不存在的
                                 `_rnnoise_sigmoid`，一跑就 NameError）
"""

import sys

import torch


def tanh_approx(x):
    """C 的 `tanh_approx()`：四阶有理式 + clamp 到 [-1, 1]。"""
    n0, n1, n2 = 952.52801514, 96.39235687, 0.60863042
    d0, d1, d2 = 952.72399902, 413.36801147, 11.88600922

    x2 = x * x
    num = (n2 * x2 + n1) * x2 + n0
    den = (d2 * x2 + d1) * x2 + d0
    y = x * num / den

    return torch.clamp(y, -1.0, 1.0)


def sigmoid_approx(x):
    """C 的 `sigmoid8_approx()`：直接四阶有理式 + clamp 到 [0, 1]。

    系数就是 tanh 那套除以 2 的幂得到的，所以和 `sigmoid_from_tanh` 数学等价。
    """
    n0, n1, n2 = 238.13200378, 6.02452230, 0.00950985
    d0, d1, d2 = 952.72399902, 103.34200287, 0.74287558

    x2 = x * x
    num = (n2 * x2 + n1) * x2 + n0
    den = (d2 * x2 + d1) * x2 + d0
    y = 0.5 + x * num / den

    return torch.clamp(y, 0.0, 1.0)


def sigmoid_from_tanh(x):
    """C 的 `vec.h` 里那种写法：`0.5 + 0.5*tanh_approx(0.5x)`。

    和 `sigmoid_approx` 是同一个函数（见模块头注释），单独留一个是为了做对照：
    同一个数学式、不同的求值顺序，在 float32 下会有 ~1 ulp 的舍入差。
    """
    return 0.5 + 0.5 * tanh_approx(0.5 * x)


def pick(low_accuracy):
    """按精度档返回 `(sigmoid, tanh)` 两个可调用对象。

    `low_accuracy` 接受四种取值 —— 前两种是「整档」，后两种是**拆到单个门**：

        True / "low"    两个都用多项式近似（= C 默认的 `vec_sigmoid` / `vec_tanh`）
        False / "high"  两个都用 torch 自带（= C 的 `-DHIGH_ACCURACY`）
        "sigmoid"       **只** sigmoid 用多项式，tanh 仍用 torch
        "tanh"          **只** tanh 用多项式，sigmoid 仍用 torch

    ★ 为什么要后两种：GRU 里 sigmoid 管 z（更新门）和 r（重置门），
      tanh 管候选状态 h̃。这两者出错后**被状态反馈放大**的程度可能完全不同，
      整档切换只能看到"合起来的效果"，拆开才能定位是哪个门在驱动放大。
      用在 `scripts/err_stats/amplify.py`。
    """
    if low_accuracy is True or low_accuracy == "low":
        return sigmoid_approx, tanh_approx
    if low_accuracy is False or low_accuracy == "high":
        return torch.sigmoid, torch.tanh
    if low_accuracy == "sigmoid":
        return sigmoid_approx, torch.tanh
    if low_accuracy == "tanh":
        return torch.sigmoid, tanh_approx
    raise ValueError("low_accuracy 只能是 True/False/'low'/'high'/'sigmoid'/'tanh'，收到 %r"
                     % (low_accuracy,))


def check_sigmoid_diff(pre_act, out=sys.stdout):
    """把 sigmoid 的几种实现摆在一起比一遍，打印 max|Δ|。

    对比对象：
        精确        `1/(1+exp(-x))`
        torch       `torch.sigmoid`（float32 下的实现，约 1 ulp）
        AVX 式      `sigmoid_approx`（= C 的 `sigmoid8_approx`，Python 低精度用的）
        tanh 式     `sigmoid_from_tanh`（= C 的标量 `vec.h` 写法）

    注意本函数**不含** C 的 `_mm256_rcp_ps` 硬件倒数误差 —— 那个只能在 C 侧测。
    """
    pre_act = torch.as_tensor(pre_act, dtype=torch.float32)
    z_exact = 1.0 / (1.0 + torch.exp(-pre_act))
    z_torch = torch.sigmoid(pre_act)
    z_avx = sigmoid_approx(pre_act)
    z_tanh = sigmoid_from_tanh(pre_act)

    def gap(name, a, b):
        out.write("max |%s - %s|: %.8e\n" % (name[0], name[1], float(torch.max(torch.abs(a - b)))))

    out.write("样本数 %d\n" % pre_act.numel())
    gap(("torch", "exact"), z_torch, z_exact)
    gap(("AVX式", "exact"), z_avx, z_exact)
    gap(("tanh式", "exact"), z_tanh, z_exact)
    gap(("AVX式", "tanh式"), z_avx, z_tanh)
    gap(("torch", "AVX式"), z_torch, z_avx)


def selftest():
    """`pick()` 的取值自检：4 种取值 + 老 bool 写法的回归 + 非法值必须报错。

    `pick()` 是"高/低精度"的唯一分派点，它错了整网误差分析就全错，
    所以这里把 6 种入参（含 2 种历史写法）逐个钉住。
    """
    cases = [
        (True, sigmoid_approx, tanh_approx),      # 老写法（bool）
        (False, torch.sigmoid, torch.tanh),       # 老写法（bool）
        ("low", sigmoid_approx, tanh_approx),
        ("high", torch.sigmoid, torch.tanh),
        ("sigmoid", sigmoid_approx, torch.tanh),  # 只换 sigmoid（实验用）
        ("tanh", torch.sigmoid, tanh_approx),     # 只换 tanh（实验用）
    ]
    for arg, want_sig, want_tanh in cases:
        got_sig, got_tanh = pick(arg)
        assert got_sig is want_sig, "pick(%r) 的 sigmoid 不对" % (arg,)
        assert got_tanh is want_tanh, "pick(%r) 的 tanh 不对" % (arg,)
    for bad in (1, 0, None, "Low", "HIGH", "both"):
        try:
            pick(bad)
        except ValueError:
            continue
        raise AssertionError("pick(%r) 本该报 ValueError" % (bad,))
    return len(cases)


if __name__ == "__main__":
    # 手动跑一遍（不依赖 C、不依赖用例）：
    #     uv run examples/rnnoise_activation.py
    print("pick() 自检：%d 种取值全部正确，非法值都能报错" % selftest())
    # 只想看"四种 sigmoid 在 [-3,3] 上彼此差多少"时用它；
    # 整网级别的误差分析走 scripts/err_stats/ 下那几个脚本。
    check_sigmoid_diff(torch.linspace(-3.0, 3.0, 7))

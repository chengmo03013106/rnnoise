#!/usr/bin/env python3
# coding=utf-8
"""误差统计用的**数值工具模块**（不是报告生成器）。

`scripts/err_stats/` 下的所有脚本都从这里取两样东西：

    parse(path)   读协议文件 `[<名字>#<帧>] v0 v1 ...` -> `{(名字, 帧): [float, ...]}`
    f32(v)        舍入到 float32
    mx / mean / e 取最大 / 取平均 / 格式化

## 为什么 `parse()` 必须做一次 `f32()`

两侧都用 `%.8e` 打印，只有 **9 位有效数字**；而 float32 恰好需要 9 位才能唯一还原。
所以读进来的十进制**必须先还原成 float32** 再比较 ——
当 double 用的话，比的是"打印精度"而不是真实值，会凭空多出 ~5e-10 的假差异。
自查判据：两个 float32 之差只能是 0 或 ≥1 ULP，出现别的值就是解析口径错了。

## 为什么 `parse()` 遇到重复 key 直接报错

同一个 `(名字, 帧)` 出现两次说明上游打印逻辑有问题（例如帧号没有随循环递增）。
用 dict 存会**静默覆盖**前面那一份，把整帧数据吃掉 —— 报错比静默强。
"""

import struct


def f32(v):
    """舍入到 float32 —— 等价于 C 里 `float` 的存储精度。"""
    return struct.unpack("f", struct.pack("f", float(v)))[0]


def parse(path):
    """读协议文件 -> `{(name, frame): [float, ...]}`。"""
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if not line.startswith("["):
                raise SystemExit("%s:%d 不是协议行: %r" % (path, lineno, line))
            head, _, body = line[1:].partition("]")
            name, _, frame = head.rpartition("#")
            if not frame.isdigit():
                raise SystemExit("%s:%d 帧号不是数字: %r" % (path, lineno, line))
            key = (name, int(frame))
            if key in out:
                raise SystemExit("%s:%d key 重复: %r" % (path, lineno, key))
            out[key] = [f32(v) for v in body.split()]
    return out


def mx(vals):
    return max(vals) if vals else float("nan")


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def e(v):
    return "%.3e" % v

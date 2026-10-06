#!/usr/bin/env python3
# coding=utf-8
"""用例（case）加载器：从 `scripts/unit_cases/*.json` 读出网络各层参数。

原本这些代码在 `examples/rnn_unit.py` 里，只有 `rnn_unit.py` 用得到。
但 `gru.py`、`onnx_demo.py` 都可能需要同一份用例，所以抽成独立模块：

    from loader import load_unit_cases, case_path, GRU_LAYERS, ...

同目录的模块之间用**裸 import**（`from loader import ...`），
和 `rnn_unit.py` 里 `from gru import GRUScratch` 是同一种做法 ——
Python 跑 `examples/xxx.py` 时会把脚本所在目录放进 `sys.path`，
所以不需要额外的路径处理。

## 契约：规则必须和 C 侧一致

`examples/rnn_unit.c` 里有 `load_conv1d_case()` / `load_gru_case()` /
`load_dense_case()` / `auto_case_path()`。**字段名、长度、可选性、报错时机**
都按 C 侧那一套来；两边读出不同的数据，对比结果就是假的。

## 严格模式

读不到文件 / 字段缺失或长度不对，都**直接报错退出**（不退回内置保底数据）。
对应 C 侧 `open_case()` 打错误信息 + 返回 -1。

## 错误分两类

| 情况 | 谁负责补打 CLI 用法 |
|---|---|
| **用例文件不存在** | 调用方（`rnn_unit.py` 的 `main()`）—— 只有它知道本程序的用法 |
| **文件在但内容不合格** | 不用，错误信息本身就是全部原因 |

所以下面抛的是 `CaseError` 而不是直接 `SystemExit`：
`CaseError.missing` 告诉调用方该不该补打用法。
"""

import json
import os

# ===================== 用例的规模常量 =====================
# 这些常量描述的是**用例文件的格式**（字段该多长），所以跟加载器待在一起。
# 必须和 examples/rnn_unit.h 的 CASE_* 宏、以及 examples/rnn_unit.c 一致
# （那边宏名带 CASE_ 前缀，避免和 src/rnnoise_data.h 的真实模型尺寸撞名）。
# 规模 = 真实模型尺寸 //16（conv1 是 65->128，conv2 是 128->384）。

NB_FRAMES = 16          # 必须和 examples/rnn_unit.c 的 NB_FRAMES 一致

# conv1d 单元要跑的两层：层名 + 每帧输入宽 + 输出宽
CONV_LAYERS = (
    ("conv1", 4, 8),
    ("conv2", 8, 24),
)
# 注意：conv2 的 `inputs` 是可选的 —— 整网里它的每帧输入就是 conv1 的每帧输出，
# 用例里没有这一项（缺了不算错误，只是这一层没有可独立跑的帧）。

# gru 单元要跑的三层；三层真实规模相同（//16：每帧输入 384->24、N 384->24、门输出 3N 1152->72）。
# 注意 gru 的每帧输入宽 = conv2 的输出宽，N 也等于它。
GRU_LAYERS = ("gru1", "gru2", "gru3")
GRU_N = 24                               # 隐层宽度 N（= 每帧输入宽 = conv2 输出宽）
GRU_INPUT_SIZE = 24                      # 输入宽度
GRU_OUTPUT_SIZE = 3 * GRU_N              # 三个门拼成的输出宽度 = 72

# 整网输出层（对应 src/rnnoise_data.h 的 dense_out / vad_dense），参数在 dense.json 里。
# 输入宽 = 整网拼接后的宽度 = conv2 输出 + 3 x gru 输出 = 24 + 72 = 96（真实 1536）。
DENSE_IN_SIZE = 24 + 3 * GRU_N           # 96
DENSE_LAYERS = (
    ("dense_out", DENSE_IN_SIZE, 2),     # gains，真实 32 -> 2（//16）
    ("vad_dense", DENSE_IN_SIZE, 1),     # VAD，单标量输出不再缩
)


class CaseError(SystemExit):
    """用例数据有问题。

    继承 `SystemExit` 是为了**保持原样**：不接的话 `SystemExit` 会把消息打到
    stderr 并以退出码 1 退出，和之前的行为一致。

    `missing=True` 表示"用例文件不存在"—— 调用方据此决定要不要补打 CLI 用法。
    """

    def __init__(self, msg, missing=False):
        super().__init__(msg)
        self.missing = missing


# ===================== 字段校验 =====================

def _is_num_list(value, n):
    """是不是长度为 n 的**数字**数组。

    只认真正的数字（int/float，且排除 bool），刻意不做 `float(x)` 转换：
    C 侧那个极简解析器只认裸数字字面量，字符串 "1.5" 和 true/false 它都读不出来，
    两边规则必须完全一致，否则会出现"py 读得进去、C 读不进去"的假差异。
    """
    return (isinstance(value, list) and len(value) == n
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value))


def _is_frames(value, width):
    """是不是非空、帧数不超限、且每帧长度都对的多帧输入。"""
    return (isinstance(value, list) and 0 < len(value) <= NB_FRAMES
            and all(_is_num_list(f, width) for f in value))


def _read_json(path):
    """读用例 JSON。打不开或解析失败都算错误（严格模式）。

    对应 C 侧 open_case()：那边打错误信息 + 返回 -1，这边抛 CaseError。
    """
    if not path or not os.path.isfile(path):
        raise CaseError("[rnn_unit.py] error: 没有用例文件 %s"
                        "（scripts/unit_cases/<用例文件名>.json 不存在，也没在命令行上指定）"
                        % (path,), missing=True)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise CaseError("[rnn_unit.py] json error: 读取 %s 失败（%s）" % (path, exc))


def _fail(msg):
    """用例数据不合格：报错退出（对应 C 侧 load_*_case() 返回 -1）。"""
    raise CaseError("[rnn_unit.py] json error: %s" % msg)


def _read_layer(data, layer):
    """从用例里取出某一层的子对象。"""
    sub = data.get(layer)
    if not isinstance(sub, dict):
        _fail("用例里没有 %s 这一层" % layer)
    return sub


def _need_num_list(sub, key, n, path):
    """取 <path> 下长度必须正好是 n 的数字数组，不合格就报错退出。"""
    value = sub.get(key)
    got = len(value) if isinstance(value, list) else 0
    if not _is_num_list(value, n):
        _fail("字段 %s 缺失或长度不对（需要 %d，实际 %d）" % (path, n, got))
    return list(value)


def _need_frames(sub, key, width, path):
    """取 <path> 下的多帧输入：1..NB_FRAMES 帧、每帧必须正好 width 个数。"""
    value = sub.get(key)
    if not _is_frames(value, width):
        _fail("字段 %s 缺失、帧数为空/超上限，或某帧不是 %d 个数" % (path, width))
    return [list(f) for f in value]


# ===================== 三个单元的加载器 =====================

def _load_conv1d_case(path):
    """conv1d 用例：每层各自的 bias / float_weights（+ 只有 conv1 才有 inputs）。

    规则必须和 examples/rnn_unit.c 的 load_conv1d_case() 完全一致，
    否则两边会读到不同的数据，对比结果就是假的。返回 {层名: case}。
    """
    data = _read_json(path)
    cases = {}
    for layer, in_size, nb_out in CONV_LAYERS:
        sub = _read_layer(data, layer)
        n_w = 3 * in_size * nb_out
        case = {
            "layer": layer,
            "nb_in": in_size,
            "nb_out": nb_out,
            "bias": _need_num_list(sub, "bias", nb_out, "%s.bias" % layer),
            "float_weights": _need_num_list(sub, "float_weights", n_w,
                                            "%s.float_weights" % layer),
        }
        # inputs 可选：conv2 在整网里由 conv1 的输出喂，用例里没有这一项，
        # 那就当"没有可独立跑的帧"（不是错误）；有这一项但格式不对，仍然报错。
        case["inputs"] = (_need_frames(sub, "inputs", in_size, "%s.inputs" % layer)
                          if "inputs" in sub else [])
        cases[layer] = case
    return cases


GRU_FIELDS = (
    ("input_weights_float", GRU_INPUT_SIZE * GRU_OUTPUT_SIZE),
    ("input_bias", GRU_OUTPUT_SIZE),
    ("recurrent_weights_float", GRU_N * GRU_OUTPUT_SIZE),
    ("recurrent_bias", GRU_OUTPUT_SIZE),
    ("state", GRU_N),
)


def _load_gru_case(path):
    """gru 用例：每层各自的 input_*/recurrent_* 四个矩阵 + state（+ 可选的 inputs）。

    三层字段完全相同，所以共用一张字段表；规则必须和 examples/rnn_unit.c 的
    load_gru_case() 完全一致。返回 {层名: case}。
    """
    data = _read_json(path)
    cases = {}
    for layer in GRU_LAYERS:
        sub = _read_layer(data, layer)
        case = {"layer": layer}
        for field, n in GRU_FIELDS:
            case[field] = _need_num_list(sub, field, n, "%s.%s" % (layer, field))
        # inputs 可选：gru2 / gru3 在整网里由上一层的隐状态喂，用例里没有这一项，
        # 那就当"这一层没有可独立跑的帧"（不是错误）；有这一项但格式不对，仍然报错。
        case["inputs"] = (_need_frames(sub, "inputs", GRU_INPUT_SIZE, "%s.inputs" % layer)
                          if "inputs" in sub else [])
        cases[layer] = case
    return cases


def _load_dense_case(path):
    """整网输出层用例（dense.json）：dense_out / vad_dense 各自的 bias + float_weights。

    这两层没有记忆、也没有输入序列，所以用例里只有参数。
    规则必须和 examples/rnn_unit.c 的 load_dense_case() 完全一致。返回 {层名: case}。
    """
    data = _read_json(path)
    cases = {}
    for layer, nb_in, nb_out in DENSE_LAYERS:
        sub = _read_layer(data, layer)
        cases[layer] = {
            "layer": layer,
            "nb_in": nb_in,
            "nb_out": nb_out,
            "bias": _need_num_list(sub, "bias", nb_out, "%s.bias" % layer),
            "float_weights": _need_num_list(sub, "float_weights", nb_in * nb_out,
                                            "%s.float_weights" % layer),
        }
    return cases


# ===================== 用例文件：找 + 读 =====================

# 逻辑名 -> 解析函数。每个函数只认自己那份文件的层级结构，
# 字段名和长度必须和 examples/rnn_unit.c 的 load_<unit>_case() 一致。
CASE_LOADERS = {
    "conv": _load_conv1d_case,
    "gru": _load_gru_case,
    "dense": _load_dense_case,
}

# 单元 -> 读哪几份用例（逻辑名，顺序 = 读取顺序）。
# 第一份是该单元「自己那份」，命令行 argv[2] 只能覆盖它；
# 整网 all 另外要的 gru.json / dense.json 固定走默认路径。
UNIT_CASES = {
    "conv1d": ("conv",),
    "gru":    ("gru",),
    "all":    ("conv", "gru", "dense"),
    "all2":   ("conv", "gru", "dense"),   # 走 RNNoiseMo 类的整网，和 all 读同一批用例
}
# 注意：examples/linear.py 的线性层手工实验**不是单元**（没进 rnn_unit.py 的 UNITS），
# 所以这里没有 "linear" 这一项 —— 它直接借 conv1d 那份用例（load_unit_cases("conv1d")）。


def case_path(name):
    """按约定定位用例文件：<仓库根>/scripts/unit_cases/<name>.json

    必须和 examples/rnn_unit.c 的 auto_case_path() 找同一个文件。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "scripts", "unit_cases", "%s.json" % name)


def load_unit_cases(unit, cli_path=None):
    """读一个单元需要的全部用例，返回 (用到的文件列表, {层名: case})。

    cli_path 就是命令行上的 argv[2]，只覆盖该单元「自己那份」用例；
    其余几份一律走默认路径（用户要求：all 的路径默认，不用指定）。
    读不到 / 字段不合格都会抛 CaseError（严格模式）。
    """
    paths, cases = [], {}
    for i, name in enumerate(UNIT_CASES[unit]):
        path = cli_path if (i == 0 and cli_path) else case_path(name)
        paths.append(path)
        cases.update(CASE_LOADERS[name](path))
    return paths, cases

#!/usr/bin/env python3
# coding=utf-8
"""练习 PyTorch：把 src/*.c 里的神经网络函数用 PyTorch 重新实现，逐项对比

输出协议（和 examples/rnn_unit.c 完全一致，驱动器靠它配对）:
    [<层名>.<输出名>#<帧>] <v0> <v1> ...

  注意第一段是**层名**而不是单元名：conv1d 单元依次跑 conv1 / conv2
  （两者共用 compute_generic_conv1d），gru 单元依次跑 gru1 / gru2 / gru3
  （三者共用 compute_generic_gru），靠层名把同名的输出项区分开。

用例数据来源（和 C 侧完全一致）:
    ① argv[2] 显式指定的 JSON（只覆盖该单元「自己那份」用例）
    ② 按约定自动定位 <仓库根>/scripts/unit_cases/<用例文件名>.json
  严格模式：读不到 / 必需字段对不上，直接报错退出（不再退回内置保底数据）。

  每份用例装的是「同一算子的多层参数」：
      conv.json   conv1 / conv2 的参数（+ conv1 的输入序列）
      gru.json    gru1 / gru2 / gru3 的参数（+ gru1 的输入序列）
      dense.json  dense_out / vad_dense 的参数（整网输出层，没有输入序列）
  「单元 -> 读哪几份」写在下面的 UNIT_CASES 里；整网 all 三份都要，路径全走默认。

用法（本仓库用 uv 管理 python 环境）:
    uv run examples/rnn_unit.py conv1d                                # 自动找用例文件（= conv.json）
    uv run examples/rnn_unit.py all                                   # 整条链路，路径全默认
    uv run examples/rnn_unit.py conv1d scripts/unit_cases/conv.json    # 显式指定
    uv run scripts/unit_check.py                                       # 通常用这个，会自动跑双方并对比
"""

import json
import os
import sys

import torch
from torch import nn as nn
from gru_scratch import GRUScratch as GRUScratch

NB_FRAMES = 16          # 必须和 examples/rnn_unit.c 的 NB_FRAMES 一致

# conv1d 单元要跑的两层：层名 + 每帧输入宽 + 输出宽。
# 必须和 examples/rnn_unit.h 的 CASE_CONV1_IN / CASE_CONV1_OUT ... 一致
# （那边宏名带 CASE_ 前缀，避免和 src/rnnoise_data.h 的真实模型尺寸撞名）。
# 规模 = 真实模型尺寸 //16（conv1 是 65->128，conv2 是 128->384）。
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
    """读用例 JSON。打不开或解析失败都算错误，直接退出（严格模式）。

    对应 C 侧 open_case()：那边打错误信息 + 返回 -1，这边抛 SystemExit。
    文件找不到属于"启动就错"，顺手把完整用法打出来。
    """
    if not path or not os.path.isfile(path):
        usage(os.path.basename(sys.argv[0]) if sys.argv else "rnn_unit.py")
        raise SystemExit("[rnn_unit.py] error: 没有用例文件 %s"
                         "（scripts/unit_cases/<用例文件名>.json 不存在，也没在命令行上指定）"
                         % (path,))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit("[rnn_unit.py] json error: 读取 %s 失败（%s）" % (path, exc))


def _fail(msg):
    """用例数据不合格：报错退出（对应 C 侧 load_*_case() 返回 -1）。"""
    raise SystemExit("[rnn_unit.py] json error: %s" % msg)


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
    "linear": ("conv",),                     # 单层线性层的手工实验，借用 conv1 的参数
    "gru":    ("gru",),
    "all":    ("conv", "gru", "dense"),
}


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
    读不到 / 字段不合格都会直接报错退出（严格模式）。
    """
    paths, cases = [], {}
    for i, name in enumerate(UNIT_CASES[unit]):
        path = cli_path if (i == 0 and cli_path) else case_path(name)
        paths.append(path)
        cases.update(CASE_LOADERS[name](path))
    return paths, cases

def emit(unit, output, frame, values, out=sys.stdout):
    """按协议打印一项。格式化必须和 C 的 "%.8e" 一致。"""
    out.write("[%s.%s#%d] %s\n" % (
        unit, output, frame,
        " ".join("%.8e" % float(v) for v in values),
    ))


# ===================== 待补单元 =====================
# conv1d / gru 跑通后按同样的模式往下加：
#   - linear_demo     : compute_linear   (sgemv / cgemv8x4 / sparse_*)
#   - dense_demo      : compute_generic_dense
#   - activation_demo : compute_activation (tanh_approx / sigmoid_approx)


class Linear(nn.Module):
    def __init__(self,nb_input,nb_output,case):
        super().__init__()
        self.register_buffer(
            'weights',
            torch.tensor(case["float_weights"][:nb_input*nb_output], 
                dtype=torch.float32).reshape(nb_input, nb_output)
        )
        # C 的权重布局是 weights[输入 j][输出 i]（sgemv 的 col_stride = nb_outputs）
        # PyTorch 的 nn.Linear 是 [输出, 输入]，所以这里直接写 x @ w，不做转置
        # 因为不需要训练，所以 requires_grad 不需要设置 True
        assert self.weights.shape[0] == nb_input
        self.register_buffer(
            'bias',
            torch.tensor(case["bias"][:nb_output], dtype=torch.float32)
        )
        assert len(self.bias.shape) == 1 and self.bias.shape[0] == nb_output

    def forward(self, x):
        x = x.reshape(-1, self.weights.shape[0])
        y = x@self.weights + self.bias 
        return y

def linear_demo(cases, low_accuracy=False):
    """单层线性层的手工实验：直接拿 conv1 那一层的参数跑一次。"""
    layer, in_size, nb_out = CONV_LAYERS[0]
    net = Linear(in_size, nb_out, cases[layer])
    in_data = torch.tensor([1.0, 0.5, 0.5, 1.0])
    out_data = net(in_data)
    print(out_data)

class DenseLayer():
    def __init__(self,nb_input,nb_output,case):
        super().__init__()
        self.linear = Linear(nb_input, nb_output, case)
        self.active = torch.sigmoid

    def __call__(self,data):
        return self.active(self.linear(data))

class Conv1D(nn.Module):
    def __init__(self,nb_input,nb_output,case,input_size):
        # input_size * kernel_size = nb_inputs
        super().__init__()
        self.mem_size = nb_input - input_size
        self.input_size = input_size
        self.mem = torch.zeros(self.mem_size)
        self.linear = Linear(nb_input, nb_output, case)
        self.active = torch.tanh

    def forward(self, data):
        assert data.numel() == self.input_size
        total = torch.cat([self.mem, data], dim=0)
        self.mem = total[self.input_size:].clone()
        return self.active(self.linear(total))

# ===================== 单元: conv1d =====================
# 对应 src/nnet.c: compute_generic_conv1d()
#   tmp = [mem, input] -> 线性层 -> 激活 -> 把 tmp 尾部存回 mem
# 注意 src/nnet.c 里用的是 tanh_approx() 多项式近似 + 快速倒数，
# 不是精确 tanh，所以两边只做数学等价对比，容差 1e-3。
#
# 同一份实现被 conv1 / conv2 两层复用，两层规模不同（见 CONV_LAYERS），
# 所以这里按层循环：每层各自读用例、各自跑、各自打 [<层名>.out#帧] / [<层名>.mem#帧]。

def conv1d_demo(cases, low_accuracy=False, out=sys.stdout):
    # 注意：conv1d 这条路径不受 --acc 影响（C 侧 conv1d 恒走 tanh，
    # 这里恒走 torch.tanh），low_accuracy 只是为了统一调用签名。
    # 每层各自读用例、各自跑（conv2 在用例里没有 inputs，那一层就跳过了）。
    for layer, in_size, nb_out in CONV_LAYERS:
        case = cases[layer]
        net = Conv1D(in_size * 3, nb_out, case, in_size)
        for frame, raw in enumerate(case["inputs"]):
            x = torch.tensor(raw, dtype=torch.float32)
            out_data = net(x)
            emit(layer, "out", frame, out_data[0], out)
            # forward 里 mem 已经更新成 tmp 的尾部，和 C 的 RNN_COPY 一致
            emit(layer, "mem", frame, net.mem, out)


def get_gru_params(input_features_size, hidden_size, output_features_size):
    
    W_xh = torch.randn((input_features_size, hidden_size), dtype=torch.float32)
    W_hh = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    b_h = torch.zeros(hidden_size, dtype=torch.float32)

    W_xz = torch.randn((input_features_size, hidden_size), dtype=torch.float32)
    W_hz = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    b_z = torch.zeros(hidden_size, dtype=torch.float32)
    
    W_xr = torch.randn((input_features_size, hidden_size), dtype=torch.float32)
    W_hr = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    b_r = torch.zeros(hidden_size, dtype=torch.float32)

    return W_xh, W_hh, b_h, W_xz, W_hz, b_z, W_xr, W_hr, b_r

def gru_demo(cases, low_accuracy=True, out=sys.stdout):
    '''对比 RNNoise 里 C 实现和 PyTorch 实现的输出。

    协议行与 examples/rnn_unit.c 的 local_compute_generic_gru() 同名同序，
    方便 scripts/unit_check.py 直接配对。
    low_accuracy=True  -> gru_scratch 里手写的多项式近似（对应 C 默认）
    low_accuracy=False -> torch 自带 sigmoid/tanh（对应 C 的 -DHIGH_ACCURACY）

    同一份实现被 gru1 / gru2 / gru3 三层复用，三层规模相同，所以这里按层循环，
    把层名作为协议行的前缀传给 GRUScratch。
    输入序列只有 gru1 那一段有，三层共用（C 侧 gru_unit() 同样如此）。
    '''
    input_features_size = GRU_INPUT_SIZE
    hidden_size = GRU_N
    output_features_size = GRU_OUTPUT_SIZE

    # GRUCell 官方实现 n=tanh(X@W_in​+b_in​+r⊙(W_hn@​H+b_hn​))
    # net = nn.GRUCell(input_features_size, hidden_size, bias=True, device='cpu', dtype=torch.float32)
    assert input_features_size == hidden_size
    assert input_features_size * 3 == output_features_size
    # 输入序列只有 gru1 那一段有（gru2 / gru3 在整网里由上一层的隐状态喂）。
    # 这个单元测的是"同一份输入 × 三套权重"，所以三层共用这一份 ——
    # 必须和 C 侧 gru_unit() 读的那一份一致。
    inputs_data = torch.tensor(cases["gru1"]["inputs"], dtype=torch.float32)

    for layer in GRU_LAYERS:
        case = cases[layer]
        if layer != GRU_LAYERS[0] and case["inputs"]:
            _fail("字段 %s.inputs 不该出现（输入序列统一用 %s 那份）" % (layer, GRU_LAYERS[0]))
        # 每层新建一个实例：隐状态从 0 开始（C 侧 gru_unit() 也是每层 memset 一次），
        # 不用用例 json 里的 state —— 两侧从同一起点递推，每次运行的起点也一样。
        net = GRUScratch(input_features_size, hidden_size, output_features_size,
                         case, layer=layer)
        # GRUScratch 一次只吃一帧，整段序列自己按帧循环（和 C 的 gru_unit() 同一个结构）
        for frame, x in enumerate(inputs_data):
            net(x, low_accuracy=low_accuracy, out=out, frame=frame)


import pdb
def rnnoise_demo(cases, low_accuracy=True, out=sys.stdout):

    # 初始化
    input_features = cases['conv1']['nb_in']
    nb_out = cases['conv1']['nb_out']

    conv1 = Conv1D(input_features * 3, nb_out, cases['conv1'], input_features)
    conv2 = Conv1D(nb_out * 3, nb_out*3, cases['conv2'], nb_out)

    # ['layer', 'input_weights_float', 'input_bias', 'recurrent_weights_float', 'recurrent_bias', 'state', 'inputs']
    gru1_case = cases['gru1']
    gru_input_size = hidden_size = nb_out*3
    gru_out_size = gru_input_size * 3

    gru1 = GRUScratch(gru_input_size, hidden_size, gru_out_size,
                    gru1_case, layer=gru1_case['layer'])
    
    gru2 = GRUScratch(gru_input_size, hidden_size, gru_out_size,
                    cases['gru2'], layer=cases['gru2']['layer'])
    
    gru3 = GRUScratch(gru_input_size, hidden_size, gru_out_size,
                    cases['gru3'], layer=cases['gru3']['layer'])

    # dense = DenseLayer(hidden_size*3 + nb_out, 4, cases['dense_out'])
    # vad_dense = DenseLayer(hidden_size*3 + nb_out, 1, cases['vad_dense'])

    inputs_data = torch.tensor(cases['conv1']['inputs'], dtype = torch.float32)

    assert inputs_data.shape[1] == 4

    for frame, data in enumerate(inputs_data):
        conv1_output = conv1(data)
        
        print(f'conv1 {frame}:{conv1_output}')
        
        conv2_output = conv2(conv1_output.flatten())
        print(f'conv2 {frame}:{conv2_output}')
        
        gru1_state = gru1(conv2_output.flatten(), low_accuracy=low_accuracy, out=out, frame=frame)
        print(f'gru1 {frame}:{gru1_state}')

        gru2_state = gru2(gru1_state, low_accuracy=low_accuracy, out=out, frame=frame)
        print(f'gru2 {frame}:{gru2_state}')

        gru3_state = gru3(gru2_state, low_accuracy=low_accuracy, out=out, frame=frame)
        print(f'gru3 {frame}:{gru3_state}')
        dense_in = torch.cat((conv2_output.flatten(), gru1_state.flatten(), gru2_state.flatten(), gru3_state.flatten()), dim=0)
        # dense
        # gains = dense(dense_in)
        # print(f'gains:{gains}')
        # vad
        # vad = vad(dense_in)
        # print(f'vad:{vad}')



UNITS = {
    "conv1d": conv1d_demo,
    "linear": linear_demo,
    "gru": gru_demo,
    "all": rnnoise_demo,
}

ACC_LOW = "low"
ACC_HIGH = "high"


def split_acc(argv):
    """把 `--acc high|low` 从 argv 里摘出来，返回 (剩下的 argv, low_accuracy)。

    缺省 low —— 和 C 侧默认（不定义 HIGH_ACCURACY）对齐。
    支持 `--acc high` 和 `--acc=high` 两种写法。
    """
    rest, acc = [], None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--acc":
            if i + 1 >= len(argv):
                raise SystemExit("--acc 后面要跟 high 或 low")
            acc = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--acc="):
            acc = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    if acc is None:
        acc = ACC_LOW
    if acc not in (ACC_LOW, ACC_HIGH):
        raise SystemExit("--acc 只能是 high 或 low，收到 %r" % acc)
    return rest, acc == ACC_LOW


USAGE = """\
用法: %(prog)s <unit> [case.json] [--acc high|low]

  unit        要跑的单元（缺省 conv1d）:
                conv1d  conv1 / conv2 两层各自独立跑，打 [convN.out#帧] / [convN.mem#帧]
                        （conv2 在用例里没有 inputs，那一层跑 0 帧）
                gru     gru1 / gru2 / gru3 三层，打 [gruN.zrh_recur|sigmoid|recur_tanh|h|state#帧]
                linear  单层线性层的手工实验（不算用例）
                注意：整条链路的 all 单元目前只有 C 侧有（examples/rnn_unit.c 的 conv_gru_unit）
  case.json   可选，显式指定用例文件；不写就按约定找
              <仓库根>/scripts/unit_cases/<用例文件名>.json（conv1d -> conv.json、gru -> gru.json）
  --acc       激活函数用高精度还是低精度（缺省 low，和 C 侧默认对齐）

  -h, --help  只看这份说明

例:
  %(prog)s conv1d
  %(prog)s gru --acc high
  %(prog)s conv1d scripts/unit_cases/conv.json

和 C 逐项对比: uv run scripts/unit_check.py [unit...]
"""


def usage(prog, bad_unit=None, out=sys.stderr):
    """打完整用法。bad_unit 非 None 表示是"单元名不认识"这一类错误。"""
    out.write(USAGE % {"prog": prog})
    if bad_unit is not None:
        out.write("\n[rnn_unit.py] 不认识的 unit: %s\n" % bad_unit)


def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动
    prog = os.path.basename(argv[0]) if argv else "rnn_unit.py"
    argv, low_accuracy = split_acc(argv)
    unit = argv[1] if len(argv) > 1 else "conv1d"

    if unit in ("-h", "--help", "help"):
        usage(prog, out=sys.stdout)
        return 0
    if unit not in UNITS:
        usage(prog, unit)
        return 2

    # argv[2] 只覆盖该单元「自己那份」用例（见上面的 UNIT_CASES）；
    # 其余几份固定走默认路径，所以 all 不用在命令行指定路径。
    # 用例找不到 / 字段不合格都在 loader 里报错退出（严格模式），这里不用再判断。
    paths, cases = load_unit_cases(unit, argv[2] if len(argv) > 2 else None)

    # 打一行用例来源，驱动器拿它核对两侧读的是不是同一份文件
    print("[rnn_unit.py] case file: %s" % " ".join(paths), file=sys.stderr)
    UNITS[unit](cases, low_accuracy=low_accuracy)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

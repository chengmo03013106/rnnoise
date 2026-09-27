#!/usr/bin/env python3
# coding=utf-8
"""练习 PyTorch：把 src/*.c 里的神经网络函数用 PyTorch 重新实现，逐项对比

输出协议（和 examples/rnn_unit.c 完全一致，驱动器靠它配对）:
    [<unit>.<output>#<frame>] <v0> <v1> ...

用例数据来源（和 C 侧完全一致），按顺序尝试:
    ① argv[2] 显式指定的 JSON
    ② 按约定自动定位 <仓库根>/scripts/unit_cases/<unit>.json
    ③ 都拿不到，才用本文件里的保底数据（此时会打醒目警告）

用法（本仓库用 uv 管理 python 环境）:
    uv run examples/rnn_unit.py conv1d                                   # 自动找用例文件
    uv run examples/rnn_unit.py conv1d scripts/unit_cases/conv1d.json    # 显式指定
    uv run scripts/unit_check.py                                         # 通常用这个，会自动跑双方并对比
"""

import json
import os
import sys

import torch
from torch import nn as nn
from gru_scratch import GRUScratch as GRUScratch

NB_INPUTS = 4
NB_OUTPUTS = 8
NB_FRAMES = 16          # 必须和 examples/rnn_unit.c 的 NB_FRAMES 一致

# gru1 用例规模（必须和 examples/rnn_unit.c 的 GRU1_* 一致）
GRU1_N = 4               # 隐层宽度 N
GRU1_INPUT_SIZE = 4      # 输入宽度
GRU1_OUTPUT_SIZE = 3 * GRU1_INPUT_SIZE # 三个门拼成的输出宽度 = 12

# ===================== 保底数据（JSON 读不到时用） =====================

DEFAULT_BIAS = [
    -0.09344794601202011, -0.009017355740070343, 0.016818566247820854, -0.008708192966878414,
    -0.020227156579494476, 0.03474648296833038, -0.0021663187071681023, 0.03218846395611763,
]

DEFAULT_INT8_WEIGHTS = [
    [-5, 52, 3, -12, -11, 1, 10, 15],
    [-9, 3, -22, 26, 0, -5, -8, -8],
    [22, -3, -4, -10, 18, -3, 8, -16],
    [-16, -11, -7, -6, 3, 8, 10, -9],
    [28, -10, 24, -9, -23, 23, -36, 2],
    [-22, 9, -4, 17, 4, -3, -3, -4],
    [4, -1, 0, 12, 8, 0, 13, 24],
    [14, -6, 1, 6, -5, 4, 14, 0],
    [8, 13, 21, -19, 0, 2, 6, -20],
    [-4, 2, -16, -2, 9, -3, 3, -3],
    [14, -2, -10, 2, 8, 2, -25, -12],
    [18, -6, 14, -12, -9, 18, 24, -2]
    ]

# 布局: weights[输入 j][输出 i]，12 行 8 列
DEFAULT_FLOAT_WEIGHTS = [
    -0.012715958058834076, -0.03491615876555443, -0.022936102002859116, 0.001433930709026754, 0.10274581611156464, 0.053419407457113266, -0.070134736597538, 0.010208060033619404,
    -0.06570827215909958, 0.024664076045155525, 0.06641422212123871, 0.008140825666487217, -0.03127734363079071, -0.023495245724916458, -0.0819985494017601, 0.06685595959424973,
    0.04988429695367813, 0.05374767631292343, -0.05255509540438652, -0.008573687635362148, -0.05985933169722557, 0.017076613381505013, 0.008259531110525131, 0.037359148263931274,
    0.029935840517282486, 0.012884164229035378, 0.04319038614630699, -0.0019242754206061363, 0.03693016618490219, 0.005330721382051706, 0.09885094314813614, -0.016847431659698486,
    0.019098864868283272, -0.02490164153277874, -0.048863593488931656, 0.05896652489900589, 0.057998109608888626, -0.07738705724477768, -0.0044519370421767235, 0.04088712856173515,
    -0.012941142544150352, 0.010212413966655731, -0.011736469343304634, -0.09905973076820374, -0.014685460366308689, -0.02975836955010891, 0.07942452281713486, 0.08345158398151398,
    0.06946410983800888, 0.03175446763634682, 0.0018908561905846, 0.015228806994855404, -0.050985466688871384, 0.020332524552941322, -0.0036082908045500517, -0.07804813981056213,
    0.0016910550184547901, -0.013379653915762901, -0.061623845249414444, -0.06263639777898788, -0.08425731211900711, -0.06852665543556213, 0.023757586255669594, 0.027971982955932617,
    0.05846858769655228, -0.07009495049715042, -0.039049528539180756, -0.02381090819835663, 0.01809931918978691, 0.02042475715279579, -0.006822121329605579, 0.01043251808732748,
    0.051606178283691406, 0.00228865840472281, 0.0444168746471405, 0.0217665396630764, 0.02478734590113163, -0.023502765223383904, -0.08154599368572235, -0.09295104444026947,
    0.02701922506093979, -0.07094322890043259, -0.01833420991897583, -0.03154221177101135, 0.010410266928374767, 0.08474843204021454, -0.08997549116611481, 0.00175130320712924,
    0.02234303392469883, 0.00011592444934649393, 0.034469764679670334, -0.0014623471070080996, 0.016707254573702812, -0.08452452719211578, 0.10813271999359131, -0.07161980867385864,
]

DEFAULT_INPUTS = [
    [1.0, 0.5, 0.5, 1.0],
    [0.1, 0.2, 0.3, 0.4],
    [1.0, 1.5, 2.0, 2.5],
]

# gru1 的保底数据：和 examples/rnn_unit_data.h 的 default_gru1_* 是同一份数
# 来源 = src/rnnoise_data_little.c 里 gru1_* 六个数组的开头部分
# （DEFAULT_GRU1_INPUTS 例外：不来自模型，和 scripts/unit_cases/gru.json 的 inputs 是同一份 16 帧）
DEFAULT_GRU1_INPUT_WEIGHTS = [
    0.07904426008462906, 0.02678517811000347, 0.05699218064546585, -0.1746419370174408, 0.3865419626235962, 0.1199631541967392, 0.008525285869836807, 0.029105132445693016, -0.09679863601922989, 0.06761759519577026, 0.10397452116012573, 0.1055983379483223,
    -0.01348793599754572, -0.1211949810385704, 0.1093112975358963, 0.09981008619070053, 0.0353853739798069, 0.056797366589307785, 0.10460647940635681, -0.008868950419127941, 0.14652040600776672, -0.06337755918502808, 0.059486184269189835, -0.11613085865974426,
    -0.061782874166965485, 0.06377187371253967, -0.15101595222949982, -0.02055302821099758, -0.16379320621490479, 0.1715063452720642, 0.07868355512619019, -0.09416601806879044, 0.08398067206144333, -0.051911573857069016, 0.07532878965139389, 0.15288271009922028,
    -0.1516423225402832, -0.22818821668624878, -0.034833770245313644, 0.1898442953824997, 0.056705866008996964, 0.19930946826934814, 0.025124449282884598, -0.08921916782855988, -0.03746344894170761, 0.01307145319879055, 0.18609170615673065, -0.30130329728126526,
]

DEFAULT_GRU1_RECURRENT_WEIGHTS = [
    -0.1876479834318161, 0.11865424364805222, 0.16511623561382294, -0.25146013498306274, 0.0, 0.12828399240970612, 0.17372958362102509, -0.16779263317584991, -0.27936843037605286, -0.19829748570919037, -0.16144010424613953, -0.5673248767852783,
    -0.4375206530094147, 0.0, -0.18594495952129364, 0.2272193878889084, -0.4134431779384613, -0.20871412754058838, 0.14896003901958466, -0.2618829309940338, 0.2602773606777191, -0.22024330496788025, 0.0, 0.031176192685961723,
    -0.048705849796533585, -0.15341725945472717, -0.1399914175271988, -0.189829483628273, 0.05491531640291214, 0.2806422710418701, -0.09963357448577881, 0.0, -0.17118427157402039, -0.23799601197242737, 0.06183604523539543, 0.11852103471755981,
    0.1237349882721901, 0.3897932767868042, -0.14755399525165558, 0.4718388319015503, 0.3266546428203583, 0.0035248221829533577, -0.07181638479232788, 0.14028045535087585, 0.06942874193191528, -0.004040791653096676, -0.08553256839513779, 0.05140835419297218,
]

DEFAULT_GRU1_INPUT_BIAS = [
    0.20178377628326416, -0.21118180453777313, 0.09730073064565659, 0.1449739634990692, -0.04300708696246147, -0.02484896034002304, -0.38094568252563477, 0.11978866904973984, -0.1404617428779602, -0.05047960206866264, -0.014604244381189346, 0.09809411317110062,
]

DEFAULT_GRU1_INPUT_SUBIAS = [
    1.0515909874811769, -0.5279526938684285, 0.4844651445746422, -0.9350611604750156, 1.2423750031739473, 0.695860955864191, -0.8364140805788338, -2.416544832289219, 3.35941727925092, 1.2262674309313297, 1.331235060468316, -1.3693625796586275,
]

DEFAULT_GRU1_RECURRENT_BIAS = [
    0.19540289044380188, -0.17025351524353027, 0.06968894600868225, 0.1243063285946846, -0.021274205297231674, -0.026521384716033936, -0.4351162910461426, 0.10080118477344513, -0.1393868625164032, -0.014325922355055809, 0.02797892317175865, 0.12538938224315643,
]

DEFAULT_GRU1_RECURRENT_SUBIAS = [
    -1.8486766191199422, 2.2931714062578976, 1.096578914206475, 2.2294223280623555, -0.7869434538297355, 0.22334761917591095, 0.584290498867631, -1.2497142092324793, 1.8282969454303384, -0.7578299511224031, -1.101554736495018, 1.0470996303483844,
]

DEFAULT_GRU1_INPUTS = [
    [-0.854183, -0.968372, -0.541929, 0.011214],
    [-0.630348, -0.927736, -0.2573, 0.536831],
    [0.707474, -0.742775, 0.4974, 0.405025],
    [0.439677, 0.430957, 0.223434, -0.597826],
    [0.203097, -0.09773, -0.673586, 0.746863],
    [-0.750505, -0.353896, -0.498026, -0.32775],
    [-0.153184, 0.767674, 0.798276, 0.897173],
    [-0.498611, 0.473152, 0.584856, -0.642764],
    [-0.246261, -0.032786, 0.541538, -0.650243],
    [0.367762, 0.09972, -0.200483, -0.686358],
    [0.448414, 0.5725, -0.837239, -0.266193],
    [-0.918937, 0.653609, 0.95898, -0.075197],
    [0.770334, -0.457745, 0.670512, -0.142],
    [0.732915, -0.921939, -0.781819, 0.860526],
    [0.499883, -0.764664, 0.158104, 0.182506],
    [-0.556971, 0.990831, 0.29182, -0.089462],
]

DEFAULT_GRU1_STATE = [
    0.812216, -0.026975, 0.984494, -0.857407,
]


def _is_num_list(value, n):
    """是不是长度为 n 的数字数组。"""
    if not isinstance(value, list) or len(value) != n:
        return False
    try:
        [float(x) for x in value]
    except (TypeError, ValueError):
        return False
    return True


def _is_frames(value, width):
    """是不是非空、帧数不超限、且每帧长度都对的多帧输入。"""
    return (isinstance(value, list) and 0 < len(value) <= NB_FRAMES
            and all(_is_num_list(f, width) for f in value))


def _read_json(path):
    """读用例 JSON；path 为空或文件不存在就返回 None。

    解析出错的细节单独打一行 stderr —— 下面那行「字段来源」要保持干净，
    驱动器会拿它和 C 的输出做字符串比对。
    """
    if not (path and os.path.isfile(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print("[rnn_unit.py] json error: %s" % exc, file=sys.stderr)
        return None


def _load_conv1d_case(path):
    """conv1d 用例：bias / float_weights / inputs。

    规则必须和 examples/rnn_unit.c 的 load_conv1d_case() 完全一致，
    否则两边会读到不同的数据，对比结果就是假的。
    """
    case = {
        "bias": list(DEFAULT_BIAS),
        "float_weights": list(DEFAULT_FLOAT_WEIGHTS),
        "inputs": [list(f) for f in DEFAULT_INPUTS],
    }
    src = {"bias": "builtin", "float_weights": "builtin", "inputs": "builtin"}

    data = _read_json(path)
    if isinstance(data, dict):
        if _is_num_list(data.get("bias"), NB_OUTPUTS):
            case["bias"] = data["bias"]
            src["bias"] = "json"
        if _is_num_list(data.get("float_weights"), NB_OUTPUTS * NB_INPUTS * 3):
            case["float_weights"] = data["float_weights"]
            src["float_weights"] = "json"
        if _is_frames(data.get("inputs"), NB_INPUTS):
            case["inputs"] = data["inputs"]
            src["inputs"] = "json"

    print("[rnn_unit.py] case data: bias=%s weights=%s inputs=%s%s" % (
        src["bias"], src["float_weights"], src["inputs"],
        (" path=" + path) if path else ""), file=sys.stderr)
    return case


def _load_gru_case(path):
    """gru1 用例：gru1_input_weights_float / gru1_input_bias / gru1_input_subias /
    gru1_recurrent_weights_float / gru1_recurrent_bias / gru1_recurrent_subias /
    inputs / state。

    规则必须和 examples/rnn_unit.c 的 load_gru_case() 完全一致。
    「字段来源」那行用 JSON 的键名、按同样的顺序打，驱动器靠它做字符串比对。
    """
    case = {
        "gru1_input_weights_float": list(DEFAULT_GRU1_INPUT_WEIGHTS),
        "gru1_input_bias": list(DEFAULT_GRU1_INPUT_BIAS),
        "gru1_input_subias": list(DEFAULT_GRU1_INPUT_SUBIAS),
        "gru1_recurrent_weights_float": list(DEFAULT_GRU1_RECURRENT_WEIGHTS),
        "gru1_recurrent_bias": list(DEFAULT_GRU1_RECURRENT_BIAS),
        "gru1_recurrent_subias": list(DEFAULT_GRU1_RECURRENT_SUBIAS),
        "inputs": [list(f) for f in DEFAULT_GRU1_INPUTS],
        "state": list(DEFAULT_GRU1_STATE),
    }
    src = {
        "gru1_input_weights_float": "builtin",
        "gru1_input_bias": "builtin",
        "gru1_input_subias": "builtin",
        "gru1_recurrent_weights_float": "builtin",
        "gru1_recurrent_bias": "builtin",
        "gru1_recurrent_subias": "builtin",
        "inputs": "builtin",
        "state": "builtin",
    }

    data = _read_json(path)
    if isinstance(data, dict):
        if _is_num_list(data.get("gru1_input_weights_float"), GRU1_INPUT_SIZE * GRU1_OUTPUT_SIZE):
            case["gru1_input_weights_float"] = data["gru1_input_weights_float"]
            src["gru1_input_weights_float"] = "json"
        if _is_num_list(data.get("gru1_input_bias"), GRU1_OUTPUT_SIZE):
            case["gru1_input_bias"] = data["gru1_input_bias"]
            src["gru1_input_bias"] = "json"
        if _is_num_list(data.get("gru1_input_subias"), GRU1_OUTPUT_SIZE):
            case["gru1_input_subias"] = data["gru1_input_subias"]
            src["gru1_input_subias"] = "json"
        if _is_num_list(data.get("gru1_recurrent_weights_float"), GRU1_N * GRU1_OUTPUT_SIZE):
            case["gru1_recurrent_weights_float"] = data["gru1_recurrent_weights_float"]
            src["gru1_recurrent_weights_float"] = "json"
        if _is_num_list(data.get("gru1_recurrent_bias"), GRU1_OUTPUT_SIZE):
            case["gru1_recurrent_bias"] = data["gru1_recurrent_bias"]
            src["gru1_recurrent_bias"] = "json"
        if _is_num_list(data.get("gru1_recurrent_subias"), GRU1_OUTPUT_SIZE):
            case["gru1_recurrent_subias"] = data["gru1_recurrent_subias"]
            src["gru1_recurrent_subias"] = "json"
        if _is_frames(data.get("inputs"), GRU1_INPUT_SIZE):
            case["inputs"] = data["inputs"]
            src["inputs"] = "json"
        if _is_num_list(data.get("state"), GRU1_N):
            case["state"] = data["state"]
            src["state"] = "json"

    print("[rnn_unit.py] case data:"
          " gru1_input_weights_float=%s gru1_input_bias=%s gru1_input_subias=%s"
          " gru1_recurrent_weights_float=%s gru1_recurrent_bias=%s gru1_recurrent_subias=%s"
          " inputs=%s state=%s%s" % (
              src["gru1_input_weights_float"], src["gru1_input_bias"], src["gru1_input_subias"],
              src["gru1_recurrent_weights_float"], src["gru1_recurrent_bias"],
              src["gru1_recurrent_subias"], src["inputs"], src["state"],
              (" path=" + path) if path else ""), file=sys.stderr)
    return case


def load_case(path, unit="conv1d"):
    """读用例 JSON；逐字段做长度校验，不合格的字段单独退回保底数据。

    path 为空 / 文件不存在时，所有字段都退回保底数据（两边都会打醒目警告）。
    每个单元读哪些字段、字段长度多少，必须和 examples/rnn_unit.c 的
    load_<unit>_case() 保持一致。
    """
    if unit == "gru":
        return _load_gru_case(path)
    return _load_conv1d_case(path)

def emit(unit, output, frame, values, out=sys.stdout):
    """按协议打印一项。格式化必须和 C 的 "%.8e" 一致。"""
    out.write("[%s.%s#%d] %s\n" % (
        unit, output, frame,
        " ".join("%.8e" % float(v) for v in values),
    ))


# ===================== 单元: conv1d =====================
# 对应 src/nnet.c: compute_generic_conv1d()
#   tmp = [mem, input] -> 线性层 -> 激活 -> 把 tmp 尾部存回 mem
# 注意 src/nnet.c 里用的是 tanh_approx() 多项式近似 + 快速倒数，
# 不是精确 tanh，所以两边只做数学等价对比，容差 1e-3。

def conv1d_unit(case, out=sys.stdout):
    # C 的权重布局是 weights[输入 j][输出 i]（sgemv 的 col_stride = nb_outputs）
    # PyTorch 的 nn.Linear 是 [输出, 输入]，所以这里直接写 x @ w，不做转置
    w = torch.tensor(case["float_weights"], dtype=torch.float32).reshape(NB_INPUTS * 3, NB_OUTPUTS)
    b = torch.tensor(case["bias"], dtype=torch.float32)

    mem = torch.zeros(NB_INPUTS * 2, dtype=torch.float32)
    for idx, raw in enumerate(case["inputs"]):
        x = torch.tensor(raw, dtype=torch.float32)
        tmp = torch.cat([mem, x])                      # C: tmp = [mem, input]
        y = torch.tanh(tmp @ w + b)    
        mem = tmp[NB_INPUTS:].clone()                  # C: RNN_COPY(mem, &tmp[input_size], ...)
        emit("conv1d", "out", idx, y, out)
        emit("conv1d", "mem", idx, mem, out)

# ===================== 待补单元 =====================
# conv1d 跑通后按同样的模式往下加：
#   - linear_demo     : compute_linear   (sgemv / cgemv8x4 / sparse_*)
#   - dense_demo      : compute_generic_dense
#   - gru_demo        : compute_generic_gru
#   - rnn             : recurrent neural networking

class Linear(nn.Module):
    def __init__(self,nb_input,nb_output,case):
        super().__init__()
        self.register_buffer(
            'weights',
            torch.tensor(case["float_weights"], 
            dtype=torch.float32).reshape(nb_input, nb_output)
        )
        # C 的权重布局是 weights[输入 j][输出 i]（sgemv 的 col_stride = nb_outputs）
        # PyTorch 的 nn.Linear 是 [输出, 输入]，所以这里直接写 x @ w，不做转置
        # 因为不需要训练，所以 requires_grad 不需要设置 True
        assert self.weights.shape[0] == nb_input
        self.register_buffer(
            'bias',
            torch.tensor(case["bias"], dtype=torch.float32)
        )
        # self.weights = self.weights.reshape(NB_INPUT,-1) # 保证 input 形状
        assert len(self.bias.shape) == 1 and self.bias.shape[0] == nb_output

    def forward(self, x):
        x = x.reshape(-1, self.weights.shape[0])
        y = x@self.weights + self.bias 
        return y

def linear_demo(case):
    net = Linear(NB_INPUTS, NB_OUTPUTS, case)
    in_data = torch.tensor([1.0,0.5,0.5,1.0])
    out_data = net(in_data)
    print(out_data)
    out_data = torch.tanh(out_data)

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
        self.mem = tmp[self.input_size:].clone()
        return self.active(self.linear(total))

def conv1d_demo(case, out=sys.stdout):
    # conv1d out:
    # -0.083926 0.008154 0.043958 -0.009412 0.073748 0.090026 -0.010317 0.077486
    # mem out:
    # 0.000000
    # 0.000000 0.000000 ...
    # 0.000000 %
    net = Conv1D(NB_INPUTS*3, NB_OUTPUTS, case, NB_INPUTS)

    for frame, raw in enumerate(case["inputs"]):
        x = torch.tensor(raw, dtype=torch.float32)
        out_data = net(x)
        emit("conv1d", "out", frame, out_data[0], out)
        emit("conv1d", "mem", frame, torch.cat([net.mem1,net.mem2]), out)
        # print("out1:",out_data)
        # print("mem1_1:",net.mem1)
        # print("mem1_2:",net.mem2)

import pdb
def default_case_path(unit):
    """按约定定位用例文件：<仓库根>/scripts/unit_cases/<unit>.json

    必须和 examples/rnn_unit.c 的 auto_case_path() 找同一个文件。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "scripts", "unit_cases", "%s.json" % unit)



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

def gru_demo(case, out=sys.stdout):
    '''对比RNNoise中的c实现和PyTorch实现输出的区别'''
    time_step = NB_FRAMES
    batch_size = 1
    input_features_size = GRU1_INPUT_SIZE
    hidden_size = GRU1_N
    output_features_size = GRU1_OUTPUT_SIZE
    
    # inputs_data = torch.randn((time_step, batch_size, input_features_size))
    
    # h = torch.tensor(case['state'],dtype=torch.float32)
    h = torch.zeros(hidden_size, dtype=torch.float32)
    assert h.shape[0] == hidden_size
    outputs = []

    # 💡 输出层参数，不属于 GRU
    W_hq = torch.randn((hidden_size, output_features_size), dtype=torch.float32)
    b_q = torch.zeros(output_features_size, dtype=torch.float32)
    
    # GRUCell 官方实现 n=tanh(X@W_in​+b_in​+r⊙(W_hn@​H+b_hn​))
    # net = nn.GRUCell(input_features_size, hidden_size, bias=True, device='cpu', dtype=torch.float32)
    assert input_features_size == hidden_size
    assert input_features_size*3 == output_features_size
    
    net = GRUScratch(input_features_size, hidden_size, output_features_size, case)
    net(torch.tensor(case["inputs"], dtype=torch.float32), h)
    
UNITS = {
    #"conv1d": conv1d_unit,
    "conv1d": conv1d_demo,
    "linear": linear_demo,
    "gru": gru_demo
}

def main(argv):
    torch.set_printoptions(sci_mode=True,precision=8)
    unit = argv[1] if len(argv) > 1 else "conv1d"
    if unit not in UNITS:
        print("unknown unit: %s" % unit, file=sys.stderr)
        return 2
    if len(argv) > 2:
        case_path = argv[2]                    # 显式指定
    else:
        auto = default_case_path(unit)         # 按约定自动定位
        case_path = auto if os.path.isfile(auto) else None
    UNITS[unit](load_case(case_path, unit))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

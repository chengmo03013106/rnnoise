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
  「单元 -> 读哪几份」写在 examples/loader.py 的 UNIT_CASES 里；整网 all 三份都要，路径全走默认。

本文件只留「单元的 runner + CLI」，其余的都拆出去只留一份：
    examples/linear.py  Linear / DenseLayer / Conv1D + conv1d 单元
    examples/gru.py     RNN / GRU 手撕练习（RNNScratch / do_gru）+ 整网用的 GRUMo
    examples/utils.py   输出协议 emit() + 从用例取张量的 case_tensor()
    examples/loader.py  用例加载（load_unit_cases / 规模常量 / CaseError）
    （另：examples/linear.py 单独跑起来是个「线性层手工实验」，它不是单元，见那边的 USAGE）

用法（本仓库用 uv 管理 python 环境）:
    uv run examples/rnn_unit.py conv1d                                # 自动找用例文件（= conv.json）
    uv run examples/rnn_unit.py all                                   # 整条链路，路径全默认
    uv run examples/rnn_unit.py conv1d scripts/unit_cases/conv.json    # 显式指定
    uv run scripts/unit_check.py                                       # 通常用这个，会自动跑双方并对比
"""

import os
import sys

import torch

# 显式重新导入这些名字，是为了保持既有接口不变（**不是**本地重复定义，定义体都在各自模块里）：
# scripts/err_stats/*.py 按 `import rnn_unit as U` 用 U.Conv1D / U.DenseLayer / U.Linear /
# U.GRUMo / U.load_unit_cases —— 以后拆模块时别把这些名字断掉。
# （tests/ 直接测 examples/gru.py，不再从这里取 get_gru_params。）
from gru import GRUMo as GRUMo
from linear import Conv1D, DenseLayer, Linear, conv1d_demo
from rnnoise_scratch import RNNoiseMo as RNNoiseMo
from utils import emit

from loader import (
    CaseError,          # main() 靠它区分「文件不存在」和「内容不合格」
    GRU_INPUT_SIZE,
    GRU_LAYERS,
    GRU_N,
    GRU_OUTPUT_SIZE,
    load_unit_cases,
)

# 对外契约：上面这些**再导出**的名字，`scripts/err_stats/*.py` 按 `import rnn_unit as U` 取用。
# 列进 `__all__` 一举两得：既是契约文档，也让静态检查知道
# `Linear` 这种"本地不用、只为再导出"的名字不是多余导入。
# （pyflakes **不认** `# noqa` —— 那是 flake8 的功能，加了不起作用。）
__all__ = [
    "Conv1D", "DenseLayer", "Linear", "GRUMo", "emit",
    "CaseError", "load_unit_cases",
    "GRU_LAYERS", "GRU_N", "GRU_INPUT_SIZE", "GRU_OUTPUT_SIZE",
]


def gru_demo(cases, low_accuracy=True, out=sys.stdout):
    '''对比 RNNoise 里 C 实现和 PyTorch 实现的输出。

    协议行与 examples/rnn_unit.c 的 local_compute_generic_gru() 同名同序，
    方便 scripts/unit_check.py 直接配对。
    low_accuracy=True  -> rnnoise_activation 里手写的多项式近似（对应 C 默认）
    low_accuracy=False -> torch 自带 sigmoid/tanh（对应 C 的 -DHIGH_ACCURACY）

    同一份实现被 gru1 / gru2 / gru3 三层复用，三层规模相同，所以这里按层循环；
    层名只是协议行的前缀，由本函数（runner）加 —— GRUMo 自己不打印。
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
            # 消息与 C 侧 gru_unit() 的那条校验保持一致（examples/rnn_unit.c:196）
            raise CaseError("[rnn_unit.py] json error: 字段 %s.inputs 不该出现"
                            "（输入序列统一用 %s 那份）" % (layer, GRU_LAYERS[0]))
        # 每层新建一个实例：隐状态从 0 开始（C 侧 gru_unit() 也是每层 memset 一次），
        # 不用用例 json 里的 state —— 两侧从同一起点递推，每次运行的起点也一样。
        # low_accuracy 是**构造函数**参数（GRUMo 在 __init__ 里快照激活函数）
        net = GRUMo(input_features_size, hidden_size, output_features_size, case,
                    low_accuracy=low_accuracy)
        # 隐状态由调用方持有（forward(x, hidden)）：每层各自一份、从 0 开始，
        # 逐帧把上一次返回的新隐状态接回去。
        hidden = torch.zeros(hidden_size, dtype=torch.float32)
        # GRUMo 一次只吃一帧，整段序列自己按帧循环（和 C 的 gru_unit() 同一个结构）。
        # 只打唯一能拿到的那一项：新隐状态（GRUMo 不再返回中间量 detail）。
        for frame, x in enumerate(inputs_data):
            hidden = net(x, hidden)
            emit(layer, "state", frame, hidden, out)


def rnnoise_demo(cases, low_accuracy=True, out=sys.stdout):
    """整条链路（all 单元）—— **内联实现**，留作迁移对照的基准。

    模型结构（7 层怎么搭）和逐帧数据流将来都归 examples/rnnoise_scratch.py 的
    RNNoiseMo（见本文件下面的 rnnoise_demo_2），这里先原样保留一份作为对照。

    注意：RNNoiseMo 的状态已改成显式进出，`all2` 不再返回 convN.out / dense.in，
    所以这里多打的 [conv1.out|conv2.out|dense.in#帧] **只有 all 有** ——
    两者只能在共有的 5 项（gruN.state / dense.gains / vad.out）上逐字节对比。
    要跟 C 的 `all` 整网对齐 8 项，就用这个内联版。
    """
    # 初始化
    input_features = cases['conv1']['nb_in']
    nb_out = cases['conv1']['nb_out']

    conv1 = Conv1D(input_features * 3, nb_out, cases['conv1'], input_features,
                   low_accuracy=low_accuracy)
    conv2 = Conv1D(nb_out * 3, nb_out*3, cases['conv2'], nb_out,
                   low_accuracy=low_accuracy)

    # mem 由调用方持有（Conv1D 不再自带状态）：长度 = nb_input - input_size = 2 帧宽。
    # C 侧也是从全 0 开始。
    conv1_mem = torch.zeros(input_features * 2, dtype=torch.float32)
    conv2_mem = torch.zeros(nb_out * 2, dtype=torch.float32)

    # ['layer', 'input_weights_float', 'input_bias', 'recurrent_weights_float', 'recurrent_bias', 'state', 'inputs']
    gru1_case = cases['gru1']
    gru_input_size = hidden_size = nb_out*3
    gru_out_size = gru_input_size * 3

    # low_accuracy 走构造函数（GRUMo 在 __init__ 里快照激活函数）
    gru1 = GRUMo(gru_input_size, hidden_size, gru_out_size, gru1_case,
                 low_accuracy=low_accuracy)

    gru2 = GRUMo(gru_input_size, hidden_size, gru_out_size, cases['gru2'],
                 low_accuracy=low_accuracy)

    gru3 = GRUMo(gru_input_size, hidden_size, gru_out_size, cases['gru3'],
                 low_accuracy=low_accuracy)

    dense = DenseLayer(hidden_size*3 + nb_out*3,
                       int((hidden_size*3 + nb_out*3)/(1536/32)), cases['dense_out'],
                       low_accuracy=low_accuracy)
    vad_dense = DenseLayer(hidden_size*3 + nb_out*3, 1, cases['vad_dense'],
                           low_accuracy=low_accuracy)

    inputs_data = torch.tensor(cases['conv1']['inputs'], dtype = torch.float32)

    assert inputs_data.shape[1] == 4

    # 三层 gru 的隐状态由调用方持有（forward(x, hidden)），各自从 0 开始、逐帧接回
    gru1_state = torch.zeros(hidden_size, dtype=torch.float32)
    gru2_state = torch.zeros(hidden_size, dtype=torch.float32)
    gru3_state = torch.zeros(hidden_size, dtype=torch.float32)

    for frame, data in enumerate(inputs_data):
        conv1_output, conv1_mem = conv1(data, conv1_mem)

        print(f'conv1 {frame}:{conv1_output}')
        emit('conv1', "out", frame, conv1_output[0], out)

        conv2_output, conv2_mem = conv2(conv1_output.flatten(), conv2_mem)
        emit('conv2', "out", frame, conv2_output[0], out)

        gru1_state = gru1(conv2_output.flatten(), gru1_state)
        # gruN.state 现在由这里打（GRUMo 不再自己打印）
        emit('gru1', "state", frame, gru1_state, out)

        gru2_state = gru2(gru1_state, gru2_state)
        emit('gru2', "state", frame, gru2_state, out)

        gru3_state = gru3(gru2_state, gru3_state)
        emit('gru3', "state", frame, gru3_state, out)
        dense_in = torch.cat((conv2_output.flatten(), gru1_state.flatten(), gru2_state.flatten(), gru3_state.flatten()), dim=0)
        # 拼给 dense / vad 的输入（对应 C 的 [dense.in#帧]）
        emit('dense', "in", frame, dense_in, out)

        # dense
        gains = dense(dense_in)
        emit('dense', "gains", frame, gains[0], out)

        # vad
        vad = vad_dense(dense_in)
        emit('vad', "out", frame, vad[0], out)


def rnnoise_demo_2(cases, low_accuracy=True, out=sys.stdout):
    """整条链路（all2 单元）—— 走 examples/rnnoise_scratch.py 的 RNNoiseMo。

    和 rnnoise_demo() 的区别：
      · 层怎么搭、每帧怎么串，都在类里；
      · 状态（两块 conv mem + 三层 GRU hidden）**打包成一个 `state` 显式进出**：
        `state = rnn.init_state()` 取初值，每帧 `state, (gains, vad) = rnn.forward(data, state)`；
        类里只留全 0 初值（register_buffer）。
      · 帧号 `frame` 留在**这里的循环**（`enumerate`），不进 `RNNoiseMo.forward` ——
        它是只用于日志的 int，`torch.onnx.export` 不能把它当 graph input。

    因此 all2 只打类**返回值里有的** 5 项：
        [gruN.state#帧] / [dense.gains#帧] / [vad.out#帧]
    （conv1.out / conv2.out / dense.in 只在**内联版 `all`** 里打 —— 那三个是调试量，
      RNNoiseMo 已不再返回，要跟 C 对齐就用 `all`；
      `all2` 与 `all` 在共有的 5 项上必须**逐字节相同**。）
    """
    # 模型不打印（RNNoiseMo / GRUMo 都不打）：协议行在这里按 forward 的返回值统一打，
    # 顺序与 C 的 conv_gru_unit() 一致。
    rnn = RNNoiseMo(cases, low_accuracy=low_accuracy)

    inputs_data = torch.tensor(cases['conv1']['inputs'], dtype=torch.float32)
    assert inputs_data.shape[1] == 4

    # 初值 = init_state()（buffer 的 clone）；之后每帧拿返回值覆盖，写法完全同形。
    state = rnn.init_state()

    for frame, data in enumerate(inputs_data):
        state, (gains, vad) = rnn.forward(data, state)
        # state 顺序 = (conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state)；
        # 本单元只打 gru 的三个状态 + 两个输出，两块 conv mem 用不上。
        _, _, gru1_state, gru2_state, gru3_state = state

        emit('gru1', "state", frame, gru1_state, out)
        emit('gru2', "state", frame, gru2_state, out)
        emit('gru3', "state", frame, gru3_state, out)
        emit('dense', "gains", frame, gains[0], out)
        emit('vad', "out", frame, vad[0], out)


# 只放**真单元**（有 C 侧对手、打协议行、unit_check.py 会跑）。
# linear_demo 那个线性层手工实验不在其列 —— 入口在 examples/linear.py。
UNITS = {
    "conv1d": conv1d_demo,
    "gru": gru_demo,
    "all": rnnoise_demo,        # 内联实现（对照基准）
    "all2": rnnoise_demo_2,     # 走 RNNoiseMo 类（迁移目标）
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
                gru     gru1 / gru2 / gru3 三层，只打 [gruN.state#帧]
                        （GRUMo.forward 只返回新隐状态、不再返回中间量 detail，
                          所以 zrh_recur/sigmoid/recur_tanh/h 这 4 项目前没有 py 来源，
                          unit_check 会把它们报成「仅 C 有，py 缺」）
                all     整条链路 conv1 / conv2 / gru1..3 / dense_out / vad_dense，
                        打 [convN.out|gruN.state|dense.in|dense.gains|vad.out#帧]
                        （内联实现，留作对照基准）
                all2    同 all，但走 examples/rnnoise_scratch.py 的 RNNoiseMo 类
                        状态（conv mem + 三层 hidden）显式进出，所以类只返回
                        [gruN.state|dense.gains|vad.out#帧] 5 项 —— 比 all 少
                        convN.out / dense.in（那三个是调试量）；
                        与 all 在共有的 5 项上输出逐字节相同
  （线性层的手工实验不是单元，不在 UNITS 里 —— 入口：uv run examples/linear.py）
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

    # argv[2] 只覆盖该单元「自己那份」用例（见 loader.py 里的 UNIT_CASES）；
    # 其余几份固定走默认路径，所以 all 不用在命令行指定路径。
    # 用例找不到 / 字段不合格都由 loader 抛 CaseError（严格模式）。
    try:
        paths, cases = load_unit_cases(unit, argv[2] if len(argv) > 2 else None)
    except CaseError as exc:
        # 只有"用例文件不存在"才补打完整用法 —— 加载器不知道本程序的用法长什么样。
        # 内容不合格时错误信息本身就是全部原因，打用法反而会把原因盖掉。
        if exc.missing:
            usage(prog)
        raise

    # 打一行用例来源，驱动器拿它核对两侧读的是不是同一份文件
    print("[rnn_unit.py] case file: %s" % " ".join(paths), file=sys.stderr)
    UNITS[unit](cases, low_accuracy=low_accuracy)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

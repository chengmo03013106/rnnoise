#!/usr/bin/env python3
# coding=utf-8
"""玩具模型：一个带 mem 的 conv1 + 一个 sigmoid 全连接 —— 练 ONNX 导出与读回。

本文件是从 onnx_demo.py / ort_demo.py 里抽出来的**自包含**版本：

    ToyModel      conv1（来自 examples/linear.py，自带 mem）+ sigmoid(state @ W + b)
    export_toy()  构造 → torch.onnx.export 导出到 toy.onnx
    run_toy()     用 onnx 读回 + checker 校验 + 打印可读结构   （来自 onnx_demo.py）
    load_toy()    同上，只是打印格式不同                      （来自 ort_demo.py）

用法（在仓库根目录、用 uv 启动）:
    uv run examples/toy.py                 # 借 scripts/unit_cases/conv.json
    uv run examples/toy.py <case.json>     # 显式指定用例

导出文件落在**当前工作目录**的 toy.onnx（和 ort_demo.py 的默认文件名一致，
两个脚本在同一目录下跑就能接上）。

★ 这里只做「导出 + 读回」；真正的 onnxruntime 推理在 examples/ort_demo.py: inference_demo()。
"""

import sys

import onnx
import torch
import torch.onnx
from torch import nn

from linear import Conv1D
from loader import load_unit_cases
from rnnoise_activation import pick


class ToyModel(nn.Module):
    """conv1（带 mem）+ sigmoid(state @ W + b)，一共两个输出。

    注意 conv1 来自 examples/linear.py，它自带 `self.mem`（滑动窗口）：
    **同一个实例连续跑两次结果会不同** —— 所以导出时 trace 到的那次会把当时的 mem
    当成常量冻进图里，参考输出和导出结果必须用两个"从零开始"的实例，否则对不上。
    """

    def __init__(self, nb_input, nb_output, case, input_size, low_accuracy=True):
        super().__init__()
        self.conv1 = Conv1D(nb_input, nb_output, case, input_size, low_accuracy)
        self.weights = nn.Parameter(torch.arange(nb_input*nb_output).reshape(nb_output,nb_input)*0.1)
        self.bias = nn.Parameter(torch.zeros(nb_input, dtype=torch.float32))
        self.sigmoid = pick(low_accuracy)[0]

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """
        Forward method that requires all inputs:
        - x: A direct tensor input.
        - state: 当前隐状态张量（对应图上名为 'tensor_state' 的输入）。
        """
        y = self.conv1(x.flatten())
        h_new = self.sigmoid(state@self.weights + self.bias)
        return h_new, y


def export_toy(cases):
    low_accuracy = True
    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    state = torch.arange(nb_input*nb_output, dtype=torch.float32)*0.01
    state = state.reshape(nb_input,nb_output)
    # Example inputs
    x = torch.arange(nb_input,dtype=torch.float32)*0.01
    x = x.reshape(1,nb_input)
    model = ToyModel(nb_input*3, nb_output, cases['conv1'], nb_input, low_accuracy)
    model.eval()

    h_new, y = model(x, state)
    print(f'test\ny={y}\nh_new={h_new}\n')

    # ★ conv1 带 mem：上面那次调用已经把 mem 改掉了。导出必须换一个**干净实例**，
    #   否则图里冻的是"脏"状态的输出，onnxruntime 的 y 会和上面的参考对不上
    #   —— 不报错，只是数值悄悄不对（已实测）。
    model = ToyModel(nb_input*3, nb_output, cases['conv1'], nb_input, low_accuracy)
    model.eval()

    # The input_names and output_names are used to identify the inputs and outputs of the ONNX model
    input_names = ['x', 'tensor_state']
    # ★ output_names 是**按位置**绑定 forward 返回值的：forward 是 `return h_new, y`，
    #   所以第 0 个必须叫 'h_new'、第 1 个才是 'y'。写反了不会报错，只会让
    #   ort_demo.py 里按名字取出来的 'y' 实际装着 h_new 的值。
    output_names = ['h_new', 'y']

    # Exporting the model with all required inputs
    torch.onnx.export(model, args=(x, state),
        f='toy.onnx', verbose=True,
        dynamic_axes={
                'x':            {0: "batch_size"},
                "tensor_state": {0: "batch_size"},
                "y": {0: "batch_size"},
                "h_new": {0: "batch_size"}
                },
        input_names=input_names,
        output_names=output_names)

    return 'toy.onnx'


def run_toy(model_file):
    model = onnx.load(model_file)

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def load_toy(model_file):
    model = onnx.load(model_file)

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(f'onnx.helper.printable_graph(model.graph)={onnx.helper.printable_graph(model.graph)}')
    print('\n\n')


def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动

    # 借 conv1d 单元那份用例（scripts/unit_cases/conv.json）；argv[1] 可显式指定
    _, cases = load_unit_cases('conv1d', argv[1] if len(argv) > 1 else None)

    model_file = export_toy(cases)   # 1) 构造 + 导出
    run_toy(model_file)              # 2) 用 onnx 读回来校验、打印结构
    load_toy(model_file)             # 3) 同上（两份只差打印格式，留着你挑）
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
# coding=utf-8
"""玩具模型：一个带 mem 的 conv1 + 一个 sigmoid 全连接 —— 练 ONNX 导出与读回。

本文件是从 onnx_demo.py / ort_demo.py 里抽出来的**自包含**版本：

    ToyModel      conv1（来自 examples/linear.py，自带 mem）+ sigmoid(state @ W + b)
    export_toy()  构造 → torch.onnx.export 导出到 toy.onnx
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
import torch.jit
from linear import Conv1D
from loader import load_unit_cases
from rnnoise_activation import pick

def one_epoch(net, batch_x, target, opti, loss, no_grad=True):
    for x in batch_x:
        opti.zero_grad()
        y_tmp = net(x)
        fn = net.weights.grad_fn
        if fn:
            print(f'fn:',fn)
            if fn.next_functions:
                fn = fn.next_functions
                print(f'fn:',fn)
        print(f'y_tmp {y_tmp.data}')
        l = loss(target,y_tmp)
        # print(f'loss {l}')
        if no_grad == False:
            l.backward()    # 自动计算 dl/dweight, dl/dbias
        opti.step()

def test_train():
    net = TestNet()
    #batch_x = torch.randn((10,2),dtype=torch.float32)
    batch_x = torch.tensor([[1,2],[1,2]],dtype=torch.float32)
    target = torch.tensor([2.4,2.6],dtype=torch.float32)
    opti = torch.optim.SGD(net.get_params(), 0.01)
    loss = nn.MSELoss()
    epochs_num = 2
    print(f' -------- train -------- ')
    net.train(True)
    for e in range(epochs_num):
        one_epoch(net, batch_x, target, opti, loss, no_grad=False)

    print(f' -------- eval -------- ')
    net.eval()
    for e in range(epochs_num):
        one_epoch(net, batch_x, target, opti, loss, no_grad=False)

    print(f' -------- no grad -------- ')
    for e in range(epochs_num):
        with torch.no_grad():
            one_epoch(net, batch_x, target, opti, loss, no_grad=True)   

def graph_test():
    x = torch.tensor(2.0, requires_grad=True)

    # 第一次反向传播
    loss = x ** 2
    loss.backward()
    print(x.grad)    # tensor(4.)  ← dL/dx = 2x = 4

    # 第二次反向传播（没有清零！）
    loss = x ** 2
    loss.backward()
    print(x.grad)    # tensor(8.)  ← 累积了！不是 4，而是 4+4=8

    # &#x2705; 正确做法：每次反向传播前先清零
    x.grad.zero_()   # 原地清零（注意下划线）

    loss = x ** 2
    loss.backward()
    print(x.grad)    # tensor(4.)  ← 正确

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn((2, 1),dtype=torch.float32, requires_grad=True)
        self.bias = torch.rand(2,dtype=torch.float32, requires_grad=True)
        # nn.Parameter是专门用来给模型提供参数， 使用Parameter会注册到模型的参数列表
        self.wp = nn.Parameter(torch.randn((1, 1),dtype=torch.float32, requires_grad=True))
        self.drop = nn.Dropout(p=0.5)

    def get_params(self):
        return self.weights, self.bias

    def forward(self, x):
        return self.drop(torch.tanh(x@self.weights + self.bias))


class ToyModel(nn.Module):
    """conv1（带 mem）+ sigmoid(state @ W + b)，一共两个输出。

    注意 conv1 来自 examples/linear.py：mem 由**调用方**持有
    （`out, mem = net(data, mem)`）。这里先放在 `self.conv1_mem` 里 ——
    所以同一个实例连续跑两次结果会不同，导出时 trace 到的那次会把当时的 mem
    当成常量冻进图里（`TracingCheckError` 的来源）。
    要让这张图真能被导出：把 mem 也做成 `forward` 的输入/输出。
    """

    def __init__(self, nb_input, nb_output, case, input_size, low_accuracy=True):
        super().__init__()
        self.conv1 = Conv1D(nb_input, nb_output, case, input_size, low_accuracy)
        # conv 的 mem 由调用方持有，长度 = nb_input - input_size（= (kernel-1) 帧）
        self.conv1_mem = torch.zeros(nb_input - input_size, dtype=torch.float32)
        self.weights = nn.Parameter(torch.arange(nb_input*nb_output).reshape(nb_output,nb_input)*0.1)
        self.bias = nn.Parameter(torch.zeros(nb_input, dtype=torch.float32))
        self.sigmoid = pick(low_accuracy)[0]

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """
        Forward method that requires all inputs:
        - x: A direct tensor input.
        - state: 当前隐状态张量（对应图上名为 'tensor_state' 的输入）。
        """
        y, self.conv1_mem = self.conv1(x.flatten(), self.conv1_mem)
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

    # h_new, y = model(x, state)
    # print(f'test\ny={y}\nh_new={h_new}\n')
    # return

    # jit.trace 
    trace_module = torch.jit.trace(
        model, example_inputs=(x, state), optimize=None, 
        check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, 
        _force_outplace=False, _module_class=None, _compilation_unit=None, 
        example_kwarg_inputs=None, _store_inputs=True)
    print(f'{trace_module}')
    return

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
    # load_toy(model_file)             # 3) 同上（两份只差打印格式，留着你挑）
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

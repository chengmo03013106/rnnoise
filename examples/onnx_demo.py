#!/usr/bin/env python3
# coding=utf-8
"""
练习：onnx export
"""

import sys
import torch
from torch import nn as nn
import onnx 
import torch.onnx
import torch.jit
import pdb

# 导入onnxruntime
import onnxruntime
import numpy as np
from rnnoise_scratch import RNNoiseMo
from linear import Conv1D,Linear,DenseLayer
from gru import GRUMo

from loader import (
    CONV_LAYERS,
    CaseError,          # main() 靠它区分「文件不存在」和「内容不合格」
    load_unit_cases,
)

def usage(prog, bad_unit=None, out=sys.stderr):
    """打完整用法。bad_unit 非 None 表示是"单元名不认识"这一类错误。"""
    if bad_unit is not None:
        out.write("\n[rnn_unit.py] 不认识的 unit: %s\n" % bad_unit)


save_dir = './examples/rnn.pth'
save_w_dir = './examples/rnn.w.pth'
checkpoint_path = 'examples/check.pth'
# 转换的onnx格式的名称，文件后缀需为.onnx
onnx_file_name = "./examples/rnn_demo.onnx"

def save_demo(m):
    # save model
    torch.save(m, save_dir) # 保存整个模型
    torch.save(m.state_dict, save_w_dir) # 保存模型权重

def load_demo():
    loaded_model = torch.load(save_dir).module
    loaded_w_model = torch.load(save_w_dir).module
    w = loaded_w_model.state_dict()
    return loaded_model

def check_model_demo(m):
    # 我们可以使用异常处理的方法进行检验
    try:
        # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model(m)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")

def save_param_demo(m, h_init, argv):
    # 保存参数
    torch.save({
        'model': m.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'lr_scheduler': lr_scheduler.state_dict(),
        'h_init': h_init,
        'argv': argv,
    }, checkpoint_path)    
    # 读取参数
    checkpoint = torch.load(checkpoint_path)
    if checkpoint and checkpoint['argv']:
        print(f'{checkpoint['argv']}')


def create_trace_conv1_onnx(cases, model_file):
    low_accuracy = True
    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    state = torch.arange(nb_input*nb_output, dtype=torch.float32)*0.01
    state = state.reshape(nb_input,nb_output)
    
    # Example inputs
    input_data = torch.tensor(cases['conv1']['inputs'], dtype=torch.float32)
    mem = torch.zeros(3*nb_input-nb_input)
    model = Conv1D(nb_input*3, nb_output, cases['conv1'], nb_input, low_accuracy)
    model.eval()

    # for x in input_data:
    #     y, mem = model(x, mem)
    #     print(f'{y}\n{mem}')
    # return

    # jit.trace 
    trace_module = torch.jit.trace(
        model, example_inputs=(x[0],mem), optimize=None, 
        check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, 
        _force_outplace=False, _module_class=None, _compilation_unit=None, 
        example_kwarg_inputs=None, _store_inputs=True)
    print(f'{trace_module.graph}')

    # onnx.checker.check_model(model)
    # The input_names and output_names are used to identify the inputs and outputs of the ONNX model
    input_names = ['x', 'mem']
    output_names = ['y', 'mem']
    torch.onnx.export(model, args=(x[0], mem),
        f=model_file, verbose=True,
        dynamic_axes={
                'x':            {0: "batch_size"},
                "mem": {0: "batch_size"},
                "y": {0: "batch_size"},
                "mem": {0: "batch_size"}
                },
        input_names=input_names,
        output_names=output_names)

    return

def inference_rnnoise(cases, model_file):
    x = np.arange(nb_input,dtype=np.float32)*0.01
    x = x.reshape(1,nb_input)

    state = np.arange(0,nb_input*nb_output,dtype=np.float32)*0.01
    state.shape = (nb_input,nb_output)

def inference_gru(cases, model_file):
    x = np.arange(nb_input,dtype=np.float32)*0.01
    x = x.reshape(1,nb_input)

    state = np.arange(0,nb_input*nb_output,dtype=np.float32)*0.01
    state.shape = (nb_input,nb_output)


def create_trace_gru_onnx(cases, model_file):
    low_accuracy = True
    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    state = torch.arange(nb_input*nb_output, dtype=torch.float32)*0.01
    state = state.reshape(nb_input,nb_output)
    
    # Example inputs
    hidden_size = nb_output * 3
    gru_out_size = hidden_size * 3 
    input_data = torch.tensor(cases['gru1']['inputs'], dtype=torch.float32)
    hidden = torch.zeros(hidden_size, dtype=torch.float32)
    # ★ GRUMo.forward 返回 (新隐状态, 中间量 dict)；而 torch.jit.trace **不接受 dict 输出**：
    #     "Encountering a dict at the output of the tracer ..."。
    #   导出只需要新隐状态，所以包一层"只取 h"的外壳给 trace / export 用。
    #   （另一种做法是把 detail 改成 NamedTuple、或挪出 forward —— 那是 gru.py 的设计选择。）
    class _HiddenOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, hidden):
            return self.inner(x, hidden)[0]

    model = _HiddenOnly(GRUMo(hidden_size, hidden_size, gru_out_size, cases['gru1']))
    model.eval()

    # for x in input_data:
    #     hidden = model(x, hidden)
    #     print(f'{hidden}')
    # return

    # jit.trace 
    trace_module = torch.jit.trace(
        model, example_inputs=(input_data[0],hidden), optimize=None, 
        check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, 
        _force_outplace=False, _module_class=None, _compilation_unit=None, 
        example_kwarg_inputs=None, _store_inputs=True)
    print(f'{trace_module.graph}')

    # onnx.checker.check_model(model)
    # The input_names and output_names are used to identify the inputs and outputs of the ONNX model
    # GRUMo.forward(x, hidden) 只返回一个新隐状态（没有单独的 y），所以 output_names 只给一项；
    # 名字按**位置**绑定返回值。
    # x / hidden 都是 1 维向量（[F] / [H]），第 0 轴是特征维而不是 batch ——
    # 没有真正可变的维，不要声明 dynamic_axes（声明了反而让 shape 推导把它们当同一维）。
    input_names = ['x', 'hidden']
    output_names = ['hidden']
    torch.onnx.export(model, args=(input_data[0], hidden),
        f=model_file, verbose=True,
        input_names=input_names,
        output_names=output_names)

    return

def inference_conv1(cases, model_file):
    # onnxruntime.InferenceSession用于获取一个 ONNX Runtime 推理器
    ort_session = onnxruntime.InferenceSession(model_file)  

    # 构建字典的输入数据，字典的key需要与我们构建onnx模型时的input_names相同

    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    rng = np.random.default_rng()
    # x = rng.random(nb_input, dtype=np.float32)
    input_data = np.array(cases['conv1']['inputs'], dtype=np.float32)

    # x = x.reshape(1,nb_input)
    mem = np.zeros(nb_input*2, dtype=np.float32)
    output_names = ['y','mem']
    np.set_printoptions(formatter={'float_kind': '{:e}'.format})

    for x in input_data:
        # 更建议使用这种方法,因为避免了手动输入key
        ort_inputs = {
            ort_session.get_inputs()[0].name : x, 
            ort_session.get_inputs()[1].name : mem
            }
        
        # run是进行模型的推理，第一个参数为输出张量名的列表，一般情况可以设置为None
        # 第二个参数为构建的输入值的字典
        # 由于返回的结果被列表嵌套，因此我们需要进行[0]的索引
        y, mem = ort_session.run(output_names, ort_inputs)
        # output = {ort_session.get_outputs()[0].name}
        # ort_output = ort_session.run([output], ort_inputs)[0]    
        print(f'y={y}\nmem={mem}\n')

def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动
    paths, cases = load_unit_cases('all2', argv[2] if len(argv) > 2 else None)
    # model_file = './examples/conv1.onnx'
    # create_trace_conv1_onnx(cases, model_file)
    # inference_conv1(cases, model_file)

    model_file = './examples/gru.onnx'
    create_trace_gru_onnx(cases, model_file)
    # inference_gru(cases, model_file)
    
    return 
    
    hidden_size = input_features_size = cases['conv1']['nb_in']
    net = RNNoiseMo(cases, low_accuracy=True)
    # 加载权重，将model.pth转换为自己的模型权重
    # 如果模型的权重是使用多卡训练出来，我们需要去除权重中多的module. 具体操作可以见5.4节
    # ❌ model = model.load_state_dict(torch.load(save_w_dir))

    # 导出模型前，必须调用model.eval()或者model.train(False)
    net.eval() 

    batch_size = 1 # 随机的取值，当设置dynamic_axes后影响不大
    # dummy_input就是一个输入的实例，仅提供输入shape、type等信息 
    dummy_input = torch.randn((batch_size, input_features_size),dtype=torch.float32) 
    hidden = torch.randn((batch_size, hidden_size), dtype=torch.float32)
    opset_version = onnx.defs.onnx_opset_version()

    # script module ， using torch.jit.trace() 
    # export your model with dynamic control flow, you will need to use scripting.
    # Use torch.jit.script() to produce a ScriptModule.

    # 导出模型
    # torch.onnx.export(model,        # 模型的名称
    #               dummy_input,   # 一组实例化输入
    #               onnx_file_name,   # 文件保存路径/名称
    #               export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
    #               opset_version=15, # ONNX 算子集的版本，当前已更新到15
    #               do_constant_folding=True,  # 是否执行常量折叠优化
    #               input_names = ['input','hidden'],   # 输入模型的张量的名称
    #               output_names = ['output'], # 输出模型的张量的名称
    #               # dynamic_axes将batch_size的维度指定为动态，
    #               # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
    #               dynamic_axes={'input' : {0 : 'batch_size'},    
    #                             'output' : {0 : 'batch_size'}}
    #                             )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
# coding=utf-8
"""
练习： onnx runtime 
"""

import os
import sys
import torch
from torch import nn as nn
import onnx 
import torch.onnx

from loader import (
    CaseError,          # main() 靠它区分「文件不存在」和「内容不合格」
    load_unit_cases,
)

import pdb

def usage(prog, bad_unit=None, out=sys.stderr):
    """打完整用法。bad_unit 非 None 表示是"单元名不认识"这一类错误。"""
    if bad_unit is not None:
        out.write("\n[rnn_unit.py] 不认识的 unit: %s\n" % bad_unit)


# 导入onnxruntime
import onnxruntime
import numpy as np
def inference_demo(model_file, cases):
    # onnxruntime.InferenceSession用于获取一个 ONNX Runtime 推理器
    ort_session = onnxruntime.InferenceSession(model_file)  

    # 构建字典的输入数据，字典的key需要与我们构建onnx模型时的input_names相同
    # 输入的input_img 也需要改变为ndarray格式
    # ort_inputs = {'input': input_img} 

    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    x = np.arange(nb_input,dtype=np.float32)*0.01
    x = x.reshape(1,nb_input)

    state = np.arange(0,nb_input*nb_output,dtype=np.float32)*0.01
    state.shape = (nb_input,nb_output)

    dict_input = {'tensor_state': state}

    # 没有用
    list_input = np.arange(0,nb_input, dtype=np.float32)
    list_input = np.tan(list_input)
    
    # 更建议使用这种方法,因为避免了手动输入key
    ort_inputs = {
        ort_session.get_inputs()[0].name : x, 
        ort_session.get_inputs()[1].name : state
        }
    output_names = ['y','h_new']

    # run是进行模型的推理，第一个参数为输出张量名的列表，一般情况可以设置为None
    # 第二个参数为构建的输入值的字典
    # 由于返回的结果被列表嵌套，因此我们需要进行[0]的索引
    y, h_new = ort_session.run(output_names, ort_inputs)
    # output = {ort_session.get_outputs()[0].name}
    # ort_output = ort_session.run([output], ort_inputs)[0]    

    print(f'y={y}\nh_new={h_new}')

def load_toy(model_file):
    model = onnx.load(model_file)

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(f'onnx.helper.printable_graph(model.graph)={onnx.helper.printable_graph(model.graph)}')
    print(f'\n\n')

def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动
    unit = 'conv1d'
    paths, cases = load_unit_cases(unit, argv[2] if len(argv) > 2 else None)
    model_file = 'toy.onnx'
    load_toy(model_file)
    inference_demo(model_file, cases)
    return 


    hidden_size = 4
    input_features_size = 4
    output_features_size = 8
    net = RNNoiseMo(input_features_size,hidden_size,output_features_size)
    # 加载权重，将model.pth转换为自己的模型权重
    # 如果模型的权重是使用多卡训练出来，我们需要去除权重中多的module. 具体操作可以见5.4节
    # ❌ model = model.load_state_dict(torch.load(save_w_dir))

    # 导出模型前，必须调用model.eval()或者model.train(False)
    # net.eval() 

    # dummy_input就是一个输入的实例，仅提供输入shape、type等信息 
    batch_size = 1 # 随机的取值，当设置dynamic_axes后影响不大
    dummy_input = torch.randn((batch_size, input_features_size), requires_grad=True) 
    hidden = torch.randn((batch_size, hidden_size), dtype=torch.float32)

    opset_version = onnx.defs.onnx_opset_version()

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

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
    paths, cases = load_unit_cases('all2', argv[2] if len(argv) > 2 else None)
    model_file = './examples/conv1.onnx'
    load_toy(model_file)
    inference_demo(model_file, cases)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

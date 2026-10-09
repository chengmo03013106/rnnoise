#!/usr/bin/env python3
# coding=utf-8
"""
rnnoise model trace->export->ORT
"""

import os
import sys
import torch
from torch import nn as nn
import onnx 
import torch.onnx
# 导入onnxruntime
import onnxruntime
import numpy as np

from rnnoise_scratch import RNNoiseMo
from loader import (
    CaseError,          # main() 靠它区分「文件不存在」和「内容不合格」
    load_unit_cases,
)

import pdb


def inference_demo(cases, ort_session, nb_input, nb_output):
    state = np.arange(0,nb_input*nb_output,dtype=np.float32)*0.01
    state.shape = (nb_input,nb_output)

    input_data = np.array(cases['conv1']['inputs'], dtype=np.float32)
    np.set_printoptions(formatter={'float_kind': '{:e}'.format})

    output_names = []
    for idx in range(len(ort_session.get_outputs())):
        output_names.append(ort_session.get_outputs()[idx].name)

    conv1_mem = np.zeros(nb_input*2, dtype=np.float32)
    conv2_mem = np.zeros(nb_output*2, dtype=np.float32)
    gru_state_size = nb_output*3
    gru1_state = np.zeros(gru_state_size, dtype=np.float32)
    gru2_state = np.zeros(gru_state_size, dtype=np.float32)
    gru3_state = np.zeros(gru_state_size, dtype=np.float32)
    input_names_list = ort_session.get_inputs()
    for frame, x in enumerate(input_data):
    # 构建字典的输入数据，字典的key需要与我们构建onnx模型时的input_names相同
        input_names = {
            input_names_list[0].name : x,
            input_names_list[1].name : conv1_mem,
            input_names_list[2].name : conv2_mem,
            input_names_list[3].name : gru1_state,
            input_names_list[4].name : gru2_state,
            input_names_list[5].name : gru3_state
        }

        conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state, gains, vad \
            = ort_session.run(output_names, input_names)
        
        print(f' -------- inference frame {frame} --------')
        print(f'conv1 mem:{conv1_mem}\nconv2 mem:{conv2_mem}')
        print(f'gru1 state:{gru1_state}\ngru2 state:{gru2_state}\ngru3 state:{gru3_state}')
        print(f'gains:{gains}\nvad:{vad}')

def create_trace_rnnoise_onnx(cases, model_file, nb_input, nb_output):
    low_accuracy = True

    model = RNNoiseMo(cases, low_accuracy)
    model.eval()

    # Example inputs
    input_data = torch.tensor(cases['conv1']['inputs'], dtype=torch.float32)

    # 初值 = init_state()（buffer 的 clone）；之后每帧都是同一个写法。
    state = model.init_state()

    for frame, x in enumerate(input_data):
        state, (gains, vad) = model(x, state)
        # ★ 必须从**返回的 state** 里取：`model.conv1_mem` 那些 buffer 永远是初值（全 0），
        #   拿它们打印只会每帧都打同一份零（旧的写法就是这个问题）。
        conv1_mem, conv2_mem, gru1_state, gru2_state, gru3_state = state
        print(f' -------- frame {frame} --------')
        print(f'conv1 mem:{conv1_mem}\nconv2 mem:{conv2_mem}')
        print(f'gru1 state:{gru1_state}\ngru2 state:{gru2_state}\ngru3 state:{gru3_state}')
        print(f'gains:{gains}\nvad:{vad}')
    return 

    # jit.trace 
    trace_module = torch.jit.trace(
        model, example_inputs=(input_data[0], state), optimize=None, 
        check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, 
        _force_outplace=False, _module_class=None, _compilation_unit=None, 
        example_kwarg_inputs=None, _store_inputs=True)
    print(f'{trace_module.graph}')

    # onnx.checker.check_model(model)

    # The input_names and output_names are used to identify the inputs and outputs of the ONNX model
    input_names = ['x', 'conv1_mem', 'conv2_mem', 'gru1_state', 'gru2_state', 'gru3_state']
    output_names = ['conv1_mem', 'conv2_mem', 'gru1_state', 'gru2_state', 'gru3_state', 'gains', 'vad']
    torch.onnx.export(model, args=(input_data[0], state),
        f=model_file, verbose=True,
        dynamic_axes={
                'x': {0: "batch_size"}
                },
        input_names=input_names,
        output_names=output_names)

    # 收集逐步的 torch 结果 (y, mem)，供与 ORT 侧 inference_conv1 做数值对比。
    # mem 由 forward 更新后传给下一步，和 ORT 侧喂 mem 的方式保持一致。
    # outs = []
    # for x in input_data:
    #     y, mem = model(x, mem)
    #     outs.append((y.detach(), mem.detach()))
    # return outs


def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动
    paths, cases = load_unit_cases('all2', argv[2] if len(argv) > 2 else None)
    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    model_file = './examples/rnnoise.onnx'
    create_trace_rnnoise_onnx(cases, model_file, nb_input, nb_output)

    # onnxruntime.InferenceSession用于获取一个 ONNX Runtime 推理器
    ort_session = onnxruntime.InferenceSession(model_file)  
    inference_demo(cases, ort_session, nb_input, nb_output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

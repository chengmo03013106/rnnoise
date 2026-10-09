#!/usr/bin/env python3
# coding=utf-8
"""
export test
conv1 model trace->export->ORT
gru model trace->export->ORT
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
from linear import Conv1D,Linear,DenseLayer
from gru import GRUMo

from loader import (
    CaseError,          # main() 靠它区分「文件不存在」和「内容不合格」
    load_unit_cases,
)

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


def create_trace_conv1_onnx(cases, model_file, nb_input, nb_output):
    low_accuracy = True

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
        model, example_inputs=(input_data[0],mem), optimize=None, 
        check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, 
        _force_outplace=False, _module_class=None, _compilation_unit=None, 
        example_kwarg_inputs=None, _store_inputs=True)
    print(f'{trace_module.graph}')

    # onnx.checker.check_model(model)
    # The input_names and output_names are used to identify the inputs and outputs of the ONNX model
    input_names = ['x', 'mem']
    output_names = ['y', 'mem']
    torch.onnx.export(model, args=(input_data[0], mem),
        f=model_file, verbose=True,
        dynamic_axes={
                'x':            {0: "batch_size"},
                "mem": {0: "batch_size"},
                "y": {0: "batch_size"},
                "mem": {0: "batch_size"}
                },
        input_names=input_names,
        output_names=output_names)

    # 收集逐步的 torch 结果 (y, mem)，供与 ORT 侧 inference_conv1 做数值对比。
    # mem 由 forward 更新后传给下一步，和 ORT 侧喂 mem 的方式保持一致。
    outs = []
    for x in input_data:
        y, mem = model(x, mem)
        outs.append((y.detach(), mem.detach()))
    return outs

def inference_gru(cases, ort_session, nb_input, nb_output):
    input_data = np.array(cases['gru1']['inputs'], dtype=np.float32)
    hidden_size = nb_output * 3

    hidden = np.zeros(hidden_size, dtype=np.float32)
    output_names = [ort_session.get_outputs()[0].name]

    # np.set_printoptions()
    np.set_printoptions(precision=8, suppress=False, formatter={'float_kind': '{:e}'.format})

    for x in input_data:
        # 更建议使用这种方法,因为避免了手动输入key
        ort_inputs = {
            ort_session.get_inputs()[0].name : x, 
            ort_session.get_inputs()[1].name : hidden
            }
        
        # run是进行模型的推理，第一个参数为输出张量名的列表，一般情况可以设置为None
        # 第二个参数为构建的输入值的字典
        # ⭐️ 由于返回的结果被列表嵌套，因此我们需要进行[0]的索引
        hidden = ort_session.run(output_names, ort_inputs)[0]
        print(f'h={hidden}\n')

def create_trace_gru_onnx(cases, model_file, nb_input, nb_output):
    low_accuracy = True
    state = torch.arange(nb_input*nb_output, dtype=torch.float32)*0.01
    state = state.reshape(nb_input,nb_output)
    
    # Example inputs
    hidden_size = nb_output * 3
    gru_out_size = hidden_size * 3 
    input_data = torch.tensor(cases['gru1']['inputs'], dtype=torch.float32)
    hidden = torch.zeros(hidden_size, dtype=torch.float32)
    
    model = GRUMo(hidden_size, hidden_size, gru_out_size, cases['gru1'])
    model.eval()
    print(f' -------- 模型执行 --------')
    for x in input_data:
        hidden = model(x, hidden)
        print(f'{hidden}')
    return

    # jit.trace 
    # trace_module = torch.jit.trace(
    #     model, example_inputs=(input_data[0],hidden), optimize=None, 
    #     check_trace=True, check_inputs=None, check_tolerance=1e-05, strict=True, 
    #     _force_outplace=False, _module_class=None, _compilation_unit=None, 
    #     example_kwarg_inputs=None, _store_inputs=True)
    # print(f'{trace_module.graph}')

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

def inference_conv1(cases, ort_session, nb_input, nb_output):
    input_data = np.array(cases['conv1']['inputs'], dtype=np.float32)
    mem = np.zeros(nb_input*2, dtype=np.float32)
    output_names = ['y','mem']
    np.set_printoptions(formatter={'float_kind': '{:e}'.format})
    outs = []

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
        outs.append((y, mem))
    return outs


# ---------------- 通用数值对比 ----------------
# 约定：create_trace_xxx 返回 torch.Tensor（或嵌套的 list/tuple）；
#       inference_xxx     返回 ndarray（或同结构）。
# 两边结构一致即可逐项对比；以后新增 xxx 对时，这里的代码不用改。

def to_numpy(v):
    """torch.Tensor / ndarray / 标量 -> np.ndarray。"""
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

def _flatten(v):
    """把嵌套的 list/tuple 摊平成一个个待比较的值。"""
    if isinstance(v, (list, tuple)):
        return [t for x in v for t in _flatten(x)]
    return [v]

def compare(name, torch_out, ort_out, atol=1e-05):
    """通用数值对比：create_trace_xxx 的结果 vs inference_xxx 的结果。"""
    a, b = _flatten(torch_out), _flatten(ort_out)
    if len(a) != len(b):
        print(f'[compare:{name}] 元素个数不一致: torch={len(a)} ort={len(b)}')
        return False
    ok = True
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = to_numpy(x), to_numpy(y)
        if x.shape != y.shape:
            print(f'[compare:{name}][{i}] shape 不一致: {x.shape} vs {y.shape}')
            ok = False
            continue
        diff = float(np.max(np.abs(x - y))) if x.size else 0.0
        same = np.allclose(x, y, atol=atol, rtol=0)
        ok = ok and same
        print(f'[compare:{name}][{i}] {"OK  " if same else "DIFF"} max|Δ|={diff:.3e} (atol={atol:.1e})')
    print(f'[compare:{name}] {"数值一致" if ok else "存在差异"}')
    return ok

def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动
    paths, cases = load_unit_cases('all2', argv[2] if len(argv) > 2 else None)
    # 构建字典的输入数据，字典的key需要与我们构建onnx模型时的input_names相同
    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']

    model_file = './examples/gru.onnx'
    create_trace_gru_onnx(cases, model_file, nb_input, nb_output)
    ort_session = onnxruntime.InferenceSession(model_file)  
    inference_gru(cases, ort_session, nb_input, nb_output)
    
    return 

if __name__ == "__main__":
    sys.exit(main(sys.argv))

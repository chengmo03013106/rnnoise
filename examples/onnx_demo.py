#!/usr/bin/env python3
# coding=utf-8
"""
练习：onnx, onnx runtime 
"""

import os
import sys
import torch
from torch import nn as nn
import onnx 
import torch.onnx

from rnnoise_scratch import RNNoiseMo
from rnnoise_activation import pick
from linear import Conv1D
from loader import (
    CaseError,          # main() 靠它区分「文件不存在」和「内容不合格」
    load_unit_cases,
)

import pdb

def usage(prog, bad_unit=None, out=sys.stderr):
    """打完整用法。bad_unit 非 None 表示是"单元名不认识"这一类错误。"""
    if bad_unit is not None:
        out.write("\n[rnn_unit.py] 不认识的 unit: %s\n" % bad_unit)


save_dir = './examples/rnn.pth'
save_w_dir = './examples/rnn.w.pth'
checkpoint_path = 'examples/check.pth'
# 转换的onnx格式的名称，文件后缀需为.onnx
onnx_file_name = "./examples/demo.onnx"

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
    

# 导入onnxruntime
import onnxruntime
def inference_demo():
    # onnxruntime.InferenceSession用于获取一个 ONNX Runtime 推理器
    ort_session = onnxruntime.InferenceSession(onnx_file_name)  

    # 构建字典的输入数据，字典的key需要与我们构建onnx模型时的input_names相同
    # 输入的input_img 也需要改变为ndarray格式
    ort_inputs = {'input': input_img} 
    # 我们更建议使用下面这种方法,因为避免了手动输入key
    # ort_inputs = {ort_session.get_inputs()[0].name:input_img}

    # run是进行模型的推理，第一个参数为输出张量名的列表，一般情况可以设置为None
    # 第二个参数为构建的输入值的字典
    # 由于返回的结果被列表嵌套，因此我们需要进行[0]的索引
    ort_output = ort_session.run(None,ort_inputs)[0]
    # output = {ort_session.get_outputs()[0].name}
    # ort_output = ort_session.run([output], ort_inputs)[0]    

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
    def __init__(self, nb_input, nb_output, case, input_size, low_accuracy=True):
        super().__init__()
        self.conv1 = Conv1D(nb_input, nb_output, case, input_size, low_accuracy)
        self.weights = nn.Parameter(torch.arange(nb_input*nb_output).reshape(nb_output,nb_input)*0.1)
        self.bias = nn.Parameter(torch.zeros(nb_input, dtype=torch.float32))
        self.sigmoid = pick(low_accuracy)[0]
        
    def forward(self, x: torch.Tensor, d: dict, l: list):
        """
        Forward method that requires all inputs:
        - x: A direct tensor input.
        - input_dict: A dictionary containing the tensor under the key 'tensor_state'.
        - input_list: A list where the first element is the tensor.
        """
        y = self.conv1(x.flatten())
        h = d['tensor_state']
        h_new = self.sigmoid(h@self.weights + self.bias)
        return y, h_new

def export_toy(cases):
    low_accuracy = True
    nb_input = cases['conv1']['nb_in']
    nb_output = cases['conv1']['nb_out']
    state = torch.randn((nb_input,nb_output))

    # Example inputs
    x = torch.arange(nb_input)*0.1
    x = x.reshape(1,nb_input)

    model = ToyModel(nb_input*3, nb_output, cases['conv1'], nb_input, low_accuracy)
    # state, y = model.forward(x,state)

    dict_input = {'tensor_state': state}
    list_input = [torch.rand((1,nb_input), dtype=torch.float32)]

    # The input_names and output_names are used to identify the inputs and outputs of the ONNX model
    input_names = ['x', 'tensor_state', 'list_input_index_0']
    output_names = ['y','h_new']

    # Exporting the model with all required inputs
    onnx_program = torch.onnx.export(model,args=(x, dict_input, list_input),
        f = 'toy.onnx', verbose=True,
        dynamic_axes={
                'x':            {0: "batch_size"},
                "tensor_state": {0: "batch_size"},
                "list_input_index_0":   {0: "batch_size"},
                "y": {0:"batch_size"},
                "h_new": {0: "batch_size"}
                },
        input_names=input_names, 
        output_names=output_names)    
    pdb.set_trace()
    print(f'{onnx_program.model.graph.inputs[0].shape}')
    print(f'{onnx_program.model.graph.inputs[1].shape}')
    print(f'{onnx_program.model.graph.inputs[2].shape}')

def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    torch.set_num_threads(1)  # 固定线程数，避免浮点累加顺序抖动
    unit = 'conv1d'
    paths, cases = load_unit_cases(unit, argv[2] if len(argv) > 2 else None)
    export_toy(cases)
    # test_train()
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

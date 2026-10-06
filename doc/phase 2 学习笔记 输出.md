对于图像分类任务，可以使用sklearn.metrics中的classification_report函数来计算模型的准确率、召回率、F1值等指标

使用 torcheval 或 torchmetric 来对模型进行评估。

### 1 train() 与 eval() 区别

不包含 batch normalization，Dropout 等特殊层时，两种方式的训练结果几乎相同 ❌
包含 batch normalization，Dropout  等特殊层时，两种训练方式结果有差异  ❌

不能说训练结果，要说 对 forward 计算没有影响。
训练结果包括 
是否计算梯度
backward
optimizer.step()

eval() 底层调用 train(False)

我的问题：
> eval(), train() 观察结果
我的代码，前向传播实现如下
```python
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn((2, 1),dtype=torch.float32,requires_grad=True)
        self.bias = torch.rand(2,dtype=torch.float32,requires_grad=True)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.drop(torch.tanh(x@self.weights + self.bias))

net = TestNet()
y = net(x)

```

我观察到的内容： 开启train模式后，有张量值是 0，有些一个元素是0，有些两个元素是0，这也就是文档中说的，dropout 的概率，每一批次，每一 *channel* ❌ 都是相对独立。eval 没有出现0，是因为开启后修改了模型行为，忽略了dropout 层，也符合官方文档描述

✅ 对输入 tensor 的元素进行随机 mask，每个元素以概率 p 被置零。
⭐️ **Dropout train mode 不只是随机置零，还会对未被 mask 的元素进行 inverted scaling，使训练阶段输出的期望保持一致。** 不会减少结果。

⭐️ weights, bias 没有在 net.parameters() ， 因为requires_grad=True 和 Parameter 不是一回事！！！
requires_grad=True： autograd 会计算
Parameter： (named_)parameters(), state_dict() 中会得到
```
 -------- train -------- 
y_tmp tensor([-1.87405467e+00, -0.00000000e+00])
y_tmp tensor([-1.85786414e+00, -0.00000000e+00])
y_tmp tensor([-0., -0.])
y_tmp tensor([-1.83735991e+00, -1.91832781e+00])
y_tmp tensor([-0.00000000e+00, -1.89883900e+00])
 -------- eval -------- 
y_tmp tensor([7.90366173e-01, 5.78452528e-01])
y_tmp tensor([8.26157689e-01, 6.47538245e-01])
y_tmp tensor([8.51773262e-01, 6.98341846e-01])
y_tmp tensor([8.70898664e-01, 7.36944973e-01])
```

官方文档中 `Gradients for non-differentiable functions` 部分没有看明白，请用中文解释
答： 如果一个函数在某个点没有正常意义上的导数，Autograd 到底怎么办？

### 2 model.eval()  no_grad() 分别负责什么
eval() : 对 dropout 层 会做特殊处理，但是参数依然会进行求导
no_grad() 接下来不会对参数做自动求导

测试观察：
y_tmp 在 `torch.no_grad()` 开启后， **`requires_grad=False` ， **`grad_fn` 这个属性是 None，因为没有构建 backward graph**

通过设置 requires_grad=False， 开启推理 mode，使用 no-grad mode 都可以禁用自动梯度计算 

回答你的问题

> 为什么 `model.eval()` 不能代替 `torch.no_grad()`？
答：

❌ ❌ ❌ ❌ eval 是改变 dropout batchnormalation 层的操作，从而节约显存和处理器消耗

eval() 的主要作用是**切换 Module 的 evaluation behavior**, 而不是降低资源占用，它不是通用的性能优化开关。

no grad 是告诉系统不计算梯度，在每次 forward 过程中，不会把计算结果保存到 .grad_fn , .grad_fn.next_functions 中。两个功能就不一样。
正常 forward - Autograd 记录计算历史 - 后续可以 backward 
调试过程中观察的方式是 tensor.grad_fn, tensor.grad_fn.next_functions 吗？
答： 现在可以看，辅助定位，但不是真正的profile方法

> 为什么 inference 通常同时使用 `eval()` 和 `no_grad()`？
答：这两个功能互补干扰，❌ 目标都是为了降低资源占用，从不同的角度提高效率 ❌

✅ 因为它们解决的是两个不同的问题：eval() 保证模型采用 inference/evaluation behavior；no_grad() 避免为不需要训练的 forward 构建 backward graph。

> 一个完全没有 Dropout/BatchNorm 的 RNNoise 模型，`eval()` 是否会明显改变当前计算结果？
答：不会

> 为什么部署时通常更关心模型参数，而不是 Python class 本身？
因为模型的训练其实就是参数的调整，工程中更多的是在已有参数的基础之上对新模型进行 fine-tuning，所以参数更重要。
model architecture + parameters = runnable model
state_dict() 保存 register buffer + parameters

### 总结 
no grad(), requires_grad=True/False: 计算图

train() / eval(): 改变模型行为，和计算图无关, 避免某些不确定行为，比如 dropout 是随机mask
使用 eval() 直接pass 随机行为，得到确定性结果

requires_grad=True 是哪些变量需要？
训练的过程就是调整模型参数 weights, bias 让预测结果更接近 target。
输入数据 x.requires_grad       = False   
模型参数 W.requires_grad       = True
模型参数 b.requires_grad       = True
标签 y.requires_grad       = False

x 虽然 requires_grad=False ， 但仍然被计算！！！
⭐️ 如果一个 operation 的输入 tensor 中至少有一个需要 gradient，该 operation 就会被记录到 backward graph。

- 打印 y_tmp.grad_fn， y_tmp.grad_fn.next_functions 两个变量
🙋 y_tmp 是通过 tanh + 矩阵乘法 + 加法得到

输入1：tanh 输出
输入2：dropout mask / scale
dropout 层执行的实际就是 mulBackward0, 这是从最后一层往前的

```text
fn: <MulBackward0 object at 0x11ba7c5e0>
fn: ((<TanhBackward0 object at 0x11ba7c670>, 0), (None, 0)) 

真实计算图可能是这样
MulBackward0
│
├── TanhBackward0
│      │
│      ↓
│   AddBackward0
│      ├── MmBackward0
│      │      ├── x
│      │      └── weights
│      │
│      └── bias
│
└── None

```

⭐️ x.grad.zero_() 清零 和 optim.grad_zero() 的区别
x.grad.zero_() 只清 x， 另外 zero_grad() **并不一定真的把内存里的 grad 写成 `0`，可能写成`None`**
optimizer.zero_grad() 清理初始化时的全部管理的参数


### ONNX

#### 继承 **nn.Module**： 
- 执行model(xx)时，可以让 *forward()* 被调用
- 内部维护 *_parameters*, *_buffers*, *_modules，* 并且 **Parameter**, **Module** 时登记如表， 这会导致连带的后果（不只是 ONNX）：state_dict() / load_state_dict() 丢参数、.to(device/dtype) 不迁移、.eval() 传不下去、.parameters() 为空（optimizer 收不到）、torch.onnx.export 遍历参数转 initializer 时拿不到 → 导出的图带不出权重。

#### register_buffer 作用
- 进 state_dict()
- .to() 可用
- 不被 optimizer 吸收更新

单独实现一个变量，不在 state_dict() ， 不能被迁移GPU，与 weights，bias 等分离。

ONNX 是静态计算图，没有"对象属性"这个概念。**torch.onnx.export 的机制** 是跑一遍你的 forward，把「**张量→张量的运算**」记成节点；self.h = ... 这种 Python 赋值它不记。

forward() 执行直接带所有需要的参数，使用 类变量则没有意义。 注意！！！ 中间变量需要传，权重参数不需要，输入数据每次输入。

状态 / 缓存（跨调用延续）例如代码中的 **KV cache、RNN hidden、conv mem**	在 graph input 和 graph output（成对）

简单过了一次规范，使用 protobuffer 语法描述 ✅

然后直接做 `PyTorch RNNoise` 导出 `rnnoise.onnx` -> 再导入 -> 在用 `netron` 看

export 约定

模块继承 nn.Module
模型参数 Parameter 实现

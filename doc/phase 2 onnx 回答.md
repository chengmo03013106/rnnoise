你的设备约束一确定，路线就应该重新收敛：

> **以 PyTorch 2.2.2 的 TorchScript-based ONNX Exporter 为主。**
>
> 你现在看到的 `torch.export`、FX、Export IR Spec，并不是当前 RNNoise→ONNX 的第一学习主线。

这一点我先明确下来。PyTorch 2.2 的 ONNX 文档明确写的是：`torch.onnx.export()` 这条路线基于 TorchScript；传入普通 `nn.Module` 时，exporter 会先做相当于 `torch.jit.trace()` 的转换，再把 TorchScript graph 转成 ONNX。只有你显式传入 `ScriptModule` 时，才走 scripting 路线。([PyTorch Docs][1])

而 `torch.export` 在 2.2 文档里还是 prototype，而且是另一套 graph capture 机制。它内部使用 TorchDynamo + AOT Autograd，并以 FX 作为底层图表示。([PyTorch Docs][2])

所以你现在最需要做的事情，是把这几条路线彻底分开。

---

# 一、先把路线图纠正过来

你之前写：

```text
torch.export
    ↓
onnx.export
    ↓
runtime
```

对于你当前 PyTorch 2.2.2 环境，**不要这么理解。**

你现在应该理解为：

```text
                    PyTorch 2.2.2

PyTorch nn.Module
       │
       │
       ▼
torch.onnx.export()
       │
       │
       ├── torch.jit.trace()
       │       或
       │   torch.jit.script()
       │
       ▼
TorchScript Graph
       │
       │ ONNX symbolic translation
       ▼
ONNX Graph / Model
       │
       ▼
ONNX Runtime
```

这是你现在真正要学习的路径。PyTorch 2.2 官方把这条路线明确称为 **TorchScript-based ONNX Exporter**。([PyTorch Docs][1])

而另一条：

```text
PyTorch Model
      ↓
torch.export
      ↓
ExportedProgram / FX-based IR
```

你现在只需要知道：

> **这是另外一套新式 graph capture 技术，目前不是你 RNNoise 导出到 ONNX 的主线。**

---

# 二、所以先回答你列出的几个文档：哪些学，哪些暂时不要学？

## 1. `torch.export`

### 现在：只了解，不深入

你需要知道它解决的问题：

> 把 Python/PyTorch 程序捕获成一个纯 Tensor computation graph。

它强调：

```text
AOT
Graph Capture
Graph
Operator
Functional
No Python semantics
```

这些概念对你以后学 AI Runtime 很重要。

但：

```text
torch.export API
ExportedProgram
dynamic_shapes
range_constraints
equality_constraints
```

**现在先不要深入。**

原因很简单：

> 你当前 `torch.onnx.export()` 的实际执行路线根本不是靠它。

PyTorch 2.2 自己的文档也把 `torch.export` 描述为 prototype，并明确说明它和 `torch.fx`、TorchDynamo、AOT Autograd组成另一套 PT2 技术栈。([PyTorch Docs][2])

---

# 三、`torch.export IR Spec`：现在不要系统学习

你发现：

> 它要求读者熟悉 FX。

这个判断是对的。

因为 `torch.export` 的 IR 里面本来就有大量：

```text
ExportedProgram
Graph
Node
ATen operator
Graph Signature
Range Constraint
```

而且其底层 graph representation 就使用 FX。([PyTorch Docs][2])

### 现在你的学习等级：

```text
知道：
torch.export 有自己的 IR
↓
知道：
它底层用 FX Graph
↓
停止
```

不用现在去啃 IR Spec。

---

# 四、TorchScript：这个现在要学，而且是当前主线

这个和 FX 不一样。

**TorchScript 对你当前 PyTorch 2.2 ONNX export 是直接相关的。**

官方文档明确说：

```text
torch.onnx.export()
      ↓
如果输入不是 ScriptModule
      ↓
等价于 torch.jit.trace()
      ↓
得到 TorchScript graph
      ↓
export
```

如果你自己先 `torch.jit.script()`：

```text
model
↓
ScriptModule
↓
torch.onnx.export()
```

则 exporter 不再重新 trace。([PyTorch Docs][1])

所以：

# TorchScript：要学，但只学 Export 所需的那部分。

不用把整个 TorchScript 学完。

---

# 五、你需要学 TorchScript 的哪些东西？

只学这 4 个概念。

## ① `torch.jit.trace`

必须搞懂：

```python
traced = torch.jit.trace(model, example_inputs)
```

到底发生什么。

概念：

```text
example input
      ↓
执行一次 forward
      ↓
记录过程中发生的 Tensor operations
      ↓
得到 TorchScript graph
```

PyTorch 2.2 的 ONNX 文档就明确这么描述：tracing 会执行模型一次并记录这次执行中的操作。([PyTorch Docs][1])

---

## ② Tracing 的核心限制

这是你现在最重要的知识之一。

假设：

```python
def forward(self, x):
    if x.sum() > 0:
        return x + 1
    else:
        return x - 1
```

你 trace 的输入恰好满足：

```text
x.sum() > 0
```

那么 trace 记录的是：

```text
x + 1
```

不会自动把：

```text
if
else
```

完整表达进去。

PyTorch 2.2 文档明确说，trace 会把 if/loop 按这一次实际执行的路径展开，得到与这次 traced run 一致的静态图。要保留动态 control flow，应使用 scripting。([PyTorch Docs][1])

这个一定要理解。

---

## ③ `torch.jit.script`

只需要知道：

> 如果 forward 中存在必须保留的动态 Python control flow，可以尝试 scripting。

例如：

```text
if
for
while
```

而 scripting 会分析代码，而不是只记录一次执行路径。PyTorch 2.2 文档明确把 tracing 和 scripting 区分为两种方式。([PyTorch Docs][1])

---

## ④ `.graph`

这个非常值得你现在学。

例如：

```python
traced_model.graph
```

你会看到类似：

```text
graph(%x, %weight, %bias):
    %1 = aten::matmul(...)
    %2 = aten::add(...)
    %3 = aten::tanh(...)
    return (%3)
```

这对你非常重要，因为它正好连接：

```text
C代码
↓
PyTorch
↓
TorchScript Graph
↓
ONNX Graph
```

你以后再学 vLLM、TensorRT，会不断遇到这种：

> 高级模型代码 → 中间表示 → lower/operator → runtime

---

# 六、FX：现在到底要不要学？

我的建议：

# **现在不要学。**

不是说 FX 没价值。

而是它不是你当前技术链中的必要前置。

你现在：

```text
RNNoise
↓
PyTorch
↓
torch.onnx.export
↓
TorchScript
↓
ONNX
```

这里真正直接相关的是：

```text
TorchScript
```

而不是：

```text
torch.fx
```

FX 在 PyTorch 2.2 中是一个用于 symbolic tracing、graph transformation 的体系。它的典型用途是：

```text
nn.Module
↓
FX Graph
↓
修改 graph
↓
GraphModule
```

官方 FX 文档就是这样定义它的。([PyTorch Docs][3])

以后你做：

```text
graph optimization
operator fusion
quantization
compiler
```

再系统学习 FX 非常合理。

现在：

> **知道 FX 是 PyTorch 的 Graph IR / transformation 工具即可。**

---

# 七、你问的 ONNX TorchScript 文档，要不要学？

# 要，而且这是你目前最应该学习的文档。

就是你给的：

> `onnx_torchscript.html`

它不是“参考资料”。

**它现在就是你的主教材。**

尤其看这几部分：

```text
Example: AlexNet from PyTorch to ONNX
        ↓
Tracing vs Scripting
        ↓
Avoiding Pitfalls
        ↓
Limitations / Types
        ↓
Operator support
        ↓
dynamic_axes
        ↓
verification
```

官方文档甚至直接给了：

```text
PyTorch
↓
torch.onnx.export
↓
ONNX
↓
onnx.checker
↓
ONNX Runtime
```

的完整链路。([PyTorch Docs][1])

---

# 八、所以你现在的学习优先级是

我给你明确排一下：

| 内容                       | 现在            |
| ------------------------ | ------------- |
| `torch.onnx.export` 2.2  | **必须**        |
| TorchScript              | **必须学核心部分**   |
| `torch.jit.trace`        | **必须**        |
| `torch.jit.script`       | **需要理解，不必深入** |
| TorchScript `.graph`     | **必须**        |
| ONNX Graph/Node/Operator | **必须**        |
| ONNX IR Spec             | 只学用到的部分       |
| FX                       | **暂时不学**      |
| torch.export             | **只了解概念**     |
| torch.export IR Spec     | **暂时不学**      |
| TorchDynamo              | 暂时不学          |
| AOT Autograd             | 暂时不学          |

这个范围非常重要。

因为你现在最容易重新陷入：

> “既然 export 涉及 FX，那我是不是必须先把 FX 学完？”

**不是。**

---

# 九、现在回答你问的几个概念

这是你接下来最关键的部分。

---

# 1. Export 导出到底有几个阶段？

你之前列：

```text
model validation?
torch.export?
ONNX translation?
shape constraint?
unsupported operator?
```

这个分类**混合了新旧两套 exporter 概念**。

对于你的 PyTorch 2.2 TorchScript 路线，我建议这样理解：

```text
① Model / input contract
        ↓
② Trace 或 Script
        ↓
③ TorchScript Graph
        ↓
④ ONNX symbolic translation
        ↓
⑤ ONNX Model
        ↓
⑥ ONNX checker / Runtime verification
```

这是你现在真正需要的阶段划分。

---

# 十、阶段 ①：Model / Input Contract

首先 exporter 要知道：

```text
model 是什么？
forward 接受什么？
输入是什么？
输出是什么？
```

PyTorch 2.2 的 `torch.onnx.export()` 接受：

```text
nn.Module
ScriptModule
ScriptFunction
```

而 `args` 必须能够真正调用：

```python
model(*args)
```

官方文档明确规定，Tensor 类型参数会成为 ONNX 输入；非 Tensor 参数可能被硬编码进导出的模型。([PyTorch Docs][1])

所以这里的问题就是：

> **你的 `forward` 签名和 export 提供的 `args` 是否一致？**

你之前 RNNoise 的：

```text
forward(frame, data)
```

但 export：

```text
dummy_input
```

这就是这里的问题。

---

# 十一、阶段 ②：Trace / Script

普通：

```python
torch.onnx.export(model, args, ...)
```

会：

```text
nn.Module
↓
torch.jit.trace
↓
TorchScript graph
```

官方文档明确这么描述。([PyTorch Docs][1])

如果：

```text
trace
```

失败：

可能是：

```text
Python code
state mutation
unsupported types
control flow
unsupported operations
```

---

# 十二、阶段 ③：TorchScript Graph

这时候你已经得到：

```text
Graph
```

例如：

```text
input
 ↓
MatMul
 ↓
Add
 ↓
Tanh
 ↓
output
```

你的下一个问题：

> 这个 Graph 是否准确表达了原来的 PyTorch forward？

这就是为什么官方建议 examine exported model / graph。([PyTorch Docs][1])

---

# 十三、阶段 ④：ONNX Translation

这时候开始：

```text
TorchScript operator
        ↓
ONNX operator
```

比如：

```text
aten::matmul
      ↓
onnx::MatMul
```

如果 exporter 没有某个 operator 的 symbolic conversion：

```text
Couldn't export operator foo
```

这就是：

# Unsupported Operator

PyTorch 2.2 官方文档明确把这归为 ONNX operator conversion 问题，并提供修改模型、写 custom symbolic function 或增加 exporter 支持等方案。([PyTorch Docs][1])

---

# 十四、你提到的 `shape constraint`，现在先放掉

这是非常重要的。

你把：

```text
shape constraint
```

放在 exporter 的阶段列表里，是因为看到了：

```text
torch.export
```

相关资料。

**但对于你当前 TorchScript ONNX exporter，不需要把它作为一个独立的核心阶段学习。**

PyTorch 2.2 legacy exporter 更重要的是：

```text
example input shape
↓
trace 记录
↓
如果需要动态维度
↓
dynamic_axes
```

官方 FAQ 直接说明：

> tracer 会记录 example inputs 的 shape；如果希望 ONNX 模型接受动态 shape，则使用 `dynamic_axes`。([PyTorch Docs][1])

所以现在你记：

```text
static shape
dynamic_axes
```

就够了。

以后进入 `torch.export`，再学：

```text
range constraints
equality constraints
symbolic shapes
```

---

# 十五、`unsupported operator` 到底发生在什么时候？

非常清楚：

```text
TorchScript graph
      ↓
ONNX translation
      ↓
某个 operator 无法转换
      ↓
unsupported operator
```

例如：

```text
aten::foo
```

但是：

```text
torch.onnx
```

没有：

```text
foo → ONNX xxx
```

于是失败。

PyTorch 2.2 还提供：

```python
torch.onnx.utils.unconvertible_ops(...)
```

帮助一次找出多个不能转换的 ATen operators。([PyTorch Docs][1])

这个以后你遇到 RNNoise 非标准 operation 时会非常有用。

---

# 十六、`state cannot be traced`：谁的 state？

这个问题非常好。

你现在首先要区分三种 state。

## ① 模型参数

```text
weights
bias
```

这是：

> trained parameters

它们当然可以进入导出的模型。

---

## ② Buffer

比如 BatchNorm：

```text
running_mean
running_var
```

也是 Module state，但不是通过 optimizer 学习的参数。

---

## ③ Runtime state

这才是你 RNNoise 真正麻烦的：

```text
Conv1 mem
Conv2 mem
GRU1 hidden
GRU2 hidden
GRU3 hidden
```

它们描述：

> **模型在连续时间推理中的瞬时状态。**

例如：

```text
state_t
   +
frame_t
   ↓
RNNoise
   ↓
output_t
   +
state_(t+1)
```

这就是 recurrent state。

---

# 十七、为什么 Python object 里的 runtime state 对 export 麻烦？

你当前：

```text
self.gru1.state
self.gru2.state
self.gru3.state
```

是对象内部偷偷变化的。

也就是说：

第一次：

```text
net(frame1)
```

之后：

```text
self.gru1.state = state1
```

第二次：

```text
net(frame2)
```

实际上读取了：

```text
state1
```

再生成：

```text
state2
```

但：

```text
state
```

不是 `forward` 的显式输入/输出。

从 Python 程序角度：

> 没问题。

从静态计算图角度：

> **不清楚。**

因为 graph 更容易表达的是：

```text
(frame_t, state_t)
           ↓
         graph
           ↓
(output_t, state_t+1)
```

而不是：

```text
frame_t
 ↓
修改 self.xxx
 ↓
output
```

所以你 RNNoise 后面真正要解决的不是：

> “为什么 ONNX API 报错？”

而是：

> **如何把实时模型的隐式 runtime state 表达成显式的计算图 state。**

这是你目前学习中最有价值的一个问题。

---

# 十八、什么是 Stateful Model Representation？

中文：

> **有状态的模型表示。**

最简单：

### 无状态

```text
y = f(x)
```

每次：

```text
same x
→ same y
```

---

### 有状态

```text
(y, s_new) = f(x, s_old)
```

下一次：

```text
(y2, s2) = f(x2, s_new)
```

RNNoise 本质就是：

```text
(frame_t, state_t)
       ↓
     network
       ↓
(output_t, state_t+1)
```

你的 C 代码用：

```text
struct/state object
```

保存 `state_t`。

而对于 ONNX 图，更自然的表示是：

```text
input:
frame
state

output:
output
new_state
```

你现在**应该先把这个概念彻底搞懂。**

暂时不要急着实现。

---

# 十九、什么是 `pure tensor computation`？

非常简单：

> **计算结果主要由 Tensor 输入、参数和 Tensor operators 决定，而不是依赖 Python 解释器在运行期间做额外的外部动作。**

例如：

```python
def forward(x):
    y = x @ weight
    y = torch.tanh(y)
    return y
```

这非常纯：

```text
Tensor
↓
MatMul
↓
Tanh
↓
Tensor
```

---

# 二十、什么不是 pure Tensor computation？

例如：

```python
def forward(x):
    print(x)
    global_counter += 1
    self.state = x
    with open("xxx", "w") as f:
        f.write(...)
    return x + 1
```

这些：

```text
print
文件 IO
global variable 修改
Python object state mutation
```

都不是模型 Tensor computation 本身。

注意：

> **这不是说这些代码在普通 Python 中不能运行。**

而是：

> **它们不是适合直接表达成 ONNX graph 的计算。**

---

# 二十一、什么是 Python side effect？

就是：

> **执行 forward 时除了计算返回值以外，还修改了外部可观察状态。**

例如：

```python
self.counter += 1
```

就是 side effect。

```python
self.state = new_state
```

也是。

```python
print(...)
```

虽然不会改变模型数学结果，但它是外部 I/O side effect。

```python
list.append(...)
```

修改 Python 对象，也可能成为 graph capture 的障碍。

你当前 RNNoise 的：

```text
print(...)
emit(...)
self.gru1.state = ...
```

就属于非常典型的 export-sensitive code。

---

# 二十二、什么是 Graph Capture？

先不要把这个词想复杂。

你当前 TorchScript tracing 的 graph capture 就是：

```text
Python forward
       ↓
给一个 example input
       ↓
实际执行 forward
       ↓
观察发生了哪些 Tensor operations
       ↓
记录成 Graph
```

所以：

```python
torch.jit.trace(model, example)
```

本质上就是：

> **运行一次，把这次 Tensor computation 记录下来。**

PyTorch 2.2 ONNX 文档明确这样描述 tracing。([PyTorch Docs][1])

---

# 二十三、所以什么叫 `graph-compatible forward`？

这不是 PyTorch 2.2 官方一个严格的 API 类型名称，而是我们为了理解问题使用的概念。

你可以把它理解成：

> **一个 `forward()`，它的计算能够被当前 exporter 的 tracing/scripting 机制捕获，并且能转换成 ONNX 所能表达的 operator/dataflow。**

比如：

```python
def forward(self, x):
    return torch.tanh(x @ self.weight + self.bias)
```

非常 graph-compatible。

而：

```python
def forward(self, x):
    print(x)
    if x.sum() > 0:
        self.state = x
        return ...
```

就会碰到：

```text
Python side effect
+
data-dependent Python control flow
+
mutable state
```

这些都是 tracing/export 的风险点。

官方 2.2 文档明确提醒：Python/NumPy 值在 tracing 时可能被当成常量；数据依赖的动态 control flow 也不会被普通 tracing 正确保留。([PyTorch Docs][1])

---

# 二十四、现在回头看你的 RNNoise，就非常清楚了

你现在的模型：

```text
id="5p1dgu"
RNNoiseMo
├── conv1
│    └── self.mem
├── conv2
│    └── self.mem
├── gru1
│    └── self.state
├── gru2
│    └── self.state
├── gru3
│    └── self.state
└── forward()
      ├── print
      ├── emit
      └── mutate state
```

这就是为什么你一上来直接：

```text
torch.onnx.export()
```

然后：

```text
Google
↓
修错
↓
再报错
```

会非常痛苦。

因为你实际上还没有解决：

> **这个 Python 程序怎样变成一个静态计算图？**

这不是 ONNX 文件格式的问题。

---

# 二十五、所以你当前真正的学习目标，就是这一张图

```text
Python Model
     │
     │
     ▼
Is it a nn.Module?
     │
     ▼
What are its Inputs?
What are its Outputs?
     │
     ▼
Are states explicit?
     │
     ▼
Is forward graph-compatible?
     │
     ▼
Can TorchScript trace/script it?
     │
     ▼
TorchScript Graph
     │
     ▼
Can every op map to ONNX?
     │
     ▼
ONNX Graph
     │
     ▼
Can Runtime execute it?
```

**这就是你现在真正要学的 Export Contract。**

---

# 二十六、给你一个“当前阶段的 Export Contract”

你可以直接把它作为自己的学习笔记。

## PyTorch 2.2 Legacy ONNX Export Contract

### 1. Model

应该能够作为：

```text
nn.Module
```

或者：

```text
ScriptModule
ScriptFunction
```

被 exporter 接受。([PyTorch Docs][1])

---

### 2. Forward interface

必须能够：

```python
model(*example_args)
```

正常执行。

输入数量、类型、shape 要和 export args 对得上。

---

### 3. Input / Output types

当前 legacy exporter 对 Tensor、数字类型以及由这些组成的 tuple/list 有明确支持限制；dict/string 等存在 tracing 下的特殊行为和限制。([PyTorch Docs][1])

所以：

> **你的 model input/output 最好先严格保持 Tensor/tuple of Tensor。**

---

### 4. Computation

forward 中的核心计算必须能够被：

```text
TorchScript trace/script
```

捕获。

---

### 5. Dynamic control flow

如果使用 tracing：

```text
if
for
while
```

依赖运行时数据时，不能指望 trace 自动保留完整动态行为。需要 scripting 等机制。([PyTorch Docs][1])

---

### 6. Python values

不要依赖：

```text
numpy
Tensor.item()
Python list/dict/string
```

参与需要动态变化的 Tensor computation。

因为 legacy tracer 可能把它们固化成 constant。([PyTorch Docs][1])

---

### 7. State

模型运行时需要持续变化的 state：

```text
RNN hidden
Conv memory
```

必须有明确的表示方式。

**不能简单依赖 Python object 在 forward 之外偷偷保存状态。**

这是你 RNNoise 当前最关键的问题。

---

### 8. Operators

最终 TorchScript graph 中的 operators 必须能够转换到 ONNX。

否则：

```text
unsupported operator
```

([PyTorch Docs][1])

---

### 9. Shape

legacy exporter：

```text
example input
↓
trace
↓
默认记录这些 shape
```

需要动态维度时，使用：

```text
dynamic_axes
```

这是你现在要学习的 shape 机制。([PyTorch Docs][1])

---

# 二十七、所以你现在**不应该**继续研究的东西

我给你明确踩刹车：

### 暂时不要：

```text
FX 深入
torch.export 深入
Export IR Spec 深入
TorchDynamo
AOT Autograd
symbolic shapes
range constraints
equality constraints
```

这些都不是你当前 PyTorch 2.2 legacy exporter 的第一前置知识。

---

# 二十八、你现在应该这样学习

我建议你接下来不要直接碰 RNNoise。

按下面 4 个实验走。

---

## 实验 1：最简单的 Trace

```text
x
 ↓
Linear
 ↓
Tanh
 ↓
y
```

目标：

```text
PyTorch
↓
torch.jit.trace
↓
print(graph)
```

你要回答：

> Trace 做了什么？

> Graph 里面是什么？

> forward 的每一步在哪里？

---

## 实验 2：故意制造 data-dependent control flow

```text
if x.sum() > 0:
    ...
else:
    ...
```

然后 trace。

观察：

> 为什么另一条分支没有进入 graph？

这一步让你真正理解：

> **Tracing = 记录一次 execution，不是理解整个 Python 程序。**

---

## 实验 3：明确 state

做：

```text
(x, h)
   ↓
MyRNNCell
   ↓
(y, h_new)
```

然后 trace/export。

这里你要观察：

> 为什么显式 state 比 `self.state` 更容易形成 graph？

这一步对你 RNNoise 极其重要。

---

## 实验 4：ONNX Translation

拿实验 3：

```text
TorchScript graph
↓
ONNX
```

然后：

```text
onnx.checker
↓
print graph
↓
ORT run
```

直到：

```text
PyTorch
≈
TorchScript
≈
ONNX
≈
ORT
```

然后再回 RNNoise。

---

# 二十九、这样你再回头看 RNNoise，就不会再是“缝补”

你会按这个流程诊断：

```text
RNNoise export fails
        ↓
是哪一层？

Model contract?
        ↓
Input/output?
        ↓
Trace?
        ↓
State?
        ↓
Python side effect?
        ↓
Unsupported operator?
        ↓
Shape?
        ↓
ONNX translation?
```

每个错误都进入一个**已经学过的类别**。

而不是：

```text
Error A
↓
Google
↓
Patch

Error B
↓
Google
↓
Patch

Error C
↓
Google
↓
Patch
```

这正是你说的“不体系”的根源。

---

# 三十、你现在最应该读哪一个文档？

我给你明确排序：

## 第一优先级

**PyTorch 2.2 `onnx_torchscript.html`**

重点读：

```text
Example: AlexNet
Tracing vs Scripting
Avoiding Pitfalls
Types
Differences in Operator Implementations
Adding support for operators
FAQ
torch.onnx.export API
```

尤其是官方已经明确提供的：

```text
model → trace → ONNX → checker → ORT
```

完整路径。([PyTorch Docs][1])

---

## 第二优先级

**TorchScript 2.2**

只读：

```text
trace
script
trace vs script
graph
```

特别是你给的 `mixing tracing and scripting`，暂时知道它存在即可。官方文档说明两者可以组合，但你当前 RNNoise 不需要马上进入这个复杂度。([PyTorch Docs][4])

---

## 第三优先级

**ONNX IR**

只读：

```text
Graph
Node
Value
Initializer
Input
Output
Operator
Opset
```

不需要继续读整个 spec。

---

## 第四优先级

FX

**暂时跳过。**

只保留一个认知：

> FX 是 PyTorch 的 graph representation / transformation framework。

---

## 第五优先级

`torch.export`

**暂时跳过深入。**

只知道：

> 它是另一套更现代的 graph capture 技术，在 2.2 还是 prototype，并且底层用了 TorchDynamo/AOT Autograd/FX。([PyTorch Docs][2])

---

# 三十一、最后提醒你：你现在没有跑偏

你现在问：

> 什么是 trace？

> trace 为什么需要 example input？

> state 是谁的 state？

> 为什么 Python object state 不能直接等价于 ONNX state？

> 什么是 graph capture？

> 什么叫 graph-compatible forward？

这些问题**全部在当前 Phase 2 主线上。**

你唯一差点跑偏的是：

> **开始跟着 `torch.export → FX → IR Spec` 这套新路线深入。**

对于你的 PyTorch 2.2.2 环境，**现在停在这里，不要继续往下追。**

---

# 三十二、我们现在正式把一个非常重要的学习目标固定下来

## 当前唯一目标

> **搞懂 PyTorch 2.2.2 的 `torch.onnx.export()` 到底要求什么样的 Model/Forward，为什么要求这些，以及它如何从 Python/PyTorch 得到 TorchScript Graph，再转换成 ONNX。**

暂时不要修改 RNNoise。

先把：

```text
nn.Module
   ↓
forward contract
   ↓
trace
   ↓
TorchScript graph
   ↓
ONNX translation
```

这个过程用一个**最小 toy model**跑通并解释清楚。

等你能够回答下面 10 个问题，才回 RNNoise：

1. `torch.onnx.export()` 在 PyTorch 2.2 默认是什么路线？
2. `nn.Module` 经过 export 后为什么会先出现 TorchScript？
3. `torch.jit.trace()` 到底“trace”了什么？
4. 为什么 trace 需要 example input？
5. trace 为什么无法可靠表示 data-dependent `if`？
6. `torch.jit.script()` 与 trace 的根本区别是什么？
7. 什么是 graph-compatible forward？
8. 什么是 Python side effect，为什么可能影响 export？
9. RNNoise 的 `gru.state` 为什么和普通 `weight` 不一样？
10. 为什么 `(frame, state) → (output, new_state)` 比“对象内部偷偷修改 state”更适合表达为静态 graph？

**这 10 个问题就是你当前 Phase 2 的真正验收标准。**

等这 10 个问题吃透，回到你那份 `RNNoiseMo`，你之前遇到的那些“一个错误接一个错误”的现象，就会从十几个孤立的错误变成几个明确的类别。这样才真正实现你一直要求的：**学习不是“看过”，而是知道为什么、知道怎么验证、知道出了问题应该往哪里查。**

[1]: https://docs.pytorch.org/docs/2.2/onnx_torchscript.html "TorchScript-based ONNX Exporter — PyTorch 2.2 documentation"
[2]: https://docs.pytorch.org/docs/2.2/export.html "torch.export — PyTorch 2.2 documentation"
[3]: https://docs.pytorch.org/docs/2.2/fx.html "torch.fx — PyTorch 2.2 documentation"
[4]: https://docs.pytorch.org/docs/2.2/jit.html "TorchScript — PyTorch 2.2 documentation"

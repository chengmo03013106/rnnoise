学习范围 → 学习材料 → 最低掌握标准 → 实验/代码验收 → 能回答的面试问题

# Model Deployment

路线：

```text
PyTorch model
 ↓
ONNX
 ↓
ONNX Runtime
 ↓
INT8
 ↓
C++
 ↓
Benchmark
```

---

| 级别    | 标准                    | 举例                              |
| ----- | --------------------- | ------------------------------- |
| L1 理解 | 能不用看资料解释概念、区别、原理      | `eval()` 和 `no_grad()` 为什么不是一回事 |
| L2 使用 | 能脱离教程写最小代码            | 自己保存/加载 `state_dict`            |
| L3 验证 | 能在 RNNoise 项目中使用并拿到结果 | PyTorch → ONNX → ORT 输出对齐       |
| L4 排错 | 出问题时能自己定位             | ONNX 输出和 PyTorch 不一致，知道从哪里查     |

对，这次我建议把 Phase 2 **重新定义成“验收驱动学习”**，不再使用“学习 ONNX、学习 ORT、学习量化”这种过于宽泛的表述。

你现在已经进入一个非常关键的阶段：**不能再靠“看完资料”判断自己学会了。**

从 Phase 2 开始，每个知识点都必须有 5 个东西：

> **学习范围 → 学习材料 → 最低掌握标准 → 实验/代码验收 → 能回答的面试问题**

而且我建议采用一个固定原则：

> **文档是主教材，源码/实验是核心，视频只是辅助。**
>
> 对你这种已经有多年 C++/Linux/服务端经验的人，视频不应该成为主线。你真正需要的是准确理解 API、数据流、运行时和性能行为，然后自己验证。

下面把你现在的 Phase 2 完整重构。

---

# 二、Phase 2 第一周：PyTorch → ONNX

时间：**10 月 4 日 ～ 10 月 10 日**

目标不是“学 ONNX”。

真正目标是：

> **理解一个 PyTorch 模型怎样变成一个独立于 PyTorch 的计算图，并且你能够检查这个图到底是什么。**

---

# 1. `model.eval()`

## 学什么

你需要知道：

```python
model.train()
model.eval()
```

改变的是什么。

重点不是 API 怎么调用，而是理解：

> `eval()` 是切换 **module 的 evaluation behavior**，不是关闭梯度。

PyTorch 官方明确说明 `eval()` 和 gradient tracking 是两个正交机制；`eval()` 主要影响 Dropout、BatchNorm 等具有 train/eval 行为差异的模块。([PyTorch Docs][1])

---

## 必须理解

你需要能够解释：

```text
model.eval()
```

和：

```text
torch.no_grad()
```

分别负责什么。

### 正确理解

```text
eval()
↓
模型行为模式

no_grad()
↓
Autograd 是否记录计算图
```

所以：

```python
model.eval()
with torch.no_grad():
    y = model(x)
```

是一个非常典型的 inference 写法。PyTorch 官方教程也采用这种模式。([PyTorch Docs][2])

---

## 验收实验

自己构造：

```python
class TestNet(nn.Module):
    def __init__(self):
        ...
        self.dropout = nn.Dropout(...)
```

然后：

```text
train()
同一个 x
运行两次

eval()
同一个 x
运行两次
```

观察结果。 ✅ 

再测试：

```text
普通 forward
vs
torch.no_grad()
```

观察：✅

```python
y.requires_grad
y.grad_fn
```

---

## 学会标准

你能回答：

> 为什么 `model.eval()` 不能代替 `torch.no_grad()`？

> 为什么 inference 通常同时使用 `eval()` 和 `no_grad()`？

> 一个完全没有 Dropout/BatchNorm 的 RNNoise 模型，`eval()` 是否会明显改变当前计算结果？

做到这里就够。

---

# 2. `torch.no_grad()`

## 学什么

不是背：

> 不计算梯度。

而是理解：

```text
正常 forward
↓
Autograd 记录计算历史
↓
后续可以 backward

no_grad
↓
不记录这部分梯度计算
```

PyTorch 官方文档明确把它定位为 inference/validation 中关闭 gradient tracking 的机制，并指出可以降低计算和内存开销。([PyTorch Docs][3])

---

## 验收

你自己写：

```python
x = ...
y = model(x)
print(y.requires_grad)

with torch.no_grad():
    y2 = model(x)

print(y2.requires_grad)
```

能够解释：

```text
为什么不同。
```

并能够解释：

> `no_grad()` ≠ `eval()`。

---

# 3. `state_dict`

这个知识点对你特别重要，因为它会第一次把：

```text
模型结构
```

和：

```text
模型参数
```

分离。

PyTorch 官方推荐用 `state_dict()` 保存模型的学习参数，并在恢复模型结构后使用 `load_state_dict()` 加载；推理前再调用 `eval()`。([PyTorch Docs][4])

---

## 学什么

你必须搞明白：

```text
Model class
+
state_dict
=
完整可运行模型
```

比如：

```python
model = MyRNNoise(...)
state = model.state_dict()
```

你要知道里面是什么：

```text
conv1.weight
conv1.bias
gru1.weight_ih
gru1.weight_hh
...
```

以及：

> `state_dict` 保存的不是 Python class 定义。

---

## 验收实验

完成：

```text
模型 A
↓
state_dict
↓
保存

重新创建模型 B
↓
load_state_dict
↓
输入同一个 x
↓
比较 A/B 输出
```

结果必须一致到你设定的浮点误差范围。

---

## 能回答

> `state_dict` 是模型结构还是模型参数？

> 为什么保存 `state_dict` 后，加载时还需要重新创建模型？

> `state_dict` 和整个 `torch.save(model)` 有什么区别？

> 为什么部署时通常更关心模型参数，而不是 Python class 本身？

---

# 4. ONNX

这里不要一开始去背 ONNX 全部规范。

你当前只需要建立：

```text
PyTorch Model
       ↓
ONNX
       ↓
Graph
       ↓
Node
       ↓
Operator
       ↓
Tensor
```

ONNX 的规范本质上是一个可序列化的计算图表示；模型包含 graph，graph 包含 nodes、inputs、outputs、initializers 等，node 调用某个 operator。([ONNX][5])

---

# 5. `ONNX graph`

这是 Phase 2 第一周最重要的知识。

你需要明确区分：

```text
Graph
Node
Operator
Tensor/Value
Initializer
Input
Output
```

例如：

```text
Graph
│
├── input
│
├── Node 1
│      op_type = MatMul
│
├── Node 2
│      op_type = Add
│
├── Node 3
│      op_type = Tanh
│
└── output
```

这里：

> **Node 是一个具体计算节点。**

> **Operator 是这个节点执行什么语义。**

ONNX 的规范明确规定 graph 是拓扑排序的 node 列表，每个 node 通过 `op_type` 指定所调用的 operator。([ONNX][5])

---

# 6. 这部分怎么学？

不要看 2 小时视频。

直接做：

```text
PyTorch RNNoise
        ↓
export
        ↓
rnnoise.onnx
```

然后：

```python
import onnx

model = onnx.load(...)
onnx.checker.check_model(model)
```

再用 Netron 看图。

---

# 7. ONNX 第一周验收实验

你必须做到：

### 实验 A

成功产生：

```text
rnnoise.onnx
```

### 实验 B

自己列出：

```text
graph inputs
graph outputs
node count
operator types
```

### 实验 C

随便挑 5 个 node，能够说：

```text
这个 node 输入是什么
这个 node 输出是什么
op_type 是什么
这个 operator 在 PyTorch 哪一部分对应
```

### 实验 D

找到：

```text
Conv
GRU
Gemm / MatMul
Add
Tanh
Sigmoid
```

或者你实际导出的模型中对应的 operator。

**不要要求一定出现这些名字。**

因为你的 PyTorch 实现方式不同，exporter 可能把一个高层模块表示成一个 ONNX operator，也可能展开成多个 primitive operators。

---

# 8. 第一周最终验收

你应该能够自己面对面回答：

> PyTorch 模型为什么需要导出成 ONNX？

> ONNX 是模型文件格式还是计算图表示？

> graph、node、operator 三者是什么关系？

> weight 在 ONNX 里放在哪里？

> ONNX 的 input/output 和 initializer 是什么？

> 一个 `nn.Linear` 导出以后一定是一个 `Linear` operator 吗？

最后这个问题尤其重要。

因为你刚刚在 RNNoise C 里面学到：

```text
Conv1D
↓
compute_linear
```

现在你开始碰到同样的问题：

```text
High-level op
↓
lowering
↓
primitive operators
```

这正是以后 TensorRT/ONNX Runtime/CUDA 会反复遇到的东西。

---

# 第一周最终产物

不是：

> “看完 ONNX 教程。”

而是：

```text
RNNoise PyTorch
      ↓
rnnoise.onnx
      ↓
能解释 ONNX graph
      ↓
能检查 node/operator
      ↓
PyTorch output ≈ ONNX output
```

**必须加入最后这个 output 对比。**

否则你只是“会 export”，还不算真正完成。

---

# 三、Phase 2 第二周：ONNX Runtime + C++

时间：

**10 月 11 日 ～ 10 月 17 日**

目标：

> **把“模型文件”变成“运行时”。**

---

# 1. ONNX Runtime 到底学什么？

不要把 ONNX Runtime 理解成：

> “一个可以运行 ONNX 的 Python 包。”

你要理解它大概是：

```text
ONNX graph
     ↓
Runtime
     ↓
Graph optimization
     ↓
Execution Provider
     ↓
Kernel
     ↓
CPU/GPU/NPU
```

ONNX Runtime 通过 Execution Provider 把 graph 中的 node/subgraph 分配给具体硬件后端执行；同一套 ORT API 可以配合不同 EP 使用。当前官方文档列出了默认 CPU、CUDA、TensorRT、OpenVINO、XNNPACK 等多个 EP。([ONNX Runtime][6])

---

# 2. Execution Provider 学到什么程度？

你不需要现在学：

```text
CUDA EP 源码
TensorRT EP 源码
```

只需要真正理解：

> **EP 是“实际执行算子的后端”。**

例如：

```text
ONNX Graph
    ↓
CPU EP
    ↓
CPU kernel
```

以后：

```text
ONNX Graph
    ↓
CUDA EP
    ↓
CUDA kernel
```

再以后：

```text
ONNX Graph
    ↓
TensorRT EP
    ↓
TensorRT engine
```

---

## 验收

运行：

```python
session = ort.InferenceSession(...)
```

打印：

```text
providers
```

能够解释当前为什么是：

```text
CPUExecutionProvider
```

并且知道：

> 如果换 GPU EP，改变的不是你的上层 Python/C++ 调用方式，而是 graph 的执行后端。

---

# 3. ORT Python API

必须掌握：

```text
InferenceSession
SessionOptions
get_inputs()
get_outputs()
run()
```

以及：

```text
input name
input shape
input dtype
output name
```

---

## 验收

自己写：

```python
sess = ort.InferenceSession("rnnoise.onnx")

inputs = sess.get_inputs()
outputs = sess.get_outputs()

...
sess.run(...)
```

然后：

```text
PyTorch
vs
ONNX
vs
ORT
```

三者比较。

---

# 4. 这里开始进入你的 C++ 优势区

你需要理解的 C++ 核心对象：

```text
Ort::Env
Ort::SessionOptions
Ort::Session
Ort::Value
Allocator
Run()
```

官方 ORT C API 的核心执行模型就是建立 `Session`，准备输入 `OrtValue`，调用 `Run()` 获取输出；C/C++ API 还提供了 session 输入输出信息查询等接口。([ONNX Runtime][7])

你不需要把整个 ORT API 学完。

---

# 5. 第二周的真正验收标准

完成：

```text
                 RNNoise.onnx
                      │
           ┌──────────┴─────────┐
           ↓                    ↓
       ORT Python           ORT C++
           │                    │
           └──────────┬─────────┘
                      ↓
                  same input
                      ↓
              output comparison
```

必须做到：

```text
同一模型
同一输入
同一 session 配置
```

得到：

```text
PyTorch
≈
ORT Python
≈
ORT C++
```

---

# 6. 第二周必须能够回答的问题

> ONNX Runtime 和 ONNX 是什么关系？

> 为什么 ONNX 是 graph，而 ORT 是 runtime？

> Execution Provider 是什么？

> Session 是什么？

> 为什么 Session 不应该每个 request 都重新创建？

> `Run()` 大致做了什么？

> Python ORT 和 C++ ORT 的底层 runtime 是不是两套？

> CPU EP 和 CUDA EP 的差异是什么？

> 如果一个模型只有部分 operator 被某个 EP 支持，会发生什么？

最后一个问题非常重要，因为 ORT 会根据 EP 对 graph 进行分配。官方文档也明确说明 EP 可以领取特定 nodes/subgraphs；实际部署时 operator 支持和分配情况会影响性能。([ONNX Runtime][6])

---

# 7. 第二周建议增加一个你以后很有用的知识点：Graph Optimization

不要深入源码。

只搞懂：

```text
ONNX Graph
    ↓
ORT graph optimization
    ↓
optimized graph
    ↓
execution
```

比如：

```text
Constant Folding
Node Elimination
Operator Fusion
Layout Optimization
```

ORT 官方目前把 graph optimization 分成 Basic、Extended、Layout 等层级，并支持在线/离线优化。([ONNX Runtime][8])

你的验收只需要：

> 能举一个 graph optimization 的例子，并解释为什么 fusion 可能提高性能。

例如：

```text
Conv
+
Add
```

融合以后：

```text
Conv with bias
```

减少一个 node 和中间结果。

这就够了。

---

# 四、Phase 2 第三周：Quantization

时间：

**10 月 18 日 ～ 10 月 24 日**

这一周千万不能变成：

> “运行一个 `quantize_dynamic()` 就结束。”

这个阶段实际上是你第一次真正进入 **模型部署优化**。

---

# 1. 先分清 FP32 / FP16 / INT8

你必须能够解释：

```text
FP32
↓
32 bit floating point

FP16
↓
16 bit floating point

INT8
↓
8 bit integer representation
```

重点理解：

> **FP16 和 INT8 不是同一种类型的优化。**

FP16 是 floating-point precision reduction。

INT8 是 quantization。

---

# 2. INT8 第一核心公式

必须自己推导并写出来：

$$
x_{fp32}=scale \cdot (x_{int8}-zero\_point)
$$

ONNX Runtime 当前官方量化文档也采用这个 affine mapping 定义。([ONNX Runtime][9])

然后真正理解：

```text
scale
zero_point
range
```

---

# 3. 必须自己做一个手工量化实验

不要直接用 ORT。

例如：

```text
float range:
[-1.0, 1.0]
```

自己计算：

```text
scale
zero_point
```

然后：

```text
float
↓
quantize
↓
int8
↓
dequantize
↓
float
```

比较：

```text
original
vs
reconstructed
```

得到：

```text
max_abs_error
mean_abs_error
```

这一步非常重要。

因为以后你面试被问：

> “INT8 为什么会有精度损失？”

你不能只回答：

> 因为精度降低了。

你应该能说：

> float 连续空间映射到离散量化空间，有限 bit 表达无法无损表示原始数值，因此会产生 rounding / clipping 等误差。

---

# 4. dynamic quantization

你必须理解：

```text
weight
↓
quantized

activation
↓
inference runtime 时动态计算 scale
```

ORT 的官方定义是，dynamic quantization 在运行时计算 activation 的 quantization parameters；这种方法会增加推理开销。([ONNX Runtime][9])

---

# 5. static quantization

理解：

```text
calibration dataset
       ↓
跑模型
       ↓
统计 activation range
       ↓
scale / zero_point
       ↓
保存到模型
       ↓
实际 inference
```

ORT 官方静态量化需要 calibration data，并支持 MinMax、Entropy、Percentile 等 calibration 方法。([ONNX Runtime][9])

---

# 6. 但对你的 RNNoise，不要把 dynamic/static 当成考试题

非常重要。

ORT 官方目前对方法选择给出的经验建议是：

> dynamic quantization 通常更适合 RNN / Transformer；static quantization 通常更适合 CNN。([ONNX Runtime][9])

而你的 RNNoise：

```text
Conv1D
+
GRU × 3
```

所以：

### 第一优先级

先做：

```text
FP32
↓
dynamic INT8
```

跑通。

### 第二优先级

再研究：

```text
static INT8
```

因为你需要理解 calibration。

### 不建议

一开始同时搞：

```text
dynamic
static
QDQ
QOperator
per-channel
per-tensor
QAT
```

这会再次陷入“什么都懂一点”。

---

# 7. 还要特别注意一个现实问题

**不要默认 RNNoise 的整个 ONNX graph 会变成 INT8。**

应该实际检查：

```text
哪些 weights quantized
哪些 activations quantized
哪些 operators quantized
哪些仍然 FP32
```

ORT 的量化表示可以是 QOperator 或 QDQ；量化工具也提供量化调试方法，通过比较 FP32 和量化模型的 weights/activations 来定位误差来源。([ONNX Runtime][9])

这正好延续你 Phase 1 的思维：

```text
C vs PyTorch
```

现在变成：

```text
FP32 vs INT8
```

---

# 8. 第三周验收

至少完成：

```text
manual quantization
+
ORT dynamic quantization
+
FP32 / INT8 output comparison
```

输出：

```text
max_abs_error
mean_abs_error
```

再做：

```text
model size FP32
vs
model size INT8
```

以及第一次 latency：

```text
FP32 latency
vs
INT8 latency
```

---

# 9. 第三周必须能回答

> INT8 的 scale 和 zero point 是什么？

> symmetric / asymmetric quantization 区别是什么？

> dynamic quantization 和 static quantization 区别是什么？

> calibration 是什么？

> 为什么 static quantization 需要 calibration dataset？

> 为什么量化后结果不会完全一致？

> 为什么量化有时反而更慢？

最后一个问题不是理论题。

ORT 官方也特别指出，量化性能收益取决于硬件；量化还存在 quantize/dequantize overhead，某些硬件上 INT8 可能反而没有收益。([ONNX Runtime][9])

这与你以后做性能工程非常相关。

---

# 五、Phase 2 第四周：Benchmark + C++ 完整 Demo

时间：

**10 月 25 日 ～ 10 月 31 日**

这一周不是学习新技术。

而是：

> **把前 3 周知识变成一个工程系统。**

---

# 1. 最终结构

你最终应该得到：

```text
                    RNNoise
                       │
                  PyTorch FP32
                       │
                      ONNX
                       │
          ┌────────────┴────────────┐
          │                         │
      ORT FP32                  ORT INT8
          │                         │
          └────────────┬────────────┘
                       │
                     C++
                       │
                   benchmark
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       latency       CPU         memory
```

---

# 2. Benchmark 必须先定义方法

否则：

> `INT8 = 0.8ms`

没有意义。

你应该固定：

```text
CPU model
OS
compiler
ORT version
thread count
model
input shape
input data
warm-up count
benchmark iterations
```

---

# 3. Warm-up

例如：

```text
创建 Session
↓
warmup 100 次
↓
正式测试 1000 次
```

**不要把 session 创建时间算进 inference latency。**

因为你测的是：

> inference。

不是：

> model loading + initialization。

---

# 4. 指标

你列出的：

```text
Latency
p50
p95
CPU
Memory
Model size
Output error
```

全部保留。

我建议增加：

```text
p99
```

以及：

```text
real-time factor
```

对于你的实时音频场景尤其有意义。

---

# 5. CPU

不要只看：

```text
top
```

最好至少做到：

```text
平均 CPU utilization
```

并记录：

```text
ORT intra-op threads
ORT inter-op threads
```

因为 ORT CPU EP 有 intra-op / inter-op execution settings，这些线程参数会直接影响性能。([ONNX Runtime][10])

这个地方特别适合你的背景。

你可以进一步做：

```text
threads = 1
threads = 2
threads = 4
threads = 8
```

然后画：

```text
thread count
vs
p95 latency
```

这已经开始是一个真正的 **AI inference performance experiment**。

---

# 6. 第四周最终验收标准

你必须能够交出一份：

```text
README.md
```

里面至少有：

```text
1. Architecture
2. Model
3. ONNX
4. ORT
5. C++ inference
6. INT8
7. Benchmark methodology
8. Results
9. Accuracy comparison
10. Analysis
```

并且别人 clone 下来可以：

```text
build
↓
run
↓
benchmark
```

这时候这个项目才真正可以放简历。

---

# 六、Phase 2 四周最终“知识验收表”

这个表我建议你以后一直维护。

| 知识                 | 通过标准                     | 必须能回答                        |
| ------------------ | ------------------------ | ---------------------------- |
| `eval()`           | toy model 验证             | eval 和 no_grad 区别            |
| `no_grad()`        | 验证 requires_grad/grad_fn | 为什么 inference 要它             |
| `state_dict`       | save/load + output 对齐    | 参数和模型结构区别                    |
| ONNX Graph         | 能检查 graph/node/op        | graph/node/operator 区别       |
| Operator           | 找到实际 ONNX operator       | high-level op 如何 lower       |
| ONNX export        | PyTorch → ONNX           | 为什么导出后仍能运行                   |
| ORT Session        | Python 跑通                | Session 是什么                  |
| EP                 | 查看 CPU EP 并解释            | EP 干什么                       |
| ORT C++            | C++ 跑通                   | Session/Value/Run            |
| Graph optimization | 至少理解 fusion              | 为什么优化 graph                  |
| INT8               | 手工完成一次                   | scale/zero point             |
| Dynamic            | 实际跑 dynamic              | dynamic 怎么算 activation scale |
| Static             | 理解 calibration 并最好实践     | calibration 为什么需要            |
| FP32/INT8          | 输出对比                     | 为什么有误差                       |
| Benchmark          | p50/p95/p99 + CPU/RSS    | 如何设计可信 benchmark             |

---

# 七、这里我建议对你原来的 Phase 2 做一个重要调整

你原来的计划是：

> 第一周学 `eval/no_grad/state_dict/ONNX`

> 第二周学 `ORT/EP/C++`

> 第三周学 `quantization`

> 第四周 benchmark

**这个大的顺序不变。**

但每周不要再按照：

```text
看资料
↓
看视频
↓
感觉懂了
```

进行。

改成：

```text
学习 30%
↓
自己写 20%
↓
接入 RNNoise 30%
↓
实验/排错 20%
```

比如第一周：

```text
ONNX 文档
    ↓
自己 export
    ↓
检查 graph
    ↓
Netron
    ↓
PyTorch / ONNX output comparison
```

这才叫学会。

---

# 八、学习资料的优先级，也给你固定下来

以后不要到处找教程。

## 第一优先级：官方文档

用于：

```text
API
语义
限制
版本
配置
```

例如：

PyTorch 官方明确说明 `eval()` 与 gradient tracking 正交；`state_dict` 是推荐的模型参数保存方式。([PyTorch Docs][1])

ONNX 官方规范定义 graph、node、operator、initializer 等核心概念。([ONNX][5])

ONNX Runtime 官方文档定义 EP、graph optimization、quantization、C++ API 等运行机制。([ONNX Runtime][8])

---

## 第二优先级：源码

例如：

```text
PyTorch model
ONNX model
ORT C++ demo
```

尤其你是 C++ 背景。

源码阅读对你应该比视频重要。

---

## 第三优先级：自己实验

这是最高权重。

例如：

> `eval()` 到底改变什么？

不要看别人解释。

自己写一个 Dropout 实验。

---

## 第四优先级：视频

视频只用于：

> **某个概念你连续看官方文档还是没有形成直觉。**

例如：

```text
KV Cache
GPU memory
CUDA warp
```

看一个 30～60 分钟的高质量视频可以。

但是：

> 不要“看完视频 = 学完知识”。

---

# 九、我建议你从现在开始，每个学习主题都用同一张卡片记录

比如 `onnx_operator.md`：

```text
# ONNX Operator

## 1. 一句话定义

## 2. 我自己的理解

## 3. 和 Node 的区别

## 4. RNNoise 中出现了哪些 Operator

## 5. 一个 Operator 的输入输出是什么

## 6. PyTorch -> ONNX 的对应关系

## 7. 我做过什么实验

## 8. 实验结果

## 9. 我现在能回答的问题

## 10. 我还不会的问题
```

你以后继续：

```text
ORT
EP
Quantization
TensorRT
CUDA
vLLM
KV Cache
```

全部使用这个模板。

**这会解决你之前说的“什么都学一点，但无法持续投入一个方向”的问题。**

因为你以后不会再以：

> “我看过 ONNX。”

作为学习进度。

而会记录：

> “我能够解释 ONNX graph/node/operator，并且我亲手检查了 RNNoise 导出的 37 个 node，其中哪些来自 Linear/GRU，哪些是 primitive op；我能比较 PyTorch/ONNX 输出。”

这才是可积累的能力。

---

# 十、Phase 2 完成后的真正验收，不是“我有一个 Demo”

最终我建议你给自己做一次**闭卷考试**。

不看资料，面对自己的项目，回答这 20 个问题：

### PyTorch

1. `eval()` 和 `no_grad()` 有什么区别？
2. `state_dict` 保存什么？
3. 为什么加载 `state_dict` 之前必须先构造模型？

### ONNX

4. ONNX 是什么？
5. Graph / Node / Operator 区别？
6. Initializer 是什么？
7. PyTorch 的 `Linear` 为什么未必直接变成一个 “Linear node”？

### ORT

8. ONNX Runtime 和 ONNX 什么关系？
9. Session 是什么？
10. Execution Provider 是什么？
11. CPU EP 做了什么？
12. graph optimization 为什么能提高性能？

### Quantization

13. INT8 为什么需要 scale？
14. zero point 是什么？
15. dynamic / static quantization 区别？
16. calibration 是什么？
17. INT8 为什么可能更慢？
18. 为什么 INT8 输出和 FP32 不完全一致？

### Engineering

19. 为什么 benchmark 要 warm-up？
20. 为什么 Session 创建时间不能直接算 inference latency？

**如果这 20 个问题你能不用资料完整回答，而且能用自己的 RNNoise 项目举例，Phase 2 才算真正完成。**

---

# 十一、我还会把“Phase 2 完成”的职业意义重新定义一下

你原来写：

> Phase 2 是 AI Model Deployment Engineer 的雏形。

这个方向基本正确，但我建议定义得更准确：

### Phase 1

你证明：

> **我能读模型。**

```text
C
↓
数学
↓
PyTorch
↓
模型
```

### Phase 2

你证明：

> **我能部署模型并进行基础优化。**

```text
PyTorch
↓
ONNX
↓
ORT
↓
C++
↓
INT8
↓
Benchmark
```

### Phase 3

开始证明：

> **我能理解现代 LLM inference。**

```text
Transformer
Attention
KV Cache
Prefill
Decode
Batching
```

### Phase 4

开始证明：

> **我能做 inference system。**

```text
vLLM
scheduler
continuous batching
KV Cache
GPU
CUDA
```

### Phase 5

才开始证明：

> **我有 AI Infra / AI Systems 能力。**

```text
TensorRT
CUDA optimization
multi-GPU
NCCL
RDMA
distributed inference
```

这个层级就不会乱。

---

# 十二、最重要的：10 月 4 日以后，你的学习方式应该改变

你之前 Phase 1 的模式是：

> **理解 RNNoise 中一个函数 → 自己实现 → 对比。**

Phase 2 应该升级为：

> **理解一个 AI deployment concept → 在 RNNoise 上使用 → 观察运行时行为 → benchmark → 能解释结果。**

所以你现在不是：

```text
学习 ONNX
```

而是：

```text
为什么需要 ONNX？
↓
RNNoise 怎么导出？
↓
导出了什么 graph？
↓
graph 里面有什么 operator？
↓
ONNX 输出是否与 PyTorch 一致？
↓
出问题我会不会定位？
```

也不是：

```text
学习 ONNX Runtime
```

而是：

```text
ONNX graph
↓
ORT Session
↓
CPU EP
↓
Run()
↓
C++
↓
输出对齐
↓
thread tuning
↓
latency
```

也不是：

```text
学习 INT8
```

而是：

```text
FP32
↓
为什么量化
↓
scale / zero point
↓
dynamic INT8
↓
实际运行
↓
误差
↓
性能
↓
为什么快/为什么不快
```

**这就是从 Phase 2 开始，你应该建立的学习模式。**

而且这套模式与我们之前确定的职业主线是完全一致的：你不是在单纯学习“AI 名词”，而是在逐步证明自己具备 **Model → Runtime → Optimization → C++ → Performance** 这一条完整链路。

当前 ORT 官方文档也正好覆盖了这条链：EP、图优化、CPU 线程调优、量化和 C/C++ runtime API 都是独立而又能串起来的能力模块。([ONNX Runtime][8])

**所以从今天 10 月 4 日开始，我建议你把 Phase 2 的“完成度”从百分比改成验收项：每一项都必须有“我能解释 + 我写过 + 我验证过 + 我能排错”。** 这样你到 Phase 2 结束时，手上的 RNNoise 项目才不仅是一个 demo，而是一份能够在面试中逐层展开的 AI Model Deployment / Inference Engineering 项目。

[1]: https://docs.pytorch.org/docs/main/notes/autograd.html?utm_source=chatgpt.com "Autograd mechanics — PyTorch main documentation"
[2]: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial?utm_source=chatgpt.com "Optimizing Model Parameters — PyTorch Tutorials 2.14.0+cu130 documentation"
[3]: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide?utm_source=chatgpt.com "Performance Tuning Guide — PyTorch Tutorials 2.14.0+cu130 documentation"
[4]: https://docs.pytorch.org/tutorials/beginner/saving_loading_models?utm_source=chatgpt.com "Saving and Loading Models — PyTorch Tutorials 2.14.0+cu130 documentation"
[5]: https://onnx.ai/onnx/repo-docs/IR.html?utm_source=chatgpt.com "Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification - ONNX 1.24.0 documentation"
[6]: https://onnxruntime.ai/docs/execution-providers/?utm_source=chatgpt.com "Execution Providers | onnxruntime"
[7]: https://onnxruntime.ai/docs/api/c/struct_ort_api.html?utm_source=chatgpt.com "ONNX Runtime: OrtApi Struct Reference"
[8]: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html?utm_source=chatgpt.com "Graph optimizations | onnxruntime"
[9]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html?utm_source=chatgpt.com "Quantize ONNX models | onnxruntime"
[10]: https://onnxruntime.ai/docs/performance/tune-performance/threading.html?utm_source=chatgpt.com "Thread management | onnxruntime"

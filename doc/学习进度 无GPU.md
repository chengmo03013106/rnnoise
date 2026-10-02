是的。按照你今天的实际完成情况，我会把计划重新排一次，而且这次我会把**“在家 gap、没有高端 GPU/集群、目标是尽快形成可就业能力”**作为硬约束，而不是按理想 AI Infra 工程师的完整培养路线来安排。

先给你一个核心判断：

> **你的进度比原来的“10 月 20 日完成 Phase 1”计划提前了，但不是提前很多；现在最重要的不是继续加速，而是把已经完成的 GRU 做成真正可验证、可复用的能力，然后进入 ONNX / ORT / INT8。**
>
> **GPU、CUDA、TensorRT、NCCL、RDMA 不应该现在成为你的学习阻塞点。它们保留在路线里，但真正的 GPU/多机实验要等后面，用远程 GPU/云机器解决，不应该因为家里没有 NVIDIA GPU 而停摆，更不应该现在花大量钱买设备。**

---

# 一、先判断你现在到底提前了多少

原来的计划是：

```text
9/23 ～ 9/29
RNN
GRU
自己实现 GRU
```

今天 9 月 29 日，你已经完成：

```text
Python 深度学习基础
        ↓
compute_linear
        ↓
compute_generic_conv1d
        ↓
PyTorch 重写
        ↓
C / Python 数值验证
        ↓
RNN / GRU 理论
        ↓
GRU Python 手写
        ↓
按照 C 实现方式重写
        ↓
C / Python 对齐
        ↓
发现 sigmoid / tanh 精度差异
        ↓
自己实现低精度 sigmoid / tanh
        ↓
分析 state 偏差
```

所以按**知识节点**来说，你确实比原来的 Phase 1 进度提前。

但是我要特别指出：

> **现在不能因为 GRU 已经跑通，就认为 Phase 1 已经快结束。**

因为 Phase 1 最终不是：

> “会 GRU。”

而是：

> **完成 RNNoise 整个核心 RNN 网络的理解、手写和逐层验证。**

你后面还剩：

```text
GRU 连续 state 验证
↓
Conv1 + Conv2 + GRU1/2/3
↓
output layer
↓
完整 Tensor shape
↓
整个 C/Python 网络逐层对齐
↓
误差分析报告
```

所以我会评价：

### 你的进度：**提前约 5～10 天，但不应该再压缩。**

原来的 Phase 1 结束日期 10 月 20 日，现在可以比较保守地提前到：

# **10 月 12 日完成 Phase 1**

留下 8 天缓冲。

这个安排比“既然已经提前，那 10 月 5 日就完成”更适合你的 gap 状态。

---

# 二、我现在重新确定你的最终目标

这一点必须非常明确。

你的最终目标不是：

> “把 RNNoise 学完。”

也不是：

> “学会 vLLM。”

更不是：

> “什么 AI 都懂一点。”

而是：

# **C++ / Linux 高性能系统工程师 → AI Inference / AI Systems → AI Infrastructure**

最终能力链：

```text
C++
Linux
高并发
网络
分布式
性能优化
实时系统
        │
        ↓
PyTorch / Model
        │
        ↓
ONNX / Runtime
        │
        ↓
Quantization
        │
        ↓
C++ Inference
        │
        ↓
Transformer / LLM
        │
        ↓
vLLM / Serving
        │
        ↓
CUDA / GPU / TensorRT
        │
        ↓
Inference Performance
        │
        ↓
Distributed Inference
        │
        ↓
AI Systems / AI Infra
```

这条路线保留我们之前确定的全部内容。

---

# 三、但是你的设备决定了：这条路线必须分成“本机学习”和“远程实验”

你目前：

### 机器 A

> Intel Mac x86 笔记本

### 机器 B

> Windows + WSL2，性能略强于笔记本

目前你没有提到 NVIDIA GPU，所以我按照：

> **两台机器都是 CPU 为主，没有可用 NVIDIA GPU**

来安排。

如果台式机其实有 NVIDIA GPU，只需要把后面的远程 GPU 部分改为本地即可。

---

# 四、先给你设备结论

| 技术                    | Mac x86 | Windows + WSL2 | 是否需要额外环境            |
| --------------------- | ------- | -------------- | ------------------- |
| Python / PyTorch CPU  | ✅       | ✅              | 不需要                 |
| RNNoise C             | ✅       | ✅              | 不需要                 |
| PyTorch 手写模型          | ✅       | ✅              | 不需要                 |
| ONNX                  | ✅       | ✅              | 不需要                 |
| ONNX Runtime CPU      | ✅       | ✅              | 不需要                 |
| INT8 CPU              | ✅       | ✅              | 不需要                 |
| C++ ORT               | ✅       | ✅              | 不需要                 |
| WebRTC / RNNoise Demo | ✅       | ✅              | 不需要                 |
| Transformer CPU       | ✅       | ✅              | 不需要                 |
| 小模型 vLLM              | ⚠️      | ✅              | 用 WSL2              |
| vLLM GPU              | ❌       | ❌*             | NVIDIA GPU          |
| CUDA                  | ❌       | ❌*             | NVIDIA GPU          |
| TensorRT GPU          | ❌       | ❌*             | NVIDIA GPU          |
| Nsight GPU profiling  | ❌       | ❌*             | NVIDIA GPU          |
| NCCL 多 GPU            | ❌       | ❌*             | 多 NVIDIA GPU        |
| RDMA / InfiniBand     | ❌       | ❌              | GPU + NIC + RDMA 环境 |

* 如果你的 Windows 台式机没有 NVIDIA GPU。

这里有几个关键事实。

ONNX Runtime 本身有 CPU Execution Provider，所以你前面的模型部署和 INT8 工作完全不需要 GPU；它的 CUDA/TensorRT Execution Provider 才需要 NVIDIA GPU。([ONNX Runtime][1])

vLLM 当前支持 x86 CPU，但 CPU 版本主要用于基本推理和 serving；官方要求 Linux，Intel/AMD x86 需要 AVX2/AVX512 等 CPU 特性。因此你的 **WSL2 是比 Intel Mac 更合适的 vLLM 学习环境**。([vLLM][2])

Windows 本身不是 vLLM 的原生支持平台，但官方文档明确给出了通过 WSL 运行的方式，所以你的 Windows + WSL2 是可以利用的。([vLLM][3])

CUDA 在 WSL2 中也可以跑，但前提是 Windows 主机上有兼容的 NVIDIA GPU 和对应驱动。没有 GPU，WSL2 不会凭空提供 CUDA GPU。([NVIDIA Docs][4])

而 NCCL 本质上就是 GPU-GPU collective communication 库，多 GPU / NVLink / GPU Direct / RDMA 这些内容天然需要真实 GPU 拓扑和相应网络环境；RDMA 还涉及兼容 NIC 和 GPUDirect RDMA 等条件。([NVIDIA Docs][5])

所以：

# 现在不要为了学习路线买昂贵 GPU。

至少在 Phase 1～3，没有这个必要。

---

# 五、你的新总时间表

我重新排成：

| 阶段      | 时间            | 目标                                 |
| ------- | ------------- | ---------------------------------- |
| Phase 1 | 9/29 ～ 10/12  | RNNoise Neural Network             |
| Phase 2 | 10/13 ～ 11/10 | ONNX / ORT / INT8 / C++            |
| Phase 3 | 11/11 ～ 12/22 | Transformer / LLM Inference / vLLM |
| Phase 4 | 12/23 ～ 2/2   | CUDA / GPU / TensorRT / Profiling  |
| Phase 5 | 2/3 ～ 3/30    | Distributed Inference / AI Infra   |
| Phase 6 | 3/31以后        | Training Infra，作为后续扩展              |

注意：

**Phase 6 不是当前就业主线。**

当前真正的职业目标在 Phase 5：

> **AI Inference / AI Systems / AI Infra**

Training Infra 后续再决定是否继续。

---

# 六、Phase 1：9/29 ～ 10/12

## 目标

完成：

```text
RNNoise Neural Network
```

而不是完整 RNNoise DSP。

最终：

```text
Conv1
 ↓
Conv2
 ↓
GRU1
 ↓
GRU2
 ↓
GRU3
 ↓
Output
```

能够：

```text
C
↓
数学
↓
Tensor shape
↓
PyTorch
↓
C/Python 数值验证
```

---

## 10/7

开始 output layers。

目标：

> 把核心 RNN 网络跑通。

---

## 10/8

建立最终 Tensor Shape 表。

必须包含：

```text
layer
input shape
output shape
weight shape
bias shape
state shape
```

---

## 10/9

完整随机输入测试。

检查：

```text
NaN
Inf
shape
dtype
state
```

---

## 10/10

最终：

```text
C
vs
Python
```

逐层：

```text
Conv1
Conv2
GRU1
GRU2
GRU3
Output
```

---

## 10/11

整理：

```text
phase1.md
```

最终结构：

```text
Architecture
↓
C function
↓
Math
↓
Tensor shape
↓
PyTorch
↓
Numerical comparison
↓
Mismatch analysis
```

---

## 10/12

# Phase 1 正式验收

达到：

> **你可以脱离 C 代码解释 RNNoise RNN 部分的计算过程，并可以自己重新实现。**

如果 10 月 12 日没完全通过，也不要硬进入 Phase 2。

**最多允许延期到 10 月 15 日。**

这就是保守计划里的 buffer。

---

# 八、Phase 1 完成以后，你距离最终目标还有什么？

此时你已经具备：

```text
深度学习基础
RNN
GRU
Tensor
PyTorch
模型结构理解
模型重实现
数值验证
C/Python 对齐
```

但还没有：

```text
ONNX
Inference Runtime
Quantization
C++ deployment
LLM
GPU
vLLM
CUDA
TensorRT
Distributed
```

所以：

> **Phase 1 只是证明“你能进入 AI 模型内部”。**

还不能把自己包装成 AI Inference Engineer。

---

# 九、Phase 2：10/13 ～ 11/10

这个阶段非常重要。

因为它开始把你的：

> “学习项目”

变成：

> **“工程项目”。**

最终：

```text
PyTorch
 ↓
ONNX
 ↓
ONNX Runtime
 ↓
INT8
 ↓
C++
 ↓
benchmark
```

---

## 第一周 10/13～10/19

学习：

```text
model.eval()
torch.no_grad()
state_dict
ONNX
ONNX graph
operator
```

完成：

```text
RNNoise PyTorch
        ↓
       ONNX
```

环境：

**Mac / WSL2 均可。**

---

## 第二周 10/20～10/26

学习：

```text
ONNX Runtime
Execution Provider
C++ API
```

完成：

```text
ONNX
 ↓
ORT Python
 ↓
ORT C++
```

环境：

**本机完全可以。**

ORT 有 CPU Execution Provider，因此现在不需要 GPU。([ONNX Runtime][1])

---

## 第三周 10/27～11/2

学习：

```text
FP32
FP16
INT8
scale
zero point
dynamic quantization
static quantization
calibration
```

先做：

```text
FP32
vs
INT8
```

环境：

**CPU 足够。**

---

## 第四周 11/3～11/10

完成 benchmark：

```text
Latency
p50
p95
CPU
Memory
Model size
Output error
```

最终：

# RNNoise C++ AI Inference Demo

这是真正的第一个就业项目。

---

# 十、Phase 2 结束以后，可以开始明显增加 AI 投递

这时候你的目标岗位：

```text
模型部署工程师
AI Model Deployment
AI Inference Service
AI Backend
AI Platform Backend
C++ AI Backend
ONNX Runtime
AI Serving
音频 AI Deployment
```

这里不要等 vLLM。

因为你已经有：

```text
PyTorch
ONNX
ORT
INT8
C++
Inference
Benchmark
```

再加你已有十多年 C++ 高并发/实时系统经验，本身已经可以开始测试市场。

你的历史经历里，腾讯云部分已经有 60w+ PCU、C++ 协程化、Redis/RPC 异步化、单核心使用率降低超过 40%、无锁队列等生产级性能工程证据。

这正好可以和 Phase 2 拼起来：

```text
过去：
C++ 高性能系统

现在：
AI Model Inference

组合：
C++ AI Inference / AI Performance
```

---

# 十一、Phase 3：11/11 ～ 12/22

这是你从：

> AI Model Engineer

向：

> AI Inference Engineer

转变的关键阶段。

---

## 第一部分：Transformer

### 11/11～11/17

学习：

```text
Embedding
Attention
Q
K
V
Self Attention
Multi Head Attention
```

环境：

**CPU 完全可以。**

---

## 11/18～11/24

学习：

```text
Transformer Block
MLP
LayerNorm
Residual
Position Encoding
```

仍然：

**CPU 完全可以。**

---

## 11/25～12/1

学习：

```text
Prefill
Decode
KV Cache
TTFT
TBT/TPOT
```

这是非常关键的。

环境：

**CPU 可以理解，实际 LLM 性能实验有限。**

---

# 十二、12/2 开始 vLLM

### 12/2～12/8

在 WSL2：

```text
vLLM CPU
```

跑小模型。

vLLM 当前官方支持 x86 CPU 的基本 inference/serving，但需要 Linux；所以这里推荐你**不要在 Intel Mac 上折腾，直接使用 Windows + WSL2**。([vLLM][2])

注意：

> **不要拿 CPU vLLM 的性能数字跟网上的 GPU benchmark 比。**

我们这里只是为了：

```text
request
↓
scheduler
↓
batch
↓
model
↓
response
```

建立认知。

---

## 12/9～12/15

学习：

```text
Continuous Batching
Scheduler
PagedAttention
KV Cache
```

---

## 12/16～12/22

完成：

```text
vLLM
+
C++ Gateway
+
Concurrency
+
Benchmark
```

你的 C++ 老本开始真正进入 AI Serving。

---

# 十三、Phase 3 结束以后

可以投：

```text
大模型推理工程师
Inference Service Engineer
Model Serving Engineer
AI Platform
LLM Infrastructure
vLLM
C++ AI Infra
AI Serving
```

但不要只投“纯算法/训练”。

---

# 十四、Phase 4：12/23～2/2

这里是最大的设备分界点。

目标：

```text
CUDA
GPU
Nsight
TensorRT
```

但是：

# **不要求你现在拥有 GPU。**

我们把 Phase 4 拆成两层。

---

## Phase 4-A：本机理论 + CPU 代码

12/23～1/7：

学习：

```text
GPU architecture
Thread
Block
Warp
SM
Register
Shared Memory
Global Memory
Memory Coalescing
Occupancy
Stream
Event
```

可以：

**本机完成。**

你甚至可以手写一些 CUDA 思维对应的 CPU benchmark、伪 kernel、矩阵计算。

但不能声称完成真实 GPU 优化。

---

# 十五、Phase 4-B：第一次租 GPU

1/8～1/21：

开始：

```text
CUDA
↓
GPU kernel
↓
Nsight Systems
↓
Nsight Compute
```

这一阶段需要：

# NVIDIA GPU

你的两台电脑如果没有 NVIDIA GPU，就不要硬搞。

直接使用：

> 云 GPU / 远程 Linux GPU 机器。

而且不需要全天使用。

真正需要 GPU 的时候：

```text
租几个小时
完成实验
保存代码
保存 profiling 数据
关闭机器
```

对你现在 gap 的经济状态，比买一张昂贵 GPU 更合理。

---

# 十六、1/22～2/2

学习：

```text
TensorRT
FP16
INT8
FP8
Operator Fusion
```

重新把 RNNoise：

```text
ONNX
↓
TensorRT
↓
FP16 / INT8
```

跑一次。

这样你的 Phase 2 项目又被升级：

```text
PyTorch
 ↓
ONNX
 ↓
ORT
 ↓
TensorRT
 ↓
C++
 ↓
Benchmark
```

这个项目就开始非常像一个真正的 **Model Deployment / Inference Optimization** 项目了。

TensorRT Execution Provider 本身就是 NVIDIA GPU 推理路径，因此这一部分确实不能在你的纯 CPU 设备上做真实 benchmark。([ONNX Runtime][6])

---

# 十七、Phase 4 结束以后可以投：

```text
Inference Performance Engineer
Model Acceleration Engineer
AI Performance Engineer
TensorRT Engineer
CUDA Inference
GPU Optimization
模型加速工程师
推理优化工程师
```

这里是你非常重要的第二个职业转折点。

---

# 十八、Phase 5：2/3～3/30

现在才进入：

# Distributed AI Infrastructure

完整保留之前说过的：

```text
Multi-GPU
Tensor Parallel
Pipeline Parallel
NCCL
NVLink
RDMA
GPU topology
Kubernetes
GPU scheduling
KV Cache management
PD Disaggregation
```

---

# 十九、但是这里我要非常现实地告诉你

你在家里**无法完整实现 Phase 5 的真实环境**。

尤其：

```text
2 GPU
4 GPU
8 GPU

+

NVLink

+

RDMA NIC

+

InfiniBand / RoCE

+

多节点
```

不是一台普通家用电脑可以模拟出来的。

NCCL 本身是围绕多 GPU collective communication 设计的，并且其 GPU Direct / RDMA 路径需要真实 GPU、NIC、驱动和网络拓扑。([NVIDIA Docs][5])

所以这阶段我们采用：

# 70% 本机学习 + 30% 远程实验

---

# 二十、2/3～2/16

本机：

```text
Tensor Parallel
Pipeline Parallel
Data Parallel
NCCL API
```

先把原理搞懂。

---

# 二十一、2/17～2/28

远程 GPU：

```text
2 GPU
NCCL
AllReduce
AllGather
ReduceScatter
```

做简单实验。

不是为了跑大型模型。

只是：

> 看通信。

---

# 二十二、3/1～3/10

学习：

```text
GPU topology
PCIe
NVLink
NUMA
GPU ↔ NIC
```

这里你原来的：

```text
CPU NUMA
cache
memory
PCIe
网络
```

经验会开始重新变得非常有价值。

---

# 二十三、3/11～3/20

学习：

```text
RDMA
InfiniBand
RoCE
GPUDirect RDMA
```

注意：

### 不要求你在家真实部署 RDMA。

你需要达到：

> 知道 RDMA 为什么存在，知道 CPU networking 与 RDMA/GDRDMA 的数据路径差别，知道 NIC/GPU/NUMA topology 为什么影响 AI inference。

真正的 RDMA benchmark 用远程机器完成。

---

# 二十四、3/21～3/30

学习：

```text
GPU scheduling
KV Cache management
PD Disaggregation
Kubernetes GPU scheduling
```

最终形成：

```text
Request
↓
Gateway
↓
Scheduler
↓
GPU
↓
KV Cache
↓
Multi GPU
↓
Network
↓
RDMA
```

完整 AI Inference 系统图。

---

# 二十五、3 月底，你的最终目标是什么？

不是：

> “我已经有 5 年 AI Infra 经验。”

绝对不能这么包装。

而是：

> **我具备从模型理解、模型重实现、模型部署、量化、C++ inference、LLM serving、GPU profiling，到分布式推理基础的完整技术链，并且把这些能力建立在多年高性能 C++ / 实时系统 / 分布式系统生产经验之上。**

这是一个可信的技术定位。

---

# 二十六、你最终可以把自己定位成什么？

到 Phase 3：

> **AI Inference / Model Serving Engineer**

到 Phase 4：

> **AI Inference Performance / Model Acceleration Engineer**

到 Phase 5：

> **AI Systems / AI Infrastructure Engineer**

这三个其实是一条成长链。

---

# 二十七、你的 RNNoise 项目最后应该演化成这个东西

这才是整个路线最重要的一点。

开始的时候：

```text
RNNoise
C code
```

现在：

```text
RNNoise
C
↓
PyTorch
↓
C/Python numerical alignment
```

Phase 2：

```text
RNNoise
↓
PyTorch
↓
ONNX
↓
ONNX Runtime
↓
INT8
↓
C++
```

Phase 4：

```text
RNNoise
↓
ONNX
↓
TensorRT
↓
FP16 / INT8
↓
GPU
```

最终：

```text
Realtime Audio
      ↓
AI Model
      ↓
Inference Runtime
      ↓
Quantization
      ↓
C++
      ↓
Latency / CPU / Memory
      ↓
GPU
      ↓
Performance
```

于是你一开始认为只是一个：

> “RNNoise 临时学习项目”

实际上变成了：

# **你第一次完整走通 AI Model → AI Runtime → AI Performance 的纵向项目。**

这就是我要你保留下来的核心项目。

---

# 二十八、就业问题必须同时进行

你现在 gap，不能按照：

```text
9月～3月
学习
学习
学习
学习
学习

4月
开始找工作
```

绝对不建议。

你的投递路线应该是：

```text
现在
│
├── 传统 C++ / 高性能 / 音视频：持续投
│
├── 10月中旬 Phase 2：开始 AI Deployment
│
├── 11月 Phase 2完成：大量投 AI Inference/Serving
│
├── 12月 Phase 3：增加 LLM Inference
│
├── 1月 Phase 4：增加 CUDA/TensorRT
│
└── 2～3月 Phase 5：增加 AI Infra
```

这样你的学习并不是脱离就业。

而是：

> **每完成一个阶段，就扩大一次岗位池。**

---

# 二十九、你现在实际上有两个技术杠杆

你的第一杠杆：

```text
多年 C++
+
Linux
+
高并发
+
实时音视频
+
分布式
+
性能
```

腾讯云部分已经有非常强的生产规模证据，包括 60w+ PCU、99.95% 可用性、协程化、Redis/RPC 异步、40%+ 单核心使用率优化、无锁队列。

你的第二个杠杆正在建立：

```text
AI Model
+
Inference
+
Quantization
+
Runtime
+
GPU
```

两个东西叠加之后，才是：

# **AI Systems Engineer**

而不是：

```text
老 C++ 程序员
+
学了一些 Python
+
会调用 LLM
```

这两种职业画像差别非常大。

---

# 三十、最后，我给你一个非常重要的“设备策略”

目前：

### 不买昂贵 GPU

现在完全没有必要。

### 不折腾 RDMA 硬件

没有必要。

### 不买多 GPU

更没有必要。

### 现在主要使用

```text
Mac x86
   ↓
PyTorch / C / RNNoise

Windows + WSL2
   ↓
Linux
ONNX Runtime
vLLM CPU
C++
```

### 到 GPU 阶段

再临时获得：

```text
Linux
+
NVIDIA GPU
```

### 到 Distributed AI 阶段

再获得：

```text
2～4 GPU
+
NCCL
```

### 到 RDMA 阶段

再获得：

```text
GPU
+
RDMA NIC
+
RoCE / InfiniBand
```

也就是说：

> **设备跟着知识阶段走，而不是为了假想的未来需求提前投资设备。**

对于你现在处于 gap、目标是就业的现实情况，这个约束非常重要。

---

# 三十一、最终时间表，我建议你就固定下来，不再频繁改

```text
2026/09/29
      │
      │ 当前：GRU 完成
      ↓
2026/10/12
      │
      │ Phase 1
      │ RNNoise Neural Network
      ↓
2026/11/10
      │
      │ Phase 2
      │ ONNX / ORT / INT8 / C++
      │
      │ ★ 第一批 AI 岗位投递
      ↓
2026/12/22
      │
      │ Phase 3
      │ Transformer / KV Cache / vLLM
      │
      │ ★ AI Inference / Serving 投递
      ↓
2027/02/02
      │
      │ Phase 4
      │ CUDA / GPU / TensorRT / Nsight
      │
      │ ★ Inference Performance / Model Acceleration
      ↓
2027/03/30
      │
      │ Phase 5
      │ Multi-GPU / NCCL / RDMA / AI Infra
      │
      │ ★ AI Systems / AI Infra
      ↓
2027/03/31+
      │
      │ Phase 6 可选
      │ Training Infra
      │ Megatron / DeepSpeed / FSDP
```

---

**你现在的学习方式已经验证是有效的。**

尤其今天你做的：

```text
C GRU
↓
Python GRU
↓
C / Python output comparison
↓
发现 sigmoid / tanh implementation difference
↓
自己模拟 C 的低精度行为
↓
分析 state error propagation
```

这个过程其实比“把 D2L GRU 看懂了”重要得多。

因为你已经在做真正的 **numerical debugging / model implementation equivalence**。

所以现在不要再为了“赶进度”跳到 vLLM。

**你现在正确的动作是把这个 GRU 继续串回 RNNoise，10 月上旬完成 Phase 1，然后立刻进入 ONNX Runtime + INT8。**

而且由于你的目标是就业，我会把 **10 月中旬作为第一个真正的 AI 求职节点**，而不是等到 CUDA、vLLM、NCCL 全学完以后再投。

这条路线最大的现实优势就在于：

> **你不需要等自己变成完整 AI Infra 工程师才开始找工作。每一阶段都会产生一个更强的候选人版本。**

[1]: https://onnxruntime.ai/docs/execution-providers/?utm_source=chatgpt.com "Execution Providers | onnxruntime"
[2]: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/?utm_source=chatgpt.com "CPU - vLLM"
[3]: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/?utm_source=chatgpt.com "GPU - vLLM"
[4]: https://docs.nvidia.com/cuda/wsl-user-guide/?utm_source=chatgpt.com "CUDA on WSL User Guide — CUDA on WSL 13.4 documentation"
[5]: https://docs.nvidia.com/deeplearning/nccl/?utm_source=chatgpt.com "NVIDIA Deep Learning NCCL Documentation"
[6]: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html?utm_source=chatgpt.com "NVIDIA - TensorRT | onnxruntime"

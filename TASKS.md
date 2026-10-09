# RNNoise → ONNX 部署 · 总任务与进度看板

> 最后更新：2026-10-07（Phase 2 第 1 周进行中）
> 本文件是**项目总任务与进度看板**：目标 → 阶段路线 → 当前阶段拆解 → 验收 → 资产 → 附录（原始三任务）。
> 学习细节看 `doc/` 下各阶段文档；项目常驻指令看 `CODEBUDDY.md`。

---

## 0. 文档分工（约定）

每个阶段固定产出三类文件，放在 `doc/`：

| 类型 | 作用 | 示例（Phase 1 / Phase 2） |
| --- | --- | --- |
| **学习路径文件** | 阶段范围、周计划、掌握标准、验收项 | `Phase 1 学习路径.md` / `phase 2 学习路径安排.md` |
| **输出文件** | 我自己的学习产出、结论、验收记录 | `phase 1 学习输出.md` / `phase 2 学习笔记 输出.md` |
| **临时/细节文件** | 具体问题、报错、讨论、临时方案 | `RNN_UNIT整网误差分析*.md` / `phase 2 onnx export 学习步骤.md` |

**总进度**由 `doc/全流程 学习进度 每日跟进严格管理.md` 管理（Phase 1~6 时间表、设备结论、求职节点）。

---

## 1. 项目全景：一条纵向链路

```
C RNNoise 源码
   ↓  读懂网络结构 / 算子 / state
PyTorch 重实现（数值对齐 C）
   ↓  export
ONNX（独立于 PyTorch 的计算图）
   ↓  Runtime
ONNX Runtime（CPU EP）
   ↓  Quantization
INT8（精度 vs 速度）
   ↓  C++ 推理
Benchmark（latency / CPU / memory / model size / output error）
```

这是"任务1 → 任务2 → 任务3"的顺序依赖链（原始定义见 **附录 A**）。
RNNoise 只是第一个训练场，最终职业目标是 **AI Inference / AI Infra**。

---

## 2. 阶段路线图

| 阶段 | 时间 | 主题 | 状态 |
| --- | --- | --- | --- |
| Phase 1 | 9/29 ~ 10/12 | RNNoise 神经网络（读懂 + 重实现 + 数值验证） | ✅ 已完成 |
| **Phase 2** | **10/4 ~ 11/10** | **ONNX / ORT / INT8 / C++ / Benchmark** | **🔄 进行中** |
| Phase 3 | 11/11 ~ 12/22 | Transformer / KV Cache / vLLM | ⬜ 未开始 |
| Phase 4 | 12/23 ~ 2027/2/2 | CUDA / GPU / TensorRT / Profiling | ⬜ 未开始 |
| Phase 5 | 2/3 ~ 3/30 | 分布式推理 / NCCL / RDMA / AI Infra | ⬜ 未开始 |
| Phase 6 | 3/31+ | Training Infra（可选扩展） | ⬜ 未开始 |

**学习模式（Phase 2 起升级）**：
> 理解一个 deployment concept → 在 RNNoise 上使用 → 观察运行时行为 → benchmark → 能解释结果。

进度不看百分比，看**验收项**：每一项都要"我能解释 + 我写过 + 我验证过 + 我能排错"。

---

## 3. Phase 1 已完成（验收记录）

**目标**：完成 RNNoise neural network 的理解 + PyTorch 重实现 + C/Python 数值验证。

已交付：

- [x] 整网计算图（层顺序、每层输入/输出 shape、state）—— `conv1 → conv2 → gru1 → gru2 → gru3 → dense_out / vad_dense`
- [x] C 变量 ↔ 数学符号 ↔ PyTorch Tensor 映射表
- [x] C/Python 逐层数值对比报告（`Conv1/Conv2/GRU1/GRU2/GRU3/Output` 误差）
- [x] 误差分析：权重布局 / bias / 激活 / 浮点精度 / state 更新顺序
- [x] `compute_linear` / `compute_generic_conv1d` / `compute_generic_gru` 三个算子的理解
- [x] 保留实验：FP32 vs 低精度 sigmoid/tanh、state drift
- [x] 代码：`examples/rnn_unit.py` 拆分为 `linear.py` / `gru.py` / `loader.py` / `utils.py` / `rnnoise_scratch.py`

相关文档：`doc/Phase 1 学习路径.md`、`doc/phase 1 学习输出.md`、`doc/RNN_UNIT整网误差分析*.md`。

---

## 4. Phase 2（当前阶段）：ONNX / ORT / INT8 / C++

### 4.0 总目标

> 把 **RNNoise PyTorch 模型**变成一个**独立于 PyTorch 的可部署模型**，在 **CPU 上正确推理**，做出**可信的 FP32 vs INT8 benchmark**，并用 **C++** 跑通。

最终产物：`RNNoise C++ AI inference demo`（可上简历的第一个 AI 工程项目）。

### 4.1 四周计划与进度

| 周 | 时间 | 主题 | 状态 |
| --- | --- | --- | --- |
| W1 | 10/4 ~ 10/10 | PyTorch → ONNX（export contract / graph / operator） | 🔄 **进行中** |
| W2 | 10/11 ~ 10/17 | ONNX Runtime（CPU EP / Session）+ C++ API | ⬜ |
| W3 | 10/18 ~ 10/24 | Quantization（FP32/FP16/INT8、scale/zero_point、dynamic/static） | ⬜ |
| W4 | 10/25 ~ 10/31 | Benchmark + C++ 完整 Demo + README | ⬜ |

### 4.2 W1 进行中：PyTorch → ONNX

**已掌握（学习笔记已记录）**：

- [x] `eval()` vs `no_grad()`：模型行为 vs autograd 计算图，两者正交
- [x] `requires_grad=True` ≠ `nn.Parameter`（前者影响 autograd，后者进 `state_dict`）
- [x] `nn.Module` 的注册机制：`nn.Parameter` / `register_buffer` → `state_dict()` / `.to()` / `.eval()` / optimizer / exporter
- [x] `state_dict` = 参数 + buffer；模型 = 结构（class）+ state_dict
- [x] ONNX 是静态计算图；`torch.onnx.export` 跑一遍 forward，只记「张量→张量」的运算
- [x] 跨调用延续的 state（conv mem / RNN hidden / KV cache）必须作为 graph input/output **成对出现**

笔记：`doc/phase 2 学习笔记 输出.md`。

**正在做（当前焦点 = Export Contract）**：

- [ ] 搞清 `torch.onnx.export` 对一个 `model` 要求什么（**Export Contract**）
- [ ] 用小 toy 模型验证：多输入 / 多输出 / **显式 state 输入输出**
- [ ] `torch.export` 路径：`dynamo=True`（PyTorch 2.9+ 默认）、`dynamic_shapes`、`verify` / `report`
- [ ] 回到 RNNoise，把 `RNNoiseMo` 改造成可导出的 `nn.Module`
- [ ] 导出 `rnnoise.onnx` → `onnx.checker` → Netron 看图 → 列 graph inputs/outputs/node/operator
- [ ] **PyTorch output ≈ ONNX output**（这一步不能省）

细节文档：`doc/phase 2 onnx export 学习步骤.md`。

### 4.3 导出学习阶梯（严格按序，避免"什么都懂一点"）

```
Step 1  x → Linear → Tanh → y                  # 最简单：PyTorch→ONNX→ORT 打通
Step 2  (x, h) → Linear(x)+h → y               # 多输入
Step 3  → (y, h_new)                            # 多输出
Step 4  (x_t, h_t) → GRU cell → (y_t, h_(t+1)) # 显式 recurrent state（最接近 RNNoise）
Step 5  Conv → Conv → GRU×3 → Dense            # 小型 RNNoise
Step 6  完整 RNNoise：C → PyTorch → ONNX → ORT  # 真实项目
```

原则：**小模型先验证框架概念，真实项目再验证工程问题。**
遇到报错不要"报错→Google→改→再报错"，而是：
```
报错 → 判断 exporter 处于哪个阶段 → 归类到概念 → 查官方文档 → 最小 toy 实验验证 → 再回 RNNoise
```

### 4.4 `RNNoiseMo` 导出阻塞点（按概念归类，不逐个盲修）

早期逐条记过 12 项，现已解决大半：`nn.Module` ✓、子模块注册 ✓、权重 `register_buffer` ✓、
`forward` 无打印 ✓、状态参数化（`Conv1D` / `GRUMo` 都是 `out/state` 进出）✓。
**剩下的按概念顺序推**：

```
Model Contract → Pure Tensor Forward → Explicit State
→ Input/Output Contract → torch.export / ONNX Exporter → ONNX Graph → ORT
```

当前唯一硬阻塞：`RNNoiseMo` 仍把 5 个状态（conv1/conv2 的 mem + 三层 gru 的 hidden）
存在实例属性里，必须改成 `forward` 的输入/输出。

> 核心认知：RNNoise 是**状态机式实时模型**，ONNX 偏向**显式输入 → 显式计算图 → 显式输出**，二者之间存在 **Representation Gap**——这才是真正的技术含量。

### 4.5 代码资产（`examples/`）

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `linear.py` / `gru.py` / `rnnoise_scratch.py` | `Conv1D` / `DenseLayer` / `GRUMo` / 整网 | ✅ Phase 1 产物 |
| `loader.py` / `utils.py` | 用例加载 / `emit` 协议输出 | ✅ |
| `onnx_demo.py` | PyTorch → ONNX 练习（单层 conv1） | 🔄 进行中 |
| `toy.py` | 最小模型：导出 + 读回（可独立跑） | 🔄 进行中（`toy.onnx`） |
| `ort_demo.py` | ONNX → ORT 推理练习 | 🔄 进行中 |

用例数据来自 `scripts/unit_cases/*.json`（真实模型规模 //16）。
**文件级完整清单在 `doc/学习工具搭建手册.md` §4.3**（此处不重复维护）。

### 4.6 Phase 2 验收（结束时闭卷自测 20 问）

- **PyTorch**：`eval()` vs `no_grad()`；`state_dict` 存什么；为何加载前要先建模型
- **ONNX**：ONNX 是什么；Graph/Node/Operator 区别；Initializer；为何 `Linear` 未必变成 "Linear node"
- **ORT**：ORT 与 ONNX 关系；Session；Execution Provider；CPU EP 做什么；graph optimization 为何提升性能
- **Quantization**：INT8 为何要 scale；zero_point；dynamic vs static；calibration；INT8 为何可能更慢；为何 INT8 与 FP32 输出不完全一致
- **Engineering**：benchmark 为何要 warm-up；为何 Session 创建时间不算 inference latency

外加**知识验收表**（`eval/no_grad/state_dict/ONNX Graph/Operator/export/ORT Session/EP/ORT C++/graph optimization/INT8/dynamic/static/FP32-INT8/Benchmark`）逐项"能解释 + 写过 + 验证过 + 能排错"。

详见 `doc/phase 2 学习路径安排.md` 第六、十节。

---

## 5. 后续阶段（摘要，详见总进度文档）

- **Phase 3**：Transformer / Attention / KV Cache / Prefill / Decode / Batching → vLLM（WSL2，CPU）；产出 `vLLM + C++ Gateway + Benchmark`。
- **Phase 4**：CUDA / GPU 架构 / Nsight / TensorRT；RNNoise 升级为 `ONNX → TensorRT → FP16/INT8`；**GPU 用云租，不买**。
- **Phase 5**：Multi-GPU / Tensor/Pipeline Parallel / NCCL / NVLink / RDMA / GPU scheduling；70% 本机 + 30% 远程。
- **Phase 6**（可选）：Training Infra（Megatron / DeepSpeed / FSDP）。

---

## 附录 A：原始三任务定义（项目背景）

1. **任务1（重建网络、提取权重、导出 ONNX）** —— 基础：得到"可移植"的 ONNX 模型文件。
   用 PyTorch 重实现 RNNoise 网络结构 → 解析 C 源码中的权重数组并加载到对应层 → 验证前向结果与 C 一致 → `torch.onnx.export` 导出，`onnx.checker` 校验通过。
2. **任务2（ORT CPU 推理，对比原版性能）** —— 验证：用 ONNX Runtime（Python/C++）加载模型，同输入下对比 ONNX 与原 C 实现的**输出精度**与**运行速度**。
3. **任务3（INT8 量化实验）** —— 优化：FP32 → INT8（dynamic / static + calibration），评估**精度损失**与**性能提升**（速度、模型大小）。

三者顺序依赖：任务1 → 任务2 → 任务3。没有任务1 的 ONNX 就没有任务2/3；任务2 保证转换正确，是任务3 的前提。

---

## 附录 B：每日跟进

每日进度、blocker、下一步写在 `doc/全流程 学习进度 每日跟进严格管理.md`（含时间表与求职节点）。
上一条关键节点：Phase 1 完成（git `357598d phase1完成`）。

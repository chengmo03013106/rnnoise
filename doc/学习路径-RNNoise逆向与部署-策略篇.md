# 学习路径：RNNoise 逆向与部署（策略篇）

> 读者画像：多年 C++ 系统编程（TRTC 实时音视频后台，epoll/多线程/性能分析）；ML 只学过线性回归、MLP、CNN；
> **RNN/GRU 为 0 基础**；PyTorch 只跑过 MNIST/CIFAR 级教程；**ONNX / ONNX Runtime 完全没用过**；量化 0 基础；**纯 CPU 无 GPU**。
>
> 目标：达到「模型部署 + 量化」方向**能到岗干活**的水平，不是"会用"。
>
> **教学原则（本文严格遵守）**：
> 1. 不直接给可背诵的答案。每个结论要么你自己推导，要么由脚本/实验判定。
> 2. 以"可迁移"为验收：做完之后，给你任意一个神经网络的 C 实现，你能独立完成逆向 + 部署。
> 3. 不预设任何 RNN/GRU/ONNX/量化的先验。
> 4. 每一阶段都有：输入（读什么）→ 动作（写什么）→ **验收（怎么自证）** → 预估耗时。

---

## 0. 引用约定与环境事实（先读这一节，能省你两天）

### 0.1 路径约定

| 引用写法 | 实际位置 | 原因 |
|---|---|---|
| `src/rnnoise_data.h` / `src/rnnoise_data.c` | 本工作区在**仓库根目录** `rnnoise_data.h` / `rnnoise_data.c` | 这两个文件不在 Git 里（78 MB），由 `download_model.sh:30` 的 `tar xvomf` 解出来。按 `README:101` 应拷贝到 `src/`。根目录副本与 `src/` 副本是同一文件，行号一致 |
| 其余 `src/xxx.c` | 就在 `src/` | — |

**文档指代约定**（本文提到"深度文档"处均指向同一个文件，避免歧义）：

- 「深度文档」 = `doc/RNNoise 神经网络结构与权重数组详解.md`（§1.4 特征值域 / §2.3 GRU 专题 / §3.2 权重数组清单 / §3.4 int8 量化布局 都在这一份）
- 「项目解析」 = `doc/rnnoise 项目解析.md`

### 0.2 三个会浪费你时间的环境陷阱（已实测确认）

**陷阱 1：仓库根目录的 `torch/` 会遮蔽真正的 `torch` 包。**

```console
$ cd /Users/chengmo/Work/rnnoise && python3 -c "import torch,sys;print(torch.__path__)"
_NamespacePath(['/Users/chengmo/Work/rnnoise/torch'])     # ← 导入的是仓库里的目录，不是 PyTorch
```

`sys.path[0]` 在 `python3 -c` 时是当前目录。所以**永远不要在仓库根目录跑 Python**。
正确做法：工作脚本放在仓库外（如 `~/rnnoise-lab/`），或者始终把 cwd 放在 `torch/rnnoise` 之外的独立目录。
（`torch/rnnoise/dump_rnnoise_weights.py:35` 用 `sys.path.append('../weight-exchange')`，它设计成从 `torch/rnnoise` 目录运行。）

**陷阱 2：本机当前没有装 torch / numpy / onnx / onnxruntime。** 第 0 阶段第一件事就是装环境。

**陷阱 3：autotools 默认构建走 int8，不是 float。**
`configure.ac:82-88` 里 `AC_ARG_ENABLE([dnn-debug-float], ..., enable_dnn_debug_float=no)`，第 4 参数"用户未指定时的默认值"是 `no`，于是 `AS_IF` 成立 → **默认就定义 `DISABLE_DEBUG_FLOAT`**。
后果：`src/nnet_arch.h:138-143` 的分派会走 `sparse_cgemv8x4` / `cgemv8x4`（int8 内核），而不是 float 内核。
**你拿默认构建去对比你的 PyTorch FP32 模型，会看到 1e-2 量级的误差，那是量化导致的，不是你写错了。** 阶段 0 必须先解决这个。

### 0.3 面向 C++ 工程师的 ML 概念直觉表

先建立直觉，再落代码。本文后面每个 ML 概念都可以回查这张表。

| ML 概念 | C++ 直觉映射 |
|---|---|
| 张量（tensor） | 一块连续内存 + 一份 shape 元数据。shape 只是描述 stride 的约定，改 shape 不改字节 |
| 权重矩阵 `W (out, in)` | 一个二维数组。C 按 **(in, out) 行主序** 存（`w[in*nb_outputs + out]`），正好是 PyTorch `(out, in)` 的转置 |
| 层（Layer） | 一个 struct：参数指针 + `forward()` 方法。RNNoise 的 `LinearLayer`（`src/nnet.h:65-75`）就是 |
| 激活函数 | 逐元素函数。这里不是精确 `tanh`，是 `src/vec.h:337-356` 的有理多项式近似 |
| 隐藏状态 `h` | **跨调用存活的成员变量**。等价于你把 `static` 局部变量显式提到 `DenoiseState` 里 |
| `batch` / `seq_len` | 两个额外的数组维度。C 里这两个维度被"展开成一个 for 循环"，所以代码里看不见 |
| Conv1d k=3 | **不是卷积**。是"把 3 帧拼成一个长向量 + 一次大矩阵乘"，见 `src/nnet.c:113-123` |
| ONNX 图 | 一个序列化的计算 DAG。`initializer` ≈ 编译进 `.rodata` 的常量数组；`input` ≈ 函数形参 |
| 量化 scale | 定点化的 Q 值。这里是**逐输出神经元一个 Q 值**（`{name}_scale`，长度 = `nb_outputs`） |
| 校准（calibration） | 拿真实输入跑一遍，统计每层激活的动态范围，据此定 Q 值 |
| 8×4 稀疏块 | 手工 SIMD 的产物：一次 `vdotq_s32` / `dpbusds` 吃 4 个输入 × 8 个输出。**布局由指令集决定，不是数学决定** |

---

## 1. 知识缺陷诊断

格式：**缺陷 → 证据 → 在做任务 X 时会犯的具体错误（代码级）→ 会在哪一步暴露**。

### 缺陷 1：RNN/GRU 完全 0 基础 —— 最大缺口，占整网 92% 计算量

**证据**：3 层 GRU 的 `input + recurrent` 共 `3 × 2 × 384 × 1152 = 2,654,208` MAC，占全模型 2,877,312 MAC 的 **92.2%**。

**会犯的具体错误**：

| # | 错误 | 代码级后果 |
|---|---|---|
| 1a | 以为"GRU 就是一个层"，用 `nn.GRU(384, 384, num_layers=3)` 一把梭 | C 是**三个独立 GRU 串联**，且输出层吃的是 `cat([conv2_out, gru1_out, gru2_out, gru3_out])`（`src/rnn.c:53-55`、`torch/rnnoise/rnnoise.py:106`），不是只取最后一层。`num_layers=3` 语义完全不同 |
| 1b | 调用 `self.gru(x)` 不传 `h0` | `torch/rnnoise/rnnoise.py:90-93` 里 `states is None` 会造零状态 → **每帧从零记忆开始**，输出与 C 对不上，且"语音起音那段"错得最厉害 |
| 1c | 不知道 `hidden state` 是"跨帧传递的" | 在 ONNX 里把 `h0` 变成 initializer（常量）→ 模型变无状态，流式推理完全失效 |
| 1d | 门顺序搞反 | PyTorch `weight_ih_l0` 是 **r, z, n**；C 的 `zrh[]` 是 **z, r, h**（`src/nnet.c:77-79`）。导出脚本在 `wexchange/c_export/common.py:346-350` 做了交换。**你自己装配权重时不交换，前几帧看着还行，几十帧后发散** |
| 1e | 不知道 `r` 乘在哪 | C 的 `src/nnet.c:87`：`h[i] += recur[2*N+i]*r[i]` —— `r` **只乘在候选状态的递归项上**。写成 `tanh(W_in·x + r·W_hn·h)` 或 `tanh(r·(W_in·x + W_hn·h))` 都是错的 |
| 1f | 忘记 `diag` | `gruN_recurrent` 的 `W_hh` 对角被抽出来单独存（`common.py:108-124` 的 `extract_diagonal`），C 在 `src/nnet_arch.h:153-161` 补回。你不补，`W_hh` 缺了 1152 个自环元素 |

**暴露时机**：阶段 2 的双向闭环（`dump_rnnoise_weights.py` 正向导出逐字节比对）会立刻抓住 1d/1e/1f；1a/1b/1c 在阶段 3 的逐帧比对时暴露。

### 缺陷 2：PyTorch 不熟练 —— 会导致你破坏唯一可用的验证闭环

**会犯的具体错误**：

| # | 错误 | 代码级后果 |
|---|---|---|
| 2a | 不知道 `named_modules()` 的语义 | `torch/rnnoise/dump_rnnoise_weights.py:60-91` 遍历 `named_modules()`，用**模块名字**拼 C 数组名（`{name}_bias` / `{name}_weights_float` …，规则见 `common.py:221-227`）。你给子模块改个名（比如 `self.gru_1`），导出的数组全变成 `gru_1_input_*`，`init_rnnoise()` 找不到 → **闭环断裂**。这是本项目最重要的硬约束，**写在阶段 4 的约束里** |
| 2b | 不懂 `load_state_dict(strict=False)` | `dump_rnnoise_weights.py:43` 用的就是 `strict=False`。缺 key / 多 key **不报错**，静默跳过。你装错了权重名，脚本不会告诉你 |
| 2c | 用 `.data.copy_()` 而不是 `load_state_dict` / `copy_` 的 dtype 处理不当 | 若你的 tensor 是 `float64` 而不是 `float32`，`print_vector`（`common.py:78-93`）打印出的字面量会变成 `0.10000000149011612` 这种形式 → 逐字节 diff 全篇飘红，**你会误以为权重错了** |
| 2d | 忘记 `model.eval()` / 不知道 `no_grad` | 对本模型影响不大（无 dropout/BN），但是习惯问题，面试会问 |

### 缺陷 3：ONNX / ONNX Runtime 0 基础

**会犯的具体错误**：

| # | 错误 | 代码级后果 |
|---|---|---|
| 3a | 不知道计算图是静态的、不知道 `padding='valid'` 会吃时间维 | `torch/rnnoise/rnnoise.py:66-67` 的 `nn.Conv1d(kernel_size=3, padding='valid')` 每过一层时间维 −2，两层 −4。你按本能写 `dummy_input = torch.randn(1, 1, 65)`（"一帧 65 维"）→ 导出直接报 shape 错。**T 必须 ≥ 5** |
| 3b | 分不清 initializer 和 input | 权重会被固化成常量（好事），但如果 GRU 状态也被固化，模型就死了 |
| 3c | 不知道 `dynamic_axes` | 导出的模型时间维被焊死成某个 T，流式推理时帧数不匹配 |
| 3d | 性能对比口径错误（**最常见的错**） | 拿 `ORT session.run()` 的时间去比 `rnnoise_process_frame()`。`rnnoise_process_frame()`（`src/denoise.c:457-504`）包含高通滤波、960 点 FFT、基音搜索、梳状后滤波、IFFT、OLA —— **网络只占其中一小部分**。比出来"ONNX 快 10 倍"是毫无意义的结论。正确基准是 `compute_rnn()`（`src/rnn.c:44-60`） |

### 缺陷 4：量化 0 基础

**会犯的具体错误**：

| # | 错误 | 代码级后果 |
|---|---|---|
| 4a | 不知道 per-tensor / per-channel / per-output-neuron 的区别 | RNNoise 用的是**逐输出神经元**（`{name}_scale` 长度 = `nb_outputs`，见 `src/parse_lpcnet_weights.c:170-172` 的长度校验）。用 per-tensor 做 GRU，误差会大一个量级 |
| 4b | 不知道"输入值域决定能不能量化" | `conv1` 的输入是 65 维原始特征，值域约 **[−12, +5]**（DCT 后第 0 维再 −12）。C 侧 int8 内核做 `x_q = round(127·x)`，隐含假设 `x ∈ [−1,1]`。**量化 conv1 会严重截断**。这就是 `torch/rnnoise/dump_rnnoise_weights.py:15` 里 `unquantized = ['conv1', 'dense_out', 'vad_dense']` 的真正理由 |
| 4c | 不知道动态量化 vs 静态量化的适用边界 | ONNX Runtime 官方明确建议：**RNN/Transformer 用 dynamic，CNN 用 static**（`quantize_dynamic` / `quantize_static` 文档 "Method selection" 节）。RNNoise 是 GRU 主导 → 先试 dynamic |
| 4d | 不知道 QDQ 与 QOperator 两种表示 | 影响量化后的可调试性和性能。ORT 默认 S8S8 + QDQ |
| 4e | 不会设计校准集 | 校准集只有静音段 → GRU 激活范围统计偏小 → 有语音时大面积饱和 |

### 缺陷 5：C++ 工程师的思维惯性（既是优势也是陷阱）

| # | 错误 | 代码级后果 |
|---|---|---|
| 5a | 低估"数据布局"的杀伤力 | **conv2 是 384×384 方阵，转置错误不会被 shape 检查拦住**（转置后还是 384×384）。你只跑 forward 看输出，误差可能只有百分之几，肉眼分辨不出；但闭环的正向导出会在 `conv2_weights_int8`、`conv2_scale`、`conv2_subias` 三个数组上全部对不上。**这就是为什么必须依赖闭环而不是"看着差不多"** |
| 5b | 拿 `subias` 当 `bias` | `subias = bias − Σ(weight_q · scale)`（`common.py:245`），是为 AVX2 无符号量化路径准备的配套偏置（`src/nnet_arch.h:144-147`）。**真正的偏置是 `bias`** |
| 5c | 相信注释和"看起来像常量"的东西 | `src/rnn.c:41` 的 `#define INPUT_SIZE 42` 是**死宏**（真实维度 65，只在 `:47` 一行被注释掉的 printf 里出现）；`src/vec.h:384` 的 `#define SCALE (128.f*127.f)` **全库 0 处使用**；`torch/rnnoise/rnnoise.py:59` 的 `gru_size=256` 是**过时默认值**（真实 384，权威值看 `rnnoise_data.h:24` 和 `train_rnnoise.py:49`）|
| 5d | 不知道 `scale` 的计算顺序有讲究 | `common.py:232` 的 `scale = compute_scaling(weight)` 发生在 `:236` 的 `print_sparse_weight`（内部才 `extract_diagonal`）**之前**。也就是 **scale 必须在抽取对角之前算**。你自己复现时顺序反了，`*_scale` 数组就对不上 |

### 缺陷 6：缺数值归因能力（最危险，会导致你"把对的改错"）

三个误差源混在一起，分不清就会瞎改：

| 误差源 | 量级（实测/可推导） | 怎么单独隔离 |
|---|---|---|
| ① 激活函数多项式近似 | **≈ 6.0e-5**（`src/vec.h:337-350` 的 `tanh_approx`，我已跑数验证：最大绝对误差 6.004e-5 @ x ≈ −5.205） | 编译一个 `HIGH_ACCURACY` 版本（`src/nnet_arch.h:77` 的注释打开）对比 |
| ② 权重量化（int8 路径） | 1e-3 ~ 1e-2 量级 | 用 `--enable-dnn-debug-float` 构建对比 |
| ③ 你的实现错误 | 未知 | 前两项扣掉之后剩下的 |

**没有这个归因能力，你会看到 1e-3 的误差就去翻自己的代码，翻半天翻不出问题，最后把正确的实现改错。** 所以阶段 0 必须先建双 baseline。

---

## 2. 分阶段学习路径

**耗时口径**：按每天有效投入 3 小时（业余）计。总计约 **22–31 天**，即 4–6 周。

### 阶段 0：建可验证的实验台（2 天）

**目标**：让"对/错"由机器判定，并**先量化出 C 自己的误差基线**。

**输入**
- `configure.ac:82-88`（`DISABLE_DEBUG_FLOAT` 为什么默认开）
- `src/nnet_arch.h:130-162`（4 条分派路径：float/int8 × 稠密/稀疏）
- `src/nnet_arch.h:77`（`/*#define HIGH_ACCURACY */` 是注释掉的）
- `src/vec.h:337-356`（`tanh_approx` / `sigmoid_approx`）
- `Makefile.am` 的 `noinst_PROGRAMS = dump_features dump_weights_blob`
- `src/dump_features.c:487-489`（每帧写 65 + 32 + 1 = 98 个 float）

**动作**
1. 装环境：`pip3 install numpy torch onnx onnxruntime`（注意 0.2 的陷阱 1，别在仓库根目录跑）。
2. **构建四个 C 变体**，放到四个目录：
   - `build-float`：默认（`DISABLE_DEBUG_FLOAT`，走 int8）
   - `build-debugfloat`：`./configure --enable-dnn-debug-float`（走 float）
   - 后两个可选：`build-float-hiacc` / `build-debugfloat-hiacc`（打开 `src/nnet_arch.h:77` 的 `HIGH_ACCURACY`）
3. **写一个 `dump_rnn_io` 小工具**（仿 `src/dump_features.c` 的写法：放在 `src/` 下、`#include "rnn.h"`、直接调 `compute_rnn()`）：
   ```
   输入：features.f32（每帧 65 个 float）
   输出：gains.f32（每帧 32 个 float）+ vad.f32（每帧 1 个 float）
   ```
   **这一步的意义**：把神经网络从 DSP（FFT/基音/OLA）里剥离出来。后面所有精度/性能对比都只比 `compute_rnn()`，口径才干净。
4. 用 `src/dump_features.c` 产出真实特征（或用 `examples/rnnoise_demo` 的音频），跑 500 帧。

**验收（CP-0）**
- [✅] 报告 4 个变体两两之间的 `gains` 最大/平均绝对误差，并**归因**：
  - `debugfloat-hiacc` vs `debugfloat` → 误差源 ①（激活近似），应 ≈ 6e-5 量级
  - `debugfloat` vs 默认 → 误差源 ②（权重量化），这是你的**生产基线误差**

- [✅] 回答：为什么 `float_weights` 字段是 NULL 时才走 int8？（线索：`src/parse_lpcnet_weights.c:91-96` 的 `opt_array_check` 对缺失返回 NULL 且不报错）

**作业结论：**
1. 物理意义：gain 是"这个频带保留多少能量"的系数（0~1）。差 5.5e-3 就是降噪强度差 0.55 个百分点——听感无感，但这就是你的生产基线。

MaxAE=16000 个样本里 |a−b| 的最大值	最坏能坏到什么程度
MeanAE=16000 个样本 |a−b| 的平均值	典型差多少
样本 16000=500 帧 × 32 频带，每个 (帧,频带) 是一个 gain	

2. 归因
4个变体，两个独立开关，**铁律：只有"两变体之间仅差一个开关"，其差值才等于该开关单独造成的误差**。

同时变更两个变量（开关）的对比无效。因为 GRU 是递归的，两个误差源会互相放大，不满足线性叠加。所以必须单独隔离，不能靠减法。

3. int8 0-255 范围， float 小数，范围大 -3.4E+38 到 3.4E+38
量化做法
```
w_q = round(w / scale)     # 存 1 字节
w   ≈ w_q * scale          # 用时还原
```
*round() 有损 → 这就是误差源 ②。RNNoise 的 scale 是逐输出神经元一个。* 
为什么值得：体积 1/4，且 SIMD 一次能处理 4 倍权重 → 更快。生产部署标配。

**生产基线**
官方 int8 方案自身的误差 = default_int8 vs debugfloat = MeanAE 5.503e-3 / MaxAE 7.700e-2。

实现量化的判断标准，实验基线： 
* X ≈ 5.5e-3 → 你完全复现了官方水平，残差来自你的实现
* X = 5e-2 → 你的实现有 bug（比官方差 10 倍）
* X = 1e-5 → 你走的是 float 路径，比官方更准

了解误差可能存在
1. 激活函数 近似值
2. float 权重做量化 int8处理


**完整 6 对 对比（16000 样本） - 误差基线表**

| 对比 | 差几个开关 | MaxAE | MeanAE | 归因 |
|---|---|---:|---:|---|
| `default_int8` vs `hiacc_int8` | **只差激活** | 1.708e-02 | **1.110e-03** | **① 激活近似**（int8 路径下） |
| `debugfloat` vs `hiaccdebugfloat` | **只差激活** | 1.881e-03 | **3.256e-04** | **① 激活近似**（float 路径下） |
| `default_int8` vs `debugfloat` | **只差权重** | 7.700e-02 | **5.503e-03** | **② 权重量化**（近似激活下）← **生产基线** |
| `hiacc_int8` vs `hiaccdebugfloat` | **只差权重** | 6.546e-02 | **5.800e-03** | **② 权重量化**（高精度激活下） |
| `default_int8` vs `hiaccdebugfloat` | 差两个 | 7.707e-02 | 5.611e-03 | ❌ 混合，**不可用于归因** |
| `hiacc_int8` vs `debugfloat` | 差两个 | 6.539e-02 | 5.692e-03 | ❌ 混合，**不可用于归因** |

误差 1 的放大原因： GRU 递归引起的误差积累导致 ？？？？？

权重量化误差 ≈ 激活近似误差的 17 倍
只有量化开关不同的情况/只有激活函数精度不同的情况=5.503e-3 / 3.256e-4，**主导项是权重量化**

**预估耗时**：2 天 ✅

---

### 阶段 1：GRU 从 0 到能手写前向（4 天） 9月9日 开始

**🎯目标**：不看任何框架，用 numpy 手写 384 维 GRU 单步前向，与 `compute_generic_gru` 数值对齐。

**输入**
- `src/nnet.c:65-94`（GRU 前向，全部 30 行，逐行读）
- `src/nnet_arch.h:153-161`（`diag` 补回）
- `src/rnn.h:40-46`（`RNNState`：哪些是状态）
- 深度文档 §2.3（GRU 专题）

**动作**
1. **先手推**（不许看答案）：把 `src/nnet.c:82-93` 这 12 行翻译成数学式，画出 `zrh[1152]` 和 `recur[1152]` 的布局图，标清楚：
   - 哪些下标做 sigmoid、哪些做 tanh
   - `r` 乘在哪一段
   - 为什么 `:84` 只累加 `2*N` 而不是 `3*N`
2. 用 numpy 实现：
   ```
   gru_step(x[384], h_prev[384], W_ih[1152,384], W_hh[1152,384],
            b_ih[1152], b_hh[1152], diag[1152]) -> h_new[384]
   ```
   门顺序按 PyTorch 的 **r, z, n** 组织（后面要交给 PyTorch）。
3. 直接从 `rnnoise_data.c` 读 `gru1_*` 的 7 个数组（行号见 §2 阶段 2 的表），跑 100 帧，与 C 的 `compute_generic_gru` 逐帧比对。
4. **反直觉实验**（必做）：把 `diag` 全部置零，跑 200 帧，观察状态如何漂移。这一条能让你真正理解"为什么对角要单独存 float"。

**验收（CP-1）**
- [ ] numpy 版与 C 逐帧误差 < 1e-6
- [ ] 闭卷画出 `zrh[3*384]` / `recur[3*384]` 布局图，标出 sigmoid/tanh 分界和 `r` 的作用位置
- [ ] 闭卷回答 `src/nnet.c:90`：`h[i] = z[i]*state[i] + (1-z[i])*h[i]`，为什么这里 `state[i]` 还是旧值？（提示：`:92` 才写回）
- [ ] 闭卷回答 `src/nnet.c:81` 的 `celt_assert(in != state)` 为什么必须有？如果 `in == state` 会发生什么？
- [ ] 闭卷回答：如果 `h̃` 也用 sigmoid 而不是 tanh，会怎样？（提示：状态恒非负，表达能力减半）

**预估耗时**：4 天（卡在门顺序是正常的，不要跳过手推那一步）

---

### 阶段 2：从 C 数组反推 PyTorch 模型（5 天）—— 任务 1 的核心

**目标**：得到 `RNNoisePyTorch`，权重全部来自官方 C 数组，**并且能通过官方 dump 脚本的正向闭环**。

**输入（权威维度源，只读这两个）**
- `rnnoise_data.h:8-38`（`*_IN_SIZE` / `*_OUT_SIZE` / `*_STATE_SIZE` / `*_DELAY`）
- `rnnoise_data.c:724217-724227`（`init_rnnoise()` 里 10 个 `linear_init()` 的最后两个实参就是每层的 `nb_inputs, nb_outputs`）

**50 个权重数组的实测行号**（用 `grep -n "^static const" rnnoise_data.c` 可自检，总数必须是 50）：

| 数组 | 类型 | 元素数 | 行号 |
|---|---|---:|---:|
| `conv1_weights_float` | float | 24,960 | `rnnoise_data.c:13` |
| `conv1_bias` | float | 128 | `:3143` |
| `conv2_weights_int8` | opus_int8 | 147,456 | `:3169` |
| `conv2_weights_float` | float | 147,456 | `:21612` |
| `conv2_subias` / `_scale` / `_bias` | float | 384 | `:40055` / `:40113` / `:40171` |
| `gru1_input_weights_int8` | opus_int8 | 442,368 | `:40229` |
| `gru1_input_weights_float` | float | 442,368 | `:95536` |
| `gru1_input_weights_idx` | int | 13,968 | `:150843` |
| `gru1_input_subias` / `_scale` / `_bias` | float | 1,152 | `:152599` / `:152753` / `:152907` |
| `gru1_recurrent_weights_diag` | float | 1,152 | `:153061` |
| `gru1_recurrent_weights_int8` | opus_int8 | 442,368 | `:153215` |
| `gru1_recurrent_weights_float` | float | 442,368 | `:208522` |
| `gru1_recurrent_weights_idx` | int | 13,968 | `:263829` |
| `gru1_recurrent_subias` / `_scale` / `_bias` | float | 1,152 | `:265585` / `:265739` / `:265893` |
| `gru2_*` | 同 gru1 | — | `:266047` 起，偏移 +225,818 |
| `gru3_*` | 同 gru1 | — | `:491865` 起，偏移 +451,636 |
| `dense_out_weights_float` | float | 49,152 | `:717683` |
| `dense_out_bias` | float | 32 | `:723837` |
| `vad_dense_weights_float` | float | 1,536 | `:723851` |
| `vad_dense_bias` | float | 1 | `:724053` |

**导出侧生成逻辑（逆向时要反着做）**
- `common.py:271-272`：Linear 权重转置（PyTorch `(out,in)` → C `(in,out)`）
- `common.py:290-293`：Conv 权重 `np.transpose(w, (2,1,0))` → `(k, in, out)` → `reshape(-1, out)`
- `common.py:297-300`：`STATE_SIZE = in_ch × (k−1)`、`DELAY = (k−1)//2` ← **只有产物没有训练代码时，用这个反推 kernel size**
- `common.py:342-353`：GRU 门顺序 **r,z,n → z,r,h** 的交换 + 转置
- `common.py:108-124`：`extract_diagonal` 抽 3 个 N×N 对角拼成长度 3N 的 `diag`
- `common.py:158-168`：8×4 块 —— `*_weights_int8` 块内是 `o*4+k`（输出优先），`*_weights_float` 块内是 `k*8+o`（输入优先）。**两份数组布局不一样，是刻意的**
- `common.py:175-188`：`compute_scaling` —— `max(max_abs/127, max_adjacent_pair_sum/129)`
- `common.py:245`：`subias = bias − Σ(weight_q·scale)` ← 说明真 bias 是 `bias`
- `common.py:248`：`final_scale = scale / 127`

**动作**
1. **只读练习（先做，不要写代码）**：合上所有文档，只看 `rnnoise_data.h:8-38` + `init_rnnoise()` 的 10 个调用，自己写出每层的 type / in / out / 激活，画出 `cat[1536]` 的布局。然后与 `torch/rnnoise/rnnoise.py:66-72` 对照，看差在哪。
2. 写 `parse_rnnoise_data.py`：正则解析 50 个 `static const` 数组。
3. 写 `load_weights.py`：装配成 PyTorch 参数。**必须自己推导的 5 个逆变换**（顺序很重要）：
   - 逆 8×4 块置换（从 `*_weights_float` 恢复 `(384, 1152)` 稠密矩阵）
   - 逆 `extract_diagonal`（把 `diag` 加回 `W_hh` 的 3 个 N×N 对角）
   - 逆门顺序（z,r,h → r,z,n）
   - 逆转置（`(in,out)` → `(out,in)`）
   - Conv 的 `(k,in,out)` → `(out,in,k)`
   - **用 `bias`，不要 `subias`**
   - **用 `*_weights_float`，不要用 `*_weights_int8`**（int8 是 `round(w/scale)`，有损，闭环过不了）
4. 模块命名**严格照抄** `torch/rnnoise/rnnoise.py:66-72`：`conv1 / conv2 / gru1 / gru2 / gru3 / dense_out / vad_dense`，类型也必须是 `nn.Conv1d / nn.Conv1d / nn.GRU ×3 / nn.Linear / nn.Linear`。

**验收（CP-2）—— 双向闭环，这是全文最重要的检查点**

```
官方 rnnoise_data.c  ──你的解析代码──▶  PyTorch 模型  ──官方 dump_rnnoise_weights.py --quantize──▶  rnnoise_data_mine.c
                     └──────────────────── 逐字节 diff ─────────────────────────────────────────────┘
```

```console
cd torch/rnnoise && python3 dump_rnnoise_weights.py --quantize your_model.pth /tmp/rnnoise_c
diff /tmp/rnnoise_c/rnnoise_data.c rnnoise_data.c
```

- [ ] **diff 为空**。做对了做错，脚本告诉你，不需要人告诉你。
- [ ] **分层判据**（diff 非空时按此定位）：
  - `conv1_weights_float` / `dense_out_*` / `vad_dense_*` 一致，但 GRU 的不一致 → 问题在门顺序 / 8×4 / diag 三者之一
  - 只有 `*_scale` / `*_weights_int8` / `*_subias` 不一致 → 权重对了，是量化参数路径（但那是脚本算的，不该错）→ 反推权重其实对了
  - **只有小数位末尾不同**（如 `0.1` vs `0.10000000149011612`）→ **dtype 问题**，你的 tensor 是 float64，转 float32 即可，权重本身没错
- [ ] 反证题：如果 conv2 的权重忘了转置（384×384 方阵），闭环会在哪一步抓住它？如果只看 forward 输出，能不能抓住？—— 这道题的答案就是你"为什么必须依赖闭环"的理由。

**预估耗时**：5 天（卡在 8×4 逆置换很正常，那是本阶段的设计目的）

---

### 阶段 3：时序对齐 —— 从"整段"到"逐帧"（3 天）

**目标**：证明 PyTorch 的序列输出与 C 的流式输出**逐帧相等**，并把残差归因到具体来源。

**输入**
- `src/nnet.c:113-123`（`compute_generic_conv1d`：历史缓冲怎么拼、怎么回写）
- `src/rnn.c:44-60`（主链）
- `torch/rnnoise/rnnoise.py:98-109`（官方 forward：两次 `permute` + 两次 valid conv + GRU + cat）

**动作**
1. **自己推导索引算术（本阶段核心，不许查答案）**：
   - `nn.Conv1d(k=3, padding='valid')` 输出第 `i` 个位置 = `Σ_{k=0..2} W[k]·x[i+k]`（`x` 的时间下标从小到大 = 从旧到新）
   - C 的第 `c` 帧，conv1 看到的是哪 3 帧？conv2 看到的又是哪 3 个 conv1 输出？（提示：`src/nnet.c:118-119` 的 `tmp = [mem, input]`，`mem` 是 2 帧历史；`src/rnnoise_data.h:12` `CONV1_STATE_SIZE = 65*2`）
   - **要让 PyTorch 输出第 `j` 个位置 == C 的第 `j` 帧，输入序列前面要补几个零帧？** 注意 C 的 `conv1_state`/`conv2_state` 初始是全零（`src/denoise.c:286` 的 `memset`），而 PyTorch 的 valid conv 在序列开头用的是真实帧。
2. 用 500 帧真实特征，跑 PyTorch 与 `build-debugfloat`，逐帧比对 `gains` 和 `vad`。
3. 误差三分归因（见缺陷 6 的表）。

**验收（CP-3）**
- [ ] 500 帧真实特征，PyTorch vs `build-debugfloat` 的 gains 最大/平均绝对误差报数，且**不超过你自己推导出的阈值**（阈值 = 激活近似 6e-5 × 传播放大系数，你要自己估计放大系数并说明理由）
- [ ] 闭卷：如果只补了 2 个零帧（而不是正确的个数），输出误差会是什么**形态**？（关键特征：误差**不随时间衰减**，因为这是错位不是数值误差 —— 这个判据以后能救你很多次）
- [ ] 闭卷：把 `padding='valid'` 改成 `'same'`，逐帧等价性还在吗？为什么？
- [ ] 闭卷：`src/denoise.c:474-496` 里，第 `n` 次调用算出的增益 `g` 是作用在哪个频谱上的？（提示：`delayed_X`；这个 20 ms 算法延迟怎么来的）

**预估耗时**：3 天

---

### 阶段 4：导出 ONNX + ORT 推理 + 公平性能对比（5 天）—— 任务 2

**目标**：一个可流式推理的 ONNX 模型 + 一份**站得住脚**的性能报告。

**输入**
- PyTorch `torch.onnx.export` 的 `input_names` / `output_names` / `dynamic_axes` / `opset_version`
- ORT 文档：`SessionOptions.intra_op_num_threads` / `inter_op_num_threads` / `graph_optimization_level` / `optimized_model_filepath`；`InferenceSession.run_with_iobinding`
- ORT 量化文档的前提：**被量化的模型 opset 必须 ≥ 10**（为阶段 5 铺路）

**动作**
1. **设计决策（必须自己做，写下来并给理由，至少列 3 个候选）**：
   - 候选 A：直接 export 官方 forward，输入 `(B, T, 65)`，`T ≥ 5`；流式靠维护 T 帧滑窗、每帧重算整窗
   - 候选 B：写一个 `RNNoiseStream` 包装，`forward(features[65], conv1_state[130], conv2_state[256], gru1_state, gru2_state, gru3_state) -> (gains[32], vad[1], *new_states)`；内部把 `nn.Conv1d` 展开成 `F.linear`，把 `nn.GRU` 手工展开
   - 候选 C：保留 `nn.GRU` 的 forward，让 ORT 跑 GRU op
   - **不可违反的硬约束**：子模块的 **名字 + 类型 + 维度** 必须与 `torch/rnnoise/rnnoise.py:66-72` 完全一致，否则 `dump_rnnoise_weights.py:60-91` 生成的数组名就变了，CP-2 的闭环断裂
   - **我的建议路线（你可以不同意，但要写出理由）**：候选 B + 手工展开 GRU，但**保留 `nn.GRU` 模块对象**（只为喂给 dump 脚本）。理由：手工展开后的图是 `MatMul/Add/Sigmoid/Tanh`，阶段 5 里这些 op 全部可被 ORT 量化；而 ONNX `GRU` op 的量化支持需要你自己去查证（见 CP-4）
   - **对照校验技巧（强烈建议先用）**：先证明"你手工展开的 GRU == `nn.GRU`" —— 这是纯 PyTorch 内部的对比，误差应 < 1e-6。这一步能把"GRU 语义理解错误"从"权重装配错误""布局错误"里**彻底分离**出来，是极其有效的调试分解
2. 导出 ONNX，`onnx.checker.check_model`；用 `onnx` Python API 自己打印一遍：节点数、各 op 类型统计、initializer 数与名字、输入/输出名与形状。
3. ORT 推理：实现流式循环，逐帧喂，与 C 逐帧比对。
4. **公平性能对比（本阶段最容易做错）**：
   - 基准是 `compute_rnn()`，不是 `rnnoise_process_frame()`（见缺陷 3d）
   - ORT 侧必须显式控制并写在报告里：`intra_op_num_threads`、`inter_op_num_threads`、`graph_optimization_level`、warmup 次数、重复次数、是否用 `run_with_iobinding`
   - 报告至少包含：单帧 P50 / P99、**每帧调用的固定开销**（方法：把 batch 维开大测"每帧边际成本"，两者相减就是固定开销）、线程数 1/2/4/8 的 scaling 曲线
   - **用 `perf` 看热点** —— 这是你的强项，别人没有的优势，一定要用上

**验收（CP-4）**
- [ ] ORT 输出 vs `build-debugfloat` 的逐帧误差，与 CP-3 **同量级**（说明 ONNX 导出本身没引入额外误差）
- [ ] 性能报告能回答："ORT 单线程 vs C 单线程谁快？快多少？差距里多少是 `run()` 调用开销、多少是计算？"
- [ ] 闭卷：把 `intra_op_num_threads` 从 1 调到 4，RNNoise 这种 **2.88 M MAC/帧**（≈ 288 MMAC/s）的小模型为什么几乎不加速甚至变慢？（提示：单次 GEMM 只有 384×1152，ORT 的并行拆分有阈值；且图是串行依赖链，`inter_op` 也用不上）
- [ ] 闭卷：你的 ONNX 里有多少个 initializer？它们对应 C 里的什么？（答：`rnnoise_arrays[]` 里的 50 个 `static const` 数组）

**预估耗时**：5 天

---

### 阶段 5：INT8 量化（6 天）—— 任务 3，你有金标准可以对照

**目标**：不只要"跑通量化"，要能说清"为什么官方那套 int8 方案比你 naive 的 ORT 量化好/差在哪"。

**输入**
- ORT 量化文档（动态 vs 静态的选择、QDQ vs QOperator、S8S8 默认、`reduce_range`、per-channel）
- **你自己的金标准**：`torch/rnnoise/dump_rnnoise_weights.py:15` 的 `unquantized` 列表 + `common.py:175-188` 的 `compute_scaling` + `src/nnet_arch.h:153-161` 的 diag 保 float + `src/vec.h:269-276` 的 int8 内核
- 深度文档 §3.4（int8 量化布局）与 §1.4（特征值域表）

**动作（分层做，每层都要留数据）**
1. **先建评价口径**（不许只看 MSE）：
   - 层级：每层输出的相对误差
   - 系统级：`gains` 的 MAE / MaxAE；用真实音频跑完整链路，算 PESQ / STOI / SNR
   - **时间维：误差是否随帧数累积**（跑 1000 帧看漂移）—— 这条对 RNN 最关键
2. **Baseline 0：官方 int8 方案**（默认构建）。这是**金标准**，先测出它的误差和速度。
3. **Baseline 1：ORT `quantize_dynamic`**（默认参数）。
4. **Baseline 2：ORT `quantize_static`** + 你自己设计的校准集（必须覆盖：静音 / 稳态噪声 / 浊音 / 清音 / 瞬态）。
5. **自己动手改**：用 `nodes_to_exclude` 把 conv1 排除掉，看误差怎么变；开 per-channel（per-axis），看误差怎么变。
6. **归因**：用 ORT 的量化调试 API 定位误差最大的张量 —— `onnxruntime.quantization.qdq_loss_debug` 里的 `create_weight_matching()` / `modify_model_output_intermediate_tensors()` / `collect_activations()` / `create_activation_matching()`。
7. **把官方的 4 个设计决策逐条翻译成"为什么"**（这是面试真正值钱的部分）：
   - 为什么 `conv1` 必须保持 float？→ 看 `features` 值域 `[−12, +5]`，而 int8 内核假设 `x ∈ [−1,1]`（`x_q = round(127·x)`）
   - 为什么 scale 是**逐输出神经元**？→ `common.py:181` 的 `np.max(np.abs(weight), axis=0)`，`axis=0` 是输入轴 → 逐列
   - 为什么 scale 还要额外约束"相邻两个输入的权重之和"？→ `common.py:184` 的 `weight[0::2] + weight[1::2]` / 129，对应 `src/vec.h:269-276` 一次取 4 个输入做乘加
   - 为什么 `W_hh` 的对角要抽出来单独存 float？→ `src/nnet_arch.h:153-161`；对角是自环增益，决定记忆时长，量化误差会被时间放大

**验收（CP-5）**
- [ ] 一张表，五列：**官方 int8 / ORT dynamic / ORT static / ORT static(排除 conv1) / ORT static(per-channel)** × 三行：**误差 / 速度 / 模型大小**
- [ ] 闭卷：GRU 的量化误差为什么"随时间累积"？写出递推式。（提示：`e_{t+1} ≈ z·e_t + δ`，`z→1` 时误差被记住，记忆时间常数 ≈ `1/(1−z)`）
- [ ] **实验证明**：人为把 `gru1_recurrent_weights_diag` 的值全部替换成 int8 精度的值（模拟"对角也被量化"），跑 1000 帧，报告误差漂移。**用实验证明官方这个设计决策是对的**
- [ ] 闭卷：如果校准集只有静音段，量化后会怎样？为什么？
- [ ] 对照：官方方案实际进 `.rodata` 的是 **3,541,508 B = 3.38 MiB**（int8 2,801,664 + idx 335,232 + 其余 float 404,612）。你 naive 量化后是多大？差在哪？

**预估耗时**：6 天

---

### 阶段 6：抽象成可迁移方法论 + 迁移验证（3 天）

**目标**：把这次经验变成"任意模型的 C 逆向 + 部署"能力。

**动作**
1. 把上面的步骤抽象成通用清单（见 §4），**用你自己的话重写一遍**，不要抄本文。
2. **迁移验证（强制）**：不改一行代码，把 `rnnoise_data.c` 换成 `rnnoise_data_little.c`（`README:121-125`：宏完全相同，同一个二进制可加载两种 blob），跑通全流程。
   - 你必须能解释：为什么 `weights_idx` 的长度从 **13,968** 变成 **4,752**？
   - 编码：`144 组 × (1 + 96) = 13,968` vs `144 + 4,608 = 4,752`
   - 密度：`13,824 块 = 100%`（标准模型一个块都没剪）vs `4,608 块 = 33.3%`（= `(0.3+0.2+0.5)/3`，与 `torch/rnnoise/rnnoise.py:43-50` 的 `sparse_params1` 平均值精确一致）
   - 为什么标准模型密度是 100%？→ `train_rnnoise.py:57` 的 `--sparse` 是 opt-in，发布模型没开
3. 面试问答自测（见 §5 CP-6）。

**预估耗时**：3 天

---

## 3. 每阶段「必须自己攻克 vs 可借力 AI」清单

| 阶段 | 必须自己攻克 | 为什么 | AI 可提效的部分 |
|---|---|---|---|
| **0 实验台** | 四个 C 构建变体的差异与归因；`dump_rnn_io` 工具（把网络从 DSP 剥离） | 这是后面所有"对/错"判断的基准线。基准线错了，后面全错，且你不会知道 | AI 可生成 autotools 多目录构建的脚本、`dump_rnn_io.c` 的文件 IO 样板；`--enable-dnn-debug-float` 的作用可以由 AI 提示，但**必须由你用 `configure.ac:82-88` 验证** |
| **1 GRU** | `src/nnet.c:82-93` → 数学公式的手推；`zrh[]` 布局图；`r` 的作用位置；`diag` 置零的破坏性实验 | 这是 92% 的计算量，也是你唯一的 0 基础区。**这一步靠 AI 给公式，你会"看懂但不会推"，到阶段 5 分析误差累积时就露馅** | AI 可做"陪练"：你推完给它看，让它挑错；可解释 `sigmoid` 为什么适合门、`tanh` 为什么适合候选状态。但公式必须你先写 |
| **2 权重装配** | 5 个逆变换的推导（8×4 逆置换 / diag 回加 / 门顺序 / 转置 / conv 轴序）；`bias` vs `subias` 的选择；`scale` 的计算顺序 | 这是"从 C 逆向"这项能力本身。**方阵转置错误是静默的**，只有闭环能抓住；如果你让 AI 直接给代码，你就失去了建立"布局直觉"的唯一机会 | AI 可写正则解析 `rnnoise_data.c` 的样板（50 个数组很机械）；可帮你把 `common.py:135-171` 的块循环用伪代码重述。**但逆变换的顺序和方向必须你自己定** |
| **3 时序对齐** | 索引算术的完整推导（补几个零帧、为什么）；误差三分归因 | "错位 N 帧"和"数值误差"的表现完全不同（前者不随时间衰减）。这个判据是你以后排查任何流式模型的第一工具 | AI 可生成批量比对的绘图代码；可帮你检查你的推导是否有 off-by-one。**但推导过程必须你先做完再给 AI 看** |
| **4 ONNX/ORT** | 导出形态的设计决策（A/B/C 三选一 + 理由）；性能对比口径（基准必须是 `compute_rnn()`）；线程数 scaling 的解释 | 这是"部署工程师"和"调 API 的人"的分水岭。口径错了，你拿去面试的数据就是笑话 | AI 可生成 ORT 的 SessionOptions 模板、`run_with_iobinding` 样板、`onnx` 图统计脚本。**但"和谁比""控制哪些变量"必须你定** |
| **5 量化** | 评价口径设计（层级/系统级/时间维三层）；校准集设计；官方 4 个设计决策的"为什么"；diag 量化破坏性实验 | 你的目标岗位是"模型部署 + 量化"。**只会调 `quantize_dynamic` 一文不值；能说清"为什么这层不能量化"才是值钱的** | AI 可生成量化脚本、校准数据读取器、PESQ/STOI 计算代码、误差归因的绘图。**但"为什么 GRU 难量化""校准集该含什么"必须你自己先有答案，再让 AI 补充** |
| **6 迁移** | 通用清单用自己话重写；little 模型的 idx 编码推导（13,968 → 4,752） | 迁移是这个项目的最终目的。解释不了 little 模型，说明你只记住了标准模型的数字 | AI 可出模拟面试题并打分；可帮你挑方法论里的漏洞 |

---

## 4. 可迁移方法论：从 C 逆向到 PyTorch 的通用步骤清单

做完 RNNoise 后，这套清单应该能套到任意模型的 C 实现上。

### 4.1 主流程（10 步）

| 步 | 动作 | RNNoise 里的具体做法 | 通用判据 |
|---|---|---|---|
| **0** | **先找"权威维度源"，屏蔽所有注释和可疑常量** | `rnnoise_data.h:8-38` 的宏 + `init_rnnoise()` 的 10 个 `linear_init()` 末两参（`rnnoise_data.c:724218-724227`）。**`src/rnn.c:41` 的 `INPUT_SIZE 42` 是死宏** | 判据：一个常量是否生效，看它出现在 `#define`+注释里，还是出现在**函数实参**里 |
| **1** | 定位模型入口（前向主链） | `compute_rnn()`，`src/rnn.c:44-60`，只有 10 行有效代码 | 找"把所有层串起来"的那个函数。通常很短，是整个逆向的地图 |
| **2** | 找正向导出脚本（**如果有，这是全场最省力的一步**） | `torch/rnnoise/dump_rnnoise_weights.py` + `wexchange/c_export/common.py` | 有导出脚本 = 有"权威的逆变换说明书"，也有闭环验证的可能。**先找它，再动手** |
| **3** | 逐层识别：把 C 的循环翻译成"矩阵乘 + 激活 + 状态" | `src/nnet.c:113-123`（conv1d → Linear + 滚动缓冲）；`src/nnet.c:65-94`（GRU） | 找 `compute_linear` / `sgemv` 类的调用点；每个调用点 = 一层 |
| **4** | 识别权重的内存布局 | (a) 行主序 `(in,out)`（与 PyTorch 转置）；(b) 8×4 分块（`common.py:59-61`、`158-168`）；(c) 对角抽取（`common.py:108-124`）；(d) 附加数组 `scale`/`subias`/`diag`/`idx` | 看**长度校验代码**：`src/parse_lpcnet_weights.c:123-176` 的 `linear_init` 里，每个 `find_array_check` 的期望字节数就是布局的声明 |
| **5** | 识别状态（哪些数组跨帧存活） | `src/rnn.h:40-46` 的 `RNNState`：`conv1_state[130]` / `conv2_state[256]` / `gru1,2,3_state[384]` | 判据：**在函数外分配、且在函数内被写回**的数组就是状态。这是 RNN 与纯前馈的唯一区别 |
| **6** | 建立双 baseline（参考精度实现 vs 生产量化实现） | `build-debugfloat` vs 默认构建 | 几乎所有生产 C 推理都有 int8/fast-math 开关。**必须先把它找出来并量化其误差**，否则你会把自己的正确实现改错 |
| **7** | 建立可比对的最小接口（把网络从前后处理里剥离） | 自写 `dump_rnn_io`，只调 `compute_rnn()`，绕开 FFT/基音/OLA | 判据：接口的两端应该只有"网络输入"和"网络输出"，不含任何 DSP |
| **8** | 装配权重 + 逆变换，用官方脚本闭环 | CP-2 的双向 diff | **闭环优先于直觉**。宁可多花一天搭闭环，也不要靠"输出看着差不多"来判断 |
| **9** | 时序对齐（索引算术） | 补 4 个零帧的推导 | 通用问题：C 的流式实现用"历史缓冲"，框架的批处理实现用"滑窗"，两者差一个**常量偏移**。这个偏移必须自己推 |
| **10** | 误差归因三分法 | 激活近似 / 权重量化 / 实现错误 | 每次看到误差，先问"它可能来自哪三个源"，再逐个隔离。**禁止在未归因前修改代码** |

### 4.2 子清单：如何识别权重布局

1. 找**长度校验**代码（`find_array_check` 的期望字节数）→ 得到每个数组的语义长度
2. 找**块循环**（`for i in range(M//8)` / `for j in range(N//4)` 这种）→ 得到块形状
3. 找**内核的展开方式**（`src/vec.h:269-276`：一次 8 个输出 × 4 个输入）→ 得到块内顺序
4. **警惕**：同一层的不同副本可能有不同布局（RNNoise 的 `*_weights_float` 是 `k*8+o`，`*_weights_int8` 是 `o*4+k`，见 `common.py:162-163`）
5. **警惕**：是否有元素被"抽走"单独存储（`diag`）。判据：矩阵里是否有整块的结构性零
6. **警惕**：是否有配套的"补偿数组"（`subias` 抵消无符号量化的常数项）

### 4.3 子清单：如何验证数值一致性

| 层级 | 验证对象 | 期望精度 | 失败意味着 |
|---|---|---|---|
| L0 | 解析出的数组元素数 vs 校验代码期望字节数 | 精确相等 | 解析错了 |
| L1 | **正向闭环**：官方脚本导出 vs 原始 C 数组 | **逐字节相同** | 逆变换错了（转置/块/门序/diag/偏差选择/dtype） |
| L2 | 手写 GRU vs `nn.GRU`（纯 PyTorch 内） | < 1e-6 | GRU 语义理解错了 |
| L3 | PyTorch vs C（float 构建） | 由激活近似误差决定（RNNoise ≈ 6e-5 × 放大系数） | 时序错位或权重装配错 |
| L4 | ONNX/ORT vs PyTorch | 与 L3 同量级 | 导出过程引入误差（opset / 算子替换） |
| L5 | INT8 vs FP32（系统级 + 时间维） | 自己定阈值并论证 | 量化策略问题 |

### 4.4 红旗清单（出现这些现象，先怀疑自己而不是代码库）

- 误差**不随时间衰减** → **时序错位**（帧偏移），不是数值问题
- 误差**随时间指数增长** → RNN 递归项错了（门顺序 / `r` 的位置 / `diag` 漏了）
- 只有方阵层（如 384×384）对不上，非方阵层都对 → **转置漏了**
- diff 只差小数末位 → **dtype 问题**（float64 vs float32）
- 改了 int8 数组但输出没变 → float 副本还在，`float_weights` 优先（见 `src/nnet_arch.h:138`）
- 输出全零 → `USE_WEIGHTS_FILE` 构建下没传 model（`src/denoise.c:298-303` 两条分支都被编译掉）

---

## 5. 检查点设计（每阶段结束的自测题 / 实验）

每个检查点的设计原则：**判据是客观的、可复现的，不依赖任何人的主观判断**。

### CP-0（阶段 0）：实验台
- **实验**：同一 500 帧特征，过 4 个 C 构建变体，报告两两 max/mean 绝对误差
- **判据**：能说出差值来自哪两个因素，并给出各自的定量估计（把 int8 权重还原成 float 重算后，误差应降到只剩激活近似那部分）
- **失败意味着**：基准线不可信，后面所有精度结论都无效

### CP-1（阶段 1）：GRU
- **实验**：numpy 版 GRU vs C `compute_generic_gru`，100 帧，误差 < 1e-6
- **实验**：`diag` 置零，跑 200 帧，记录状态漂移曲线
- **闭卷题**（4 道，见阶段 1 验收）
- **失败意味着**：92% 的计算量没搞懂，后面全是空中楼阁

### CP-2（阶段 2）：权重装配 —— 双向闭环（**全文最关键**）
- **实验**：`dump_rnnoise_weights.py --quantize` 正向导出，`diff` 为空
- **分层判据**：见阶段 2 验收表（能定位到具体是哪一类逆变换出错）
- **反证题**：conv2 方阵忘转置，闭环能否抓住？只看 forward 输出能否抓住？
- **失败意味着**：你还没建立"布局直觉"。**不要靠改参数蒙**，回到 `common.py:135-171` 逐行读

### CP-3（阶段 3）：时序对齐
- **实验**：500 帧真实特征，PyTorch vs `build-debugfloat` 逐帧比对
- **闭卷题**：补错零帧数时的误差形态（关键特征：不随时间衰减）；`padding='same'` 是否等价；`delayed_X` 的作用
- **失败意味着**：你会把"错位"误判成"数值误差"，然后花几天改正确的代码

### CP-4（阶段 4）：ONNX / ORT
- **实验**：打印 ONNX 图的节点数 / op 类型统计 / initializer 数 / 输入输出名与形状，自己列一遍
- **实验**：流式 500 帧 ORT vs C，误差与 CP-3 同量级
- **实验**：性能报告含 P50/P99/固定开销/线程 scaling 曲线，并用 `perf` 给出热点
- **闭卷题**：小模型多线程不加速的原因；initializer 对应 C 里的什么
- **失败意味着**：你产出的性能数据在面试里经不起追问

### CP-5（阶段 5）：量化
- **实验**：五方案 × 三指标对照表
- **实验**：diag 量化的破坏性实验（1000 帧误差漂移）
- **闭卷题**：GRU 误差累积递推式；校准集只有静音段的后果；官方 3.38 MiB 的构成
- **失败意味着**：你只会调 API，达不到"高级研发"的要求

### CP-6（阶段 6）：迁移与面试自测
- **实验**：换 `rnnoise_data_little.c`，不改一行代码跑通全流程，并解释 idx 长度 13,968 → 4,752 的编码与 100% → 33.3% 的密度
- **面试自测 10 问**（不看资料，口头答完再对照源码验证）：
  1. RNNoise 有几层？每层的类型、输入维、输出维、激活？
  2. 三层 GRU 是 `num_layers=3` 还是三个独立 GRU？输出层吃的是什么？
  3. 为什么 `dense_out` 的输入是 1536 而不是 384？
  4. C 的 GRU 门顺序是什么？PyTorch 是什么？谁在哪一步做了转换？
  5. 权重在 C 里按什么顺序存？和 PyTorch 差在哪？
  6. `diag` 是什么？为什么单独存？不存会怎样？
  7. 8×4 块里 32 个 int8 怎么排？float 副本和 int8 副本的排布一样吗？为什么？
  8. 为什么 `conv1` 不能量化，而 conv2 可以？
  9. `scale` 是逐层、逐通道还是逐输出神经元？为什么还要约束"相邻两个输入的权重和"？
  10. 算法延迟是多少毫秒？怎么来的？流式部署时有哪些硬约束？

---

## 6. 各阶段耗时与依赖总览

| 阶段 | 内容 | 耗时（3h/天） | 依赖 | 产出物 |
|---|---|---:|---|---|
| 0 | 实验台 | 2 天 | — | 4 个 C 构建变体、`dump_rnn_io`、误差基线表 |
| 1 | GRU 手写前向 | 4 天 | 0 | numpy GRU + 与 C 的对齐报告 + diag 破坏性实验 |
| 2 | C → PyTorch 逆向 | 5 天 | 1 | `parse_rnnoise_data.py` + `load_weights.py` + **闭环 diff 为空** |
| 3 | 时序对齐 | 3 天 | 2 | 逐帧对齐报告 + 误差三分归因 |
| 4 | ONNX + ORT + 性能 | 5 天 | 3 | 流式 ONNX 模型 + ORT 推理 + 性能报告 |
| 5 | INT8 量化 | 6 天 | 4 | 五方案对照表 + 官方设计决策的归因论证 |
| 6 | 方法论抽象 + 迁移验证 | 3 天 | 5 | 通用清单（自己话重写）+ little 模型验证 + 面试自测 |
| **合计** | | **28 天** | | |

---

## 7. 本文引用的事实清单（均已实测核验，可直接引用）

| 事实 | 出处 |
|---|---|
| 每帧 2,877,312 MAC；GRU 占 2,654,208（92.2%） | `rnnoise_data.h:8-38` + `init_rnnoise()` 维度推算 |
| `FRAME_SIZE 480` / `WINDOW_SIZE 960` / `FREQ_SIZE 481` / `NB_BANDS 32` / `NB_FEATURES 65` | `src/denoise.h:31-35` |
| `PITCH_MIN_PERIOD 60` / `PITCH_MAX_PERIOD 768` / `PITCH_FRAME_SIZE 960` / `PITCH_BUF_SIZE 1728` | `src/denoise.h:38-41` |
| 10 层的 `nb_inputs, nb_outputs`（195→128 / 384→384 / 384→1152 ×6 / 1536→32 / 1536→1） | `rnnoise_data.c:724218-724227` |
| 50 个权重数组的行号与元素数 | 见阶段 2 表（`grep -n "^static const" rnnoise_data.c` 自检） |
| GRU 前向与 `zrh` 布局（z,r,h） | `src/nnet.c:65-94` |
| Conv1d 退化为 Linear + 历史缓冲 | `src/nnet.c:113-123` |
| 4 条内核分派路径 | `src/nnet_arch.h:130-162` |
| `diag` 补回（仅 GRU recurrent，`3*M == N`） | `src/nnet_arch.h:153-161` |
| `HIGH_ACCURACY` 默认关闭，走多项式近似 | `src/nnet_arch.h:77`；近似实现 `src/vec.h:337-356` |
| `tanh_approx` 最大绝对误差 6.004e-5 @ x ≈ −5.205 | 本文实跑验证 |
| `SCALE (128.f*127.f)` 是死常量 | `src/vec.h:384`（全库 0 处使用） |
| `INPUT_SIZE 42` 是死宏 | `src/rnn.c:41`（仅 `:47` 的注释 printf 引用） |
| `LinearLayer` 结构 | `src/nnet.h:65-75` |
| int8 内核一次 8 输出 × 4 输入 | `src/vec.h:269-276` |
| `opt_array_check`（缺失不报错）vs `find_array_check`（缺失即失败） | `src/parse_lpcnet_weights.c:91-96` |
| `find_idx_check` 四道校验 | `src/parse_lpcnet_weights.c:98-121` |
| `linear_init` 长度校验即契约 | `src/parse_lpcnet_weights.c:123-176` |
| `DISABLE_DEBUG_FLOAT` 默认定义 | `configure.ac:82-88` |
| 权重导出：门顺序 r,z,n → z,r,h | `wexchange/c_export/common.py:342-353` |
| 权重导出：`scale` 先算，后抽对角 | `common.py:232` 早于 `:236` |
| `compute_scaling` 双约束 | `common.py:175-188` |
| 8×4 块：`*_int8` 输出优先 `o*4+k`，`*_float` 输入优先 `k*8+o` | `common.py:158-168` |
| `subias = bias − Σ(weight_q·scale)`；`final_scale = scale/127` | `common.py:245`、`248` |
| Conv 轴序 `(out,in,k) → (k,in,out)`；`STATE_SIZE = in_ch×(k−1)` | `common.py:290-300` |
| `unquantized = ['conv1','dense_out','vad_dense']` | `torch/rnnoise/dump_rnnoise_weights.py:15` |
| 模块名 → C 数组名的映射 | `torch/rnnoise/dump_rnnoise_weights.py:60-91` + `common.py:221-227` |
| PyTorch 模型定义（注意 `gru_size=256` 是过时默认值） | `torch/rnnoise/rnnoise.py:59-109` |
| 真实维度 `cond_size=128` / `gru_size=384` | `torch/rnnoise/train_rnnoise.py:48-49`、`:107` |
| 训练特征每帧 98 float（65 + 32 + 1） | `src/dump_features.c:487-489`；`train_rnnoise.py:74`、`:88` |
| little 模型密度 1/3，标准模型 100% | `torch/rnnoise/train_rnnoise.py:57` 的 `--sparse` 是 opt-in |
| 官方 int8 方案实际 `.rodata` = 3,541,508 B（3.38 MiB） | 深度文档 §3.2 汇总 |
| ORT：RNN 用动态量化、CNN 用静态量化 | ONNX Runtime "Quantize ONNX models" 文档 Method selection 节 |
| ORT：被量化模型 opset 需 ≥ 10 | 同上，Quantization and model opset versions 节 |
| ORT 量化调试 API | 同上，Quantization Debugging 节 |

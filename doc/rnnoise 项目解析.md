# RNNoise 项目解析 + 官方文档中文版

> 源码位置：`/Users/chengmo/Work/rnnoise`（不在本仓库内）
> 整理时间：2026-08-30

---

## 一、已有文档盘点（原文位置 / 是否覆盖你要的内容）

| 文档 | 语言 | 覆盖了你的哪些问题 | 位置 |
|---|---|---|---|
| `README` | 英文 | 功能（1-2、25-31 行）、编译与依赖（11-23 行）、训练流程（38-105 行）、模型二进制格式与可加载模型（107-125 行） | `/Users/chengmo/Work/rnnoise/README` |
| `torch/weight-exchange/README.md` | 英文 | 权重在 torch / tf.keras 之间交换、导出 C 文件 | `torch/weight-exchange/README.md` |
| `datasets.txt` | 英文 | 训练用公开语音数据集清单 + BibTeX | 根目录 |
| `doc/Doxyfile.in` | — | Doxygen 配置（只生成 API 文档，无架构说明） | `doc/` |
| 代码注释 | 英文 | **真正的技术细节全在这里** | `src/*.c/h` |

**结论：**
- **功能**：README 有（第 1-2、25-31 行）。
- **技术栈**：README 有片段（C / autotools / AVX2 / Python3 训练），但不完整。
- **第三方依赖**：**README 完全没写**（因为依赖全部是 vendored 进源码的，没有外部链接库）。我下面从源码版权头补齐。
- **理论基础**：README 只给了一行论文引用（第 2-7 行，arXiv:1709.08243），原理细节不在文档里。
- **技术细节**：README 几乎没有，只在 19-21 行提了 AVX2 编译建议、107-125 行提了模型 blob 格式。精华全在代码中。

中译版见本文 **第四、五部分**。

---

## 二、项目功能

**一句话**：一个基于 GRU 循环神经网络的实时语音降噪库，纯 C 实现，可在单个 x86/ARM 核上远快于实时地运行。

### 2.1 对外 API（`include/rnnoise.h`）

```c
int           rnnoise_get_size(void);                       // DenoiseState 字节数
int           rnnoise_get_frame_size(void);                 // 每帧采样点数 = 480
int           rnnoise_init(DenoiseState*, RNNModel*);       // 在"调用方自己分配的内存"上初始化
DenoiseState* rnnoise_create(RNNModel*);                    // model=NULL 用内置默认模型
void          rnnoise_destroy(DenoiseState*);
float         rnnoise_process_frame(DenoiseState*, float *out, const float *in);  // 返回 VAD 概率
RNNModel*     rnnoise_model_from_buffer(const void*, int);
RNNModel*     rnnoise_model_from_file(FILE*);
RNNModel*     rnnoise_model_from_filename(const char*);
void          rnnoise_model_free(RNNModel*);
```

### 2.2 关键参数（`src/denoise.h`）

| 常量 | 值 | 含义 |
|---|---|---|
| `FRAME_SIZE` | 480 | **48 kHz 下 10 ms**（不是 16 kHz/20 ms） |
| `WINDOW_SIZE` | 960 | 20 ms 分析/合成窗，50% 重叠 |
| `FREQ_SIZE` | 481 | rfft 频点数 |
| `NB_BANDS` | 32 | 频带数（ERB/Opus 频带划分） |
| `NB_FEATURES` | 65 | `32(能量DCT) + 32(基音相关DCT) + 1(基音周期)` |
| `PITCH_MIN/MAX_PERIOD` | 60 / 768 | 基音周期搜索范围（48 kHz） |

### 2.3 单帧处理流水线（`src/denoise.c:457-504`）

```
in[480] (float, ±32768 尺度)
  → 高通 DC 去除 biquad                       denoise.c:469-471
  → 加 Vorbis 窗 + 960 点 FFT                 denoise.c:186-198, 219-225
  → 32 频带能量 Ex（带间线性插值加权）         denoise.c:90-113, 63-65
  → 基音搜索 + 去倍频（复用 Opus CELT 代码）  denoise.c:359-370, src/pitch.c
  → 基音频谱 P、能量 Ep、互相关 Exp           denoise.c:371-377
  → 特征: [DCT(log Ex) | DCT(Exp) | 0.01*(period-300)]   denoise.c:378-396
  → 静音门限 E<0.04 时跳过 RNN，冻结状态      denoise.c:389-393
  → RNN 推理 → 32 个频带增益 g + VAD 概率     src/rnn.c:44-60
  → 增益时间平滑（衰减上限 0.6/帧 ≈ RT60 135ms）+ 能量补偿   denoise.c:479-487
  → 频带→频点插值 gf，乘到**上帧**频谱        denoise.c:488-494（算法延迟 = 1 帧 = 10 ms）
  → IFFT + 重叠相加合成                       denoise.c:200-217, 400-407
out[480], return vad_prob
```

**注意**：RNN 只输出 22→现在 32 个**频带增益**（不直接改波形），波形重建仍由传统 DSP（FFT/加窗/OLA/梳状滤波）完成——这正是论文标题 "A Hybrid DSP/Deep Learning Approach" 的含义。

### 2.4 附带工具

- `examples/rnnoise_demo`：`RAW 16-bit 48 kHz 单声道 PCM → 降噪 PCM`（`examples/rnnoise_demo.c`）
- `dump_features`：训练用特征生成器（`-DTRAINING` 编译，`src/dump_features.c`）
- `dump_weights_blob`：把编译进来的默认模型导出成可运行时加载的 `weights_blob.bin`（`src/write_weights.c`）

---

## 三、技术栈与第三方依赖

### 3.1 技术栈

| 层 | 技术 |
|---|---|
| 推理库 | C99，autotools（autoconf + automake + libtool），`./autogen.sh && ./configure && make` |
| SIMD | x86 SSE4.1 / AVX2+FMA（**运行时分派 RTCD**），ARM NEON；可选 `--enable-x86-rtcd` |
| 模型训练 | Python 3 + PyTorch（`torch/rnnoise/`），序列长度 2000 帧，AdamW |
| 权重导出 | `wexchange`（LPCNet 生态）：torch/tf.keras → C 数组 或 二进制 blob |
| 文档生成 | Doxygen（可选） |
| 数据 | 48 kHz / 16-bit / 机器字节序 RAW PCM |

### 3.2 第三方依赖（**全部 vendored，运行时零外部依赖**）

源码里没有 `include <some_lib.h>`，依赖都是"复制进来的代码 + 改前缀"，因此**链接时只需要 libm**。从版权头可确认来源：

|  vendored 组件 | 来源 / 版权 | 文件 |
|---|---|---|
| **Kiss FFT** | Mark Borgerding, Xiph.Org (BSD) | `src/kiss_fft.c/h`、`src/_kiss_fft_guts.h`、`src/rnnoise_tables.c`（旋转因子表，由 `dump_rnnoise_tables.c` 生成） |
| **Opus / CELT 的基音分析与 LPC** | Octasic Inc. → Xiph.Org → Jean-Marc Valin (BSD) | `src/pitch.c/h`（`xcorr_kernel`、`rnn_pitch_search`、`rnn_remove_doubling`）、`src/celt_lpc.c/h`、`src/arch.h`、`src/opus_types.h`、`src/common.h` |
| **LPCNet 的 DNN 内核** | Mozilla (2018) + Amazon (2023) | `src/nnet.c`、`src/nnet_arch.h`、`src/nnet.h`、`src/vec.h`、`src/vec_avx.h`、`src/vec_neon.h`、`src/parse_lpcnet_weights.c` |
| **x86 CPUID 检测** | Xiph.Org / Mozilla | `src/x86/x86cpu.c/h`、`src/x86/x86_dnn_map.c` |
| **模型二进制文件** | Xiph.Org 服务器（构建时下载） | `download_model.sh` + `model_version`（sha256 校验） → 生成 `src/rnnoise_data.c/h` |

download_model.sh 

**符号冲突处理**：`src/nnet.h:89-107` 用一串 `#define` 给所有 LPCNet 符号加 `rnn_` 前缀（`linear_init → rnn_linear_init`…），这样 RNNoise 和 Opus 可以同时链进一个二进制。这是很值得学的工程技巧。

Python 侧依赖（`torch/weight-exchange/requirements.txt`）：`numpy`；`torch` 与 `tensorflow` 都是**按需可选**（`import exchange` 不会强制 import 二者）。

### 3.3 构建时的外部依赖（非运行时）

`wget`（下载模型）、`autoconf/automake/libtool`、`doxygen`（可选）。

---

## 四、理论基础

### 4.1 论文

> J.-M. Valin, **A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement**, IEEE MMSP Workshop, arXiv:1709.08243, 2018.
> https://arxiv.org/pdf/1709.08243.pdf （README:2-7 已给出）

核心思想：
1. **不预测波形/频谱，只预测频带增益**（32 个 ERB-like 频带的理想增益，形式类似维纳滤波的增益），把"难建模的波形重建"留给传统 DSP。→ 模型极小、不会引入音乐噪声的怪异结构。
2. **混合架构**：特征提取（FFT、频带划分、基音分析、DCT）全是传统 DSP；只有"从特征到增益"的映射用 RNN。
3. **循环结构是关键**：GRU 让降噪器有时序记忆，能利用"噪声平稳、语音非平稳"的先验，显著优于逐帧独立的 DNN。

### 4.2 网络结构（`src/rnn.c:44-60` + `torch/rnnoise/rnnoise.py:58-109`）

```
输入 65 维特征
  ↓ Conv1d(65 → cond_size=128, k=3, valid) + tanh
  ↓ Conv1d(128 → gru_size,      k=3, valid) + tanh
  ↓ GRU(gru_size)  ×3  串联
  ↓ concat([conv2输出, gru1, gru2, gru3]) = 4×gru_size
  ├→ Linear(4×gru_size → 32) + sigmoid  → 频带增益 g
  └→ Linear(4×gru_size → 1)  + sigmoid  → VAD 概率
```

- `gru_size`：类默认 256，训练脚本 `train_rnnoise.py:49` 默认 **384**，实际发布模型以 `src/rnnoise_data.h` 中的 `GRU*_STATE_SIZE` 为准（该文件由 `autogen.sh` 下载，当前仓库中没有）。
- Conv1d(k=3, valid) 在 C 侧实现为"带历史记忆的 LinearLayer"：把前 2 帧输入缓存进 `conv*_state`，拼成 3 帧后做一次矩阵乘（`src/nnet.c:113-123`）。**状态 = 2 帧历史，无未来前瞻**。
- 计算量估算（以 gru_size=384 为例）：3 个 GRU ≈ 3 × 3 × 384 × (384+384) ≈ 2.65 M MAC/帧，10 ms 一帧 → 约 0.5~1 ms/帧量级，配合稀疏化后更低。**这是"能在单核上实时"的根本原因：模型小 + 8bit 量化 + 结构化稀疏。**

### 4.3 训练目标（`torch/rnnoise/train_rnnoise.py:145-156`）

```python
target_gain = clamp(gain, min=0) * tanh(8*gain)**2        # 抑制极小增益样本
e = pred_gain**gamma - target_gain**gamma                  # gamma=0.25 感知指数
gain_loss = mean((1 + 5*vad) * mask(gain) * e**2)          # 有语音处权重 ×6
vad_loss  = mean(|2v-1| * BCE(vad, pred_vad))              # 难样本加权
loss = gain_loss + 0.001 * vad_loss
```

三个设计点值得注意：
1. **在 `gain^gamma` 域算损失**（gamma=0.25）：等价于对增益做感知压缩，让模型在低增益（强降噪）区域更精细，符合听觉。
2. **`(1+5*vad)` 加权**：语音段权重大 6 倍——降噪器在有人说话时才值得优化。
3. **`mask(gain) = clamp(g+1, max=1)`**：训练标签里 `g = -1` 表示"该频带无监督信号"（静音/超出低通带宽/能量过低），用 mask 把这些位置的梯度清零（`src/dump_features.c:472-478`）。

### 4.4 训练数据仿真（`src/dump_features.c`）

一条 2000 帧（20 s）序列的构造流程：随机位置截取语音/背景噪声/前景噪声 → 随机增益（-45~-0 dB 等）→ 随机二阶滤波器（`rand_filt`）→ 随机低通（模拟带宽限制）→ 可选 RIR 卷积加混响 → 随机削波 / 16-bit 量化 → 混合。
语音段标签用 **Viterbi VAD**（`dump_features.c:199-254`）离线生成，再做淡入淡出，避免"语音突然截断"的错误标签。

---

## 五、优秀的技术细节（值得抄进自己的项目）

### 5.1 8-bit 权重量化 + 每输出行缩放

- 权重存成 `opus_int8`，每**一行输出**配一个 float `scale`（`src/nnet.h:65-75`, `src/vec.h:248-311`）。
- 激活也量化成 int8（`x[i] = floor(.5 + 127*_x[i])`，**输入是 [-1,1] 归一化后的**），int8×int8 累加进 int32/float，最后统一乘 scale。
- `SCALE = 128*127`、`SCALE_1 = 1/128/127`（`src/vec.h:384-385`）用于把量化误差折算回浮点域。
- **不量化的层**：`conv1`、`dense_out`、`vad_dense`（`torch/rnnoise/dump_rnnoise_weights.py:15`）——第一层和最后一层对精度最敏感，这是很实用的经验。

### 5.2 结构化稀疏：8×4 块 + 索引数组

- 权重按 **8 行 × 4 列** 的块做结构化剪枝（`sparse_params1` 里 `[8,4]`，`torch/rnnoise/rnnoise.py:43-50`），非零块用 `weights_idx` 索引数组记录列位置，数据紧凑存放（`src/vec.h:123-180` 的 `sparse_sgemv8x4`、`sparse_cgemv8x4`）。
- 各门不同稀疏度：`W_hz/W_iz` 0.2（更新门最稀疏）、`W_hr/W_ir` 0.3、`W_hn/W_in` 0.5（新门/候选值最密）。
- **训练中渐进稀疏化**：step 6000 开始、20000 结束，每 100 步更新一次 mask，密度按 `alpha = ((stop-i)/(stop-start))^3` 插值（`rnnoise.py:38-41` + `torch/sparsification/gru_sparsifier.py`）。不是训完再剪，而是边训边剪，精度损失更小。
- **GRU 循环权重保留对角元**：`keep_diagonal=True` → 对角线单独存成 `diag` 数组，在 `compute_linear_` 里加回（`src/nnet_arch.h:153-161`）。因为 h→h 的对角（自连接）最重要。
- **"little" 模型**：尺寸与常规模型完全相同、只是更稀疏（`README:121-125`），因此**可以在运行时任意切换而不改代码**——这个设计很聪明。

### 5.3 GEMV 内核的写法

- `sgemv16x1 / sgemv8x1`：一次算 16/8 行输出，**内循环遍历列、外循环固定行**，让 `out[0..15]` 全留在寄存器里，加法链无依赖，利于流水线与自动向量化（`src/vec.h:49-121`）。
- `sparse_cgemv8x4`：一次读 4 个输入、算 8 个输出，内层 32 次乘加完全展开（`src/vec.h:248-281`）。
- **依赖编译器自动向量化而非手写 intrinsic**：`src/nnet_arch.h:51-57` 用 `#pragma GCC optimize("tree-vectorize")` 强制对 DNN 代码开启向量化；`conv2d_3x3_float` 上方注释（`:193-194`）明确写"没有 intrinsic，gcc 自动向量化足够聪明"。

### 5.4 运行时 SIMD 分派（RTCD）

- 同一份 `nnet_arch.h` 被 include 三次，靠 `#define RTCD_ARCH c / sse4_1 / avx2` 生成三套符号（`src/nnet_default.c`、`src/x86/nnet_sse4_1.c`、`src/x86/nnet_avx2.c`）。**一份源码，三种实现**，无重复代码。
- 运行时用函数指针表分派，arch 存在**每个 DenoiseState 里**（`st->arch`，`src/denoise.c:304`）：`compute_linear(...) ((*RNN_COMPUTE_LINEAR_IMPL[(arch)&OPUS_ARCHMASK])(...))`（`src/x86/dnn_x86.h:46-78`）。
- `src/vec_avx.h` 自带 **AVX2 的 SSE 模拟层**（`mm256_emu`，用两个 `__m128` 模拟 `__m256`）和 **`dpbusds`（VNNI 风格 int8 点积累加）的模拟实现**（`:629-670`），所以在只支持 SSE4.1 的机器上也能用同一套 int8 代码路径。

### 5.5 零动态内存分配 / 可嵌入设计

- 推理路径上**没有 malloc**：所有中间缓冲都是栈上定长数组（`MAX_ACTIVATIONS 4096`、`MAX_RNN_NEURONS_ALL 1024`，配合 `celt_assert` 编译期/调试期检查，`src/nnet_arch.h:60`、`src/nnet.c:63`）。
- `rnnoise_get_size()` + `rnnoise_init()` 允许调用方**在外部分配内存**（内存池、共享内存、静态数组都行）；`common.h:15-27` 还提供 `OVERRIDE_RNNOISE_ALLOC/FREE` 钩子，可整体替换分配器。→ 非常适合放进不允许动态分配的服务/嵌入式环境。
- 栈上缓冲大小硬编码上限，是"用可控的固定栈空间换确定性延迟"的取舍。

### 5.6 模型与状态分离（多并发连接的正确姿势）

- `RNNModel`（只读权重）与 `DenoiseState`（每路音频一份）是**两个对象**，`rnnoise_model_from_*` 加载一次，`rnnoise_create(model)` 可以创建任意多个 state（`src/denoise.c:227-321`）。
- 对实时服务来说这是关键：**权重只占一份内存，每连接只增加一个 state**（约 30~35 KB，量级随 gru_size 变化，用 `rnnoise_get_size()` 实测）。
- 陷阱（README:112-115 明确警告）：`model` 对象必须在所有 `DenoiseState` 销毁后才能 free；从 `FILE*` 加载时文件也不能提前关闭。

### 5.7 模型 blob 文件格式（自研、极简）

```
"DNNw" | version(int) | type(int) | size(int) | block_size(int) | name[44]   ← 64 字节定长头
data[size] + 补零到 64 字节对齐
```
（`src/nnet.h:41-62`、`src/parse_lpcnet_weights.c:37-78`、`src/write_weights.c:46-69`）

设计要点：定长 64 字节头 + 64 字节对齐的数据块，便于 mmap 直接解析（`parse_lpcnet_weights.c:205-237` 里就有一段 mmap 的示例代码）；解析时对 `size/block_size/name` 都做边界校验，防止坏文件导致越界。

### 5.8 信号处理侧的巧思

- **频带能量用"相邻频带线性插值"分配**（`compute_band_energy`，`src/denoise.c:90-113`）：一个频点的能量按位置分给左右两个频带，等价于三角滤波器组，但只需一次乘加。
- **能量对数的动态范围钳制**：`Ly[i] = MAX(logMax-7, MAX(follow-1.5, Ly[i]))`（`denoise.c:382-388`）——防止某帧能量骤变把 RNN 状态带偏。
- **增益衰减上限 + 能量补偿**（`denoise.c:479-487`）：`g[i] = MAX(g[i], 0.6*lastg[i])` 限制每帧最多衰减到 0.6（对应 RT60 135 ms），避免"抽水机效应"；同时用 `delayed_Ex/Ex` 比值补偿帧间能量变化，防止瞬态噪声漏出。
- **基音梳状滤波后能量归一化**（`rnn_pitch_filter`，`denoise.c:421-455`）：先按基音相关度加回周期分量（保护语音谐波），再把总能量归一化回原值（避免放大）。
- **静音时不跑网络也不更新状态**（`denoise.c:389-393`）：省 CPU 且避免状态被静音污染。
- **历史遗留但刻意为之的常量**：`dct()` 里归一化系数是 `sqrt(2/22)`（`denoise.c:168`），而 `NB_BANDS` 现在是 32——这是早期 22 频带时代留下的数。因为训练和推理用的是同一份代码，所以不影响正确性；但自己改 `NB_BANDS` 时必须重训。这类"看起来像 bug 的历史包袱"在信号处理代码里很常见。

### 5.9 与你现有计划的**冲突点**（重要）

你在 `CDNStudy/webrtc/离线音频降噪项目.md` 和 `AI 实时音视频.md` 里写的是：

> "处理一帧 20ms（320 采样）16kHz 16-bit PCM"

**这与 RNNoise 的实际接口不符**，需要修正：

| 项 | 你的文档 | RNNoise 实际 |
|---|---|---|
| 采样率 | 16 kHz | **48 kHz**（固定，`README:26-31`） |
| 帧长 | 320 采样 / 20 ms | **480 采样 / 10 ms**（`src/denoise.h:31`） |
| 数据类型 | int16 | **float**（±32768 尺度），`rnnoise.h:94` |
| 返回值 | 无 | **VAD 概率**（0~1），可直接用来做"AI 智能开关" |
| 输出格式 | — | RAW PCM，**不是 WAV**（`README:32`） |

后果与做法：
1. 用 16 kHz 素材必须先**重采样到 48 kHz**（`ffmpeg -ar 48000`），否则模型看到的是错位频谱，效果会很差。
2. 不能改 `FRAME_SIZE` 来适配 16 kHz：`eband20ms` 频带表（`denoise.c:63`）和模型输入维数（65）、频带数（32）都和 48 kHz 绑定，改了**必须重训模型**。
3. int16 ↔ float 转换要自己 clamp：官方 demo 直接把 float 写回 `short`（`examples/rnnoise_demo.c:58`），溢出会环绕，生产代码要饱和处理。
4. 官方 demo **丢掉了第一帧输出**（`examples/rnnoise_demo.c:59`，`if (!first) fwrite`），因为算法有 1 帧（10 ms）延迟。做延迟统计时要算上这 10 ms 算法延迟 + 10 ms 帧长。
5. 你的对比实验里，WebRTC NS 通常跑在 16 kHz/32 kHz；要公平对比，两边应统一到 48 kHz，或统一到 16 kHz（RNNoise 侧需重采样，会引入额外变量，建议统一到 48 kHz）。

---

## 六、`torch/weight-exchange/README.md` 中文翻译

> # weight-exchange
>
> ## 权重交换
> 本仓库提供在 torch 与 tensorflow.keras 模块之间交换权重的工具，中间使用 numpy 格式。
>
> 加载/导出 torch 权重的例程位于 `exchange/torch`，可以这样导入：
> ```
> import exchange.torch
> ```
> 加载/导出 tensorflow 权重的例程位于 `exchange/tf`，可以这样导入：
> ```
> import exchange.tf
> ```
>
> 注意：`exchange.torch` 需要已安装 torch，`exchange.tf` 需要已安装 tensorflow。
> 为了避免工作环境中必须同时安装 torch 和 tensorflow，调用 `import exchange` 时
> **不会**导入这两个子模块。同样地，`requirements.txt` 中既不包含 Tensorflow 也不包含 Pytorch。
>
> ## C 导出
> `exchange.c_export` 模块包含把权重导出为 C 文件的例程。长远来看，所有 `dump_...` 函数
> 都应该既能接受路径字符串、也能接受 `CWriter` 实例，并根据传入类型自动选择导出格式。
> 目前这仅对 `torch.nn.GRU`、`torch.nn.Linear` 和 `torch.nn.Conv1d` 三种层实现。

---

## 七、附：关键文件速查表

| 想看什么 | 看哪个文件 |
|---|---|
| 对外 API | `include/rnnoise.h` |
| 主流程（一帧怎么走完） | `src/denoise.c:457-504` |
| 特征提取（频带/基音/DCT） | `src/denoise.c:90-398` |
| 网络前向（层连接顺序） | `src/rnn.c:44-60` |
| GRU / Conv1d 实现 | `src/nnet.c:65-123` |
| 量化 + 稀疏 GEMV（C 版） | `src/vec.h:123-311` |
| 量化 + 稀疏 GEMV（AVX2/SSE） | `src/vec_avx.h:672-880` |
| SIMD 分派 | `src/x86/dnn_x86.h`、`src/x86/x86_dnn_map.c` |
| 模型 blob 格式 | `src/nnet.h:41-62`、`src/parse_lpcnet_weights.c` |
| PyTorch 模型定义 | `torch/rnnoise/rnnoise.py` |
| 训练循环与损失函数 | `torch/rnnoise/train_rnnoise.py:128-178` |
| 稀疏化调度 | `torch/sparsification/gru_sparsifier.py` |
| 权重导出 C（含量化） | `torch/rnnoise/dump_rnnoise_weights.py` |
| 训练数据仿真 | `src/dump_features.c` |
| 构建配置 | `Makefile.am`、`configure.ac` |

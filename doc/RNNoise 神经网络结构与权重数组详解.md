# RNNoise 神经网络结构与权重数组详解

> 面向读者：有多年 C 系统编程经验、机器学习是新手的工程师。
> 目标：彻底搞懂 RNNoise 的**层结构**和**权重数组**这两件事，能自己改代码、能自己导出模型。
>
> 代码基线：本仓库 `master`，模型包 `model_version = 0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37`。
>
> **关于引用路径**：本文所有 `文件:行号` 均指仓库内源码。
> 例外是 `src/rnnoise_data.h` / `src/rnnoise_data.c` —— 这两个文件**不在 Git 里**（太大），由 `download_model.sh:30` 下载解压得到。
> 在本工作区它们当前位于仓库根目录（`rnnoise_data.h` / `rnnoise_data.c`），按 `README:101` 的指示应拷贝到 `src/`。
> 下文统一按**拷贝后的规范路径** `src/rnnoise_data.*` 引用；行号与根目录副本完全一致（两个副本已 `diff` 验证为同一文件）。

---

## 0. 阅读指南

### 0.1 全景图：从 PCM 采样到降噪输出

RNNoise 是 **DSP + 深度学习混合**架构：DSP 负责"把信号变成 65 个数字"和"把 32 个数字变回信号"，神经网络只负责中间那一步"看数字、给增益"。

```
                    ┌────────────── 输入：每帧 480 个 float（48 kHz → 10 ms） ──────────────┐
                    │                                                                      │
  in[480]           │                                                                      │
     │              │                                                                      │
     ▼              │                                                                      │
 ① 高通滤波(DC阻断) │  rnn_biquad()                                  src/denoise.c:409-419 │
     │              │  调用点                                        src/denoise.c:471     │
     ▼              │                                                                      │
 ② 拼 960 点分析窗  │  [上一帧 480 | 本帧 480]                        src/denoise.c:332-338 │
     │              │                                                                      │
     ▼              │                                                                      │
 ③ 加 Vorbis 窗+FFT │  apply_window() / forward_transform()           src/denoise.c:219-225 │
     │              │                                                src/denoise.c:186-198 │
     ▼              │                                                                      │
   X[481] 复数频谱  │                                                                      │
     │              │                                                                      │
     ├──────────► ④ 32 频带能量 Ex[]    compute_band_energy()        src/denoise.c:90-113  │
     │              │                                                                      │
 ⑤ 基音搜索 pitch   │  rnn_pitch_search / rnn_remove_doubling        src/denoise.c:359-370 │
     │              │                                                src/pitch.h:42-46     │
     ├──────────► ⑥ 回退一个基音周期的窗 → FFT → P[481]              src/denoise.c:371-374 │
     │              │                                                                      │
     ├──────────► ⑦ 频带互相关 Exp[]，用 sqrt(Ex·Ep) 归一化          src/denoise.c:376-377 │
     │              │                                                                      │
     ▼              │                                                                      │
 ⑧ 组装 65 维特征   │  features[0..31]  = DCT(对数频带能量) − 平移    src/denoise.c:380-396 │
                    │  features[32..63] = DCT(归一化基音相关)                               │
                    │  features[64]     = 0.01 × (基音周期 − 300)                           │
                    └────────────────────────────┬─────────────────────────────────────────┘
                                                 │
              ╔══════════════════════════════════▼══════════════════════════════════╗
              ║              神 经 网 络（ compute_rnn() ）                     ║
              ║                                                                     ║
              ║   features[65]                                                      ║
              ║        │                                                            ║
              ║        ▼                                                            ║
              ║   Conv1d(65→128, k=3, tanh)  + 2 帧历史        src/rnn.c:48         ║
              ║        ▼  tmp[128]                                                  ║
              ║   Conv1d(128→384, k=3, tanh) + 2 帧历史        src/rnn.c:49         ║
              ║        ▼  cat[0..383]                                               ║
              ║   GRU(384) × 3  串联                          src/rnn.c:50-52       ║
              ║        ▼  cat[384..767 / 768..1151 / 1152..1535]                    ║
              ║   拼接 → cat[1536]                            src/rnn.c:53-55       ║
              ║        ├──────────► Linear(1536→32, sigmoid) → g[32]  频带增益      ║
              ║        └──────────► Linear(1536→1,  sigmoid) → vad    语音概率      ║
              ║                                                src/rnn.c:56-57      ║
              ╚══════════════════════════════════┬══════════════════════════════════╝
                                                 │  g[32]
              ┌──────────────────────────────────▼──────────────────────────────────┐
              │  ⑨ 时间平滑（RT60 = 135 ms）+ 能量补偿        src/denoise.c:479-487  │
              │  ⑩ 频带增益插值成 481 个频点 gf[]             src/denoise.c:488      │
              │  ⑪ 基音梳状滤波 + 能量归一（后滤波）           src/denoise.c:478      │
              │                                              src/denoise.c:421-455  │
              │  ⑫ 乘到【上一帧】的频谱 delayed_X             src/denoise.c:490-493  │
              │  ⑬ IFFT → 960 点 + 加窗                      src/denoise.c:200-225   │
              │  ⑭ 重叠相加（OLA）输出 480 点                  src/denoise.c:400-407  │
              └──────────────────────────────────┬──────────────────────────────────┘
                                                 ▼
                                             out[480]
```

**关注两点**

1. **神经网络只在中间那一个小盒子里**：输入 65 个 float，输出 32 个增益 + 1 个 VAD 概率。它完全不知道"波形"是什么。
2. **所有"音频味"的东西**（滤波、加窗、FFT、基音、增益平滑、梳状滤波）都是手写 C，不在权重里。

### 0.2 关键常量速查表

信号 / 特征相关：

| 常量 | 值 | 定义位置 | 含义 |
|---|---:|---|---|
| `FRAME_SIZE` | 480 | `src/denoise.h:31` | 每帧采样数；48 kHz 下 = **10 ms** |
| `WINDOW_SIZE` | 960 | `src/denoise.h:32` | 分析窗长度 = 2 × FRAME_SIZE（50% 重叠） |
| `FREQ_SIZE` | 481 | `src/denoise.h:33` | FFT 后保留的正频率点数 = FRAME_SIZE + 1 |
| `NB_BANDS` | 32 | `src/denoise.h:34` | 频带数（近似 ERB 划分） |
| `NB_FEATURES` | 65 | `src/denoise.h:35` | **神经网络输入维度** = 2 × NB_BANDS + 1 |
| `PITCH_MIN_PERIOD` | 60 | `src/denoise.h:38` | 最小基音周期（60 采样 @48k = 800 Hz） |
| `PITCH_MAX_PERIOD` | 768 | `src/denoise.h:39` | 最大基音周期（768 采样 @48k = 62.5 Hz） |
| `PITCH_FRAME_SIZE` | 960 | `src/denoise.h:40` | 基音搜索窗长（20 ms） |
| `PITCH_BUF_SIZE` | 1728 | `src/denoise.h:41` | 基音历史缓冲 = 768 + 960 |
| `MAX_NEURONS` | 1024 | `src/rnn.h:37` | `compute_rnn` 里 `tmp[]` 的大小上界 |
| `MAX_RNN_NEURONS_ALL` | 1024 | `src/nnet.c:63` | GRU 隐层单元数上界（数组实际开 3×，见 `src/nnet.c:69-70`） |
| `MAX_CONV_INPUTS_ALL` | 1024 | `src/nnet.c:111` | Conv1d 展开后输入维上界 |
| `MAX_INPUTS` | 2048 | `src/vec.h:45` | int8 内核的输入量化缓冲上界 |
| `WEIGHT_BLOB_VERSION` | 0 | `src/nnet.h:41` | blob 头版本号 |
| `WEIGHT_BLOCK_SIZE` | 64 | `src/nnet.h:42` | blob 头大小 / 数据块对齐粒度 |
| `SPARSE_BLOCK_SIZE` | 32 | `src/parse_lpcnet_weights.c:35` | 一个 8×4 稀疏块的 int8 元素数 |

网络维度相关（**自动生成**，见 `src/rnnoise_data.h:8-38`）：

| 宏 | 值 | 含义 |
|---|---:|---|
| `CONV1_IN_SIZE` | 65 | conv1 每帧输入维 = NB_FEATURES |
| `CONV1_OUT_SIZE` | 128 | conv1 输出通道（cond_size） |
| `CONV1_STATE_SIZE` | 130 | conv1 历史记忆 = 65 × 2 帧 |
| `CONV1_DELAY` | 1 | (k−1)/2，仅作记录用，代码未使用 |
| `CONV2_IN_SIZE` | 128 | conv2 每帧输入维 |
| `CONV2_OUT_SIZE` | 384 | conv2 输出通道（= gru_size） |
| `CONV2_STATE_SIZE` | 256 | conv2 历史记忆 = 128 × 2 帧 |
| `GRU1/2/3_OUT_SIZE` | 384 | GRU 隐层单元数 |
| `GRU1/2/3_STATE_SIZE` | 384 | GRU 状态大小 = 隐层单元数 |
| `DENSE_OUT_OUT_SIZE` | 32 | 频带增益个数 = NB_BANDS |
| `VAD_DENSE_OUT_SIZE` | 1 | VAD 概率 |

派生量：

| 派生量 | 值 | 怎么来的 |
|---|---:|---|
| `cat[]` 长度 | 1536 | `CONV2_OUT_SIZE + GRU1 + GRU2 + GRU3` = 4 × 384（`src/rnn.c:46`） |
| 算法延迟 | **20 ms（2 帧）** | 10 ms 来自 50% 重叠分析窗，10 ms 来自 `delayed_X`；详见 §4.3 |
| 计算量（标准模型） | ≈ 2.88 M MAC/帧 ≈ 288 MMAC/s | 见 §2.1 |
| 计算量（little 模型） | ≈ 1.11 M MAC/帧 ≈ 111 MMAC/s | 同上 |

### 0.3 符号约定

- `M` / `N`：在 `src/nnet_arch.h:136-137` 中 `M = nb_inputs`、`N = nb_outputs`。但 `src/nnet.c:76` 的 `N` 指 GRU 隐层大小。看代码时注意区分。
- `nb_inputs` / `nb_outputs`：权重矩阵按 **(输入, 输出)** 的**行主序**存放，即 `w[in * nb_outputs + out]`。这与 PyTorch 的 `nn.Linear.weight`（形状 `(out, in)`）**正好转置**，导出时由 `wexchange/c_export/common.py:271-272` 转置。
- `z` / `r` / `h`：GRU 的更新门（update）、重置门（reset）、候选状态（new/candidate）。PyTorch 的门顺序是 **r, z, n**，C 侧是 **z, r, h**，导出时重排（见 §2.7）。

---

## 1. 网络输入：65 维特征是怎么来的

### 1.1 输入向量的分段构成

`float features[NB_FEATURES]` 在 `src/denoise.c:464` 声明，由 `rnn_compute_frame_features()`（`src/denoise.c:347-398`）填满，直接喂给 `compute_rnn()`（`src/denoise.c:476`）。

| 下标 | 维度数 | 内容 | 计算代码位置 |
|---:|---:|---|---|
| `0 .. 31` | 32 | **频带对数能量的 DCT**（倒谱系数）。第 0 维再 −12，第 1 维再 −4 | `src/denoise.c:382-388` 取对数；`src/denoise.c:394` 做 DCT；`src/denoise.c:395-396` 平移 |
| `32 .. 63` | 32 | **归一化基音相关的 DCT** | `src/denoise.c:376-377` 归一化；`src/denoise.c:378` 做 DCT |
| `64` | 1 | **基音周期**：`0.01 × (pitch_index − 300)` | `src/denoise.c:379` |

注意填充顺序：先填 32..63（`src/denoise.c:378`）和 64（`:379`），最后才填 0..31（`:394`）。读代码时容易误以为是从低到高顺序填充的。

### 1.2 逐步推导

#### 第 1 步：拼窗 + FFT（`src/denoise.c:332-345`）

```c
RNN_COPY(x, st->analysis_mem, FRAME_SIZE);              // 前半 = 上一帧 480 点
for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];   // 后半 = 本帧 480 点
RNN_COPY(st->analysis_mem, in, FRAME_SIZE);             // 记住本帧，下一帧当"上一帧"
apply_window(x);                                        // :338
forward_transform(X, x);                                // :339
```

`apply_window()`（`src/denoise.c:219-225`）用的是 **Vorbis 窗**：

```c
x[i] *= rnn_half_window[i];
x[WINDOW_SIZE - 1 - i] *= rnn_half_window[i];
```

`rnn_half_window` 由 `src/dump_rnnoise_tables.c:84-89` 生成：

```
half_window[i] = sin( 0.5π · sin²( 0.5π·(i+0.5)/480 ) )
```

两端趋近 0、中间（第 479/480 采样处）趋近 1。它满足 `w[i]² + w[479−i]² = 1`，这正是 50% 重叠相加能**完美重建**的条件（见 §4.4）。表数据在 `src/rnnoise_tables.c:570`。

`forward_transform()`（`src/denoise.c:186-198`）做 960 点实数 FFT，只保留前 `FREQ_SIZE = 481` 个正频率点。

#### 第 2 步：频带能量 Ex[]（`src/denoise.c:90-113`）

32 个频带的边界由 `eband20ms[NB_BANDS+2]` 给出（`src/denoise.c:63-65`），单位不是 Hz 而是"FFT bin"，最后一档是 400（对应 400/960×48000 = 20 kHz）。

`compute_band_energy()` 的要点：

```c
for (j=0;j<band_size;j++) {
    float frac = (float)j/band_size;
    tmp = SQUARE(X[...].r) + SQUARE(X[...].i);
    sum[i]   += (1-frac)*tmp;     // 线性分配到相邻两个频带
    sum[i+1] +=      frac*tmp;
}
```

即**每个 FFT bin 的能量按线性权重分摊给相邻两个频带**——等效于"三角滤波器组"，但实现上是"散射（scatter）"而非"gather"。这让 32 维能量随频谱微小移动平滑变化，网络更好学。

首尾两频带做了补偿（`src/denoise.c:107-108`）：

```c
sum[1] = (sum[0]+sum[1])*2/3;
sum[NB_BANDS] = (sum[NB_BANDS]+sum[NB_BANDS+1])*2/3;
```

因为第 0 个和第 `NB_BANDS+1` 个"虚拟频带"只收到一半的散射，这里按 2/3 系数补回来（经验值，不是严格的 1/2）。

> **顺带一个真实行为**：`eband20ms[33] = 400` 而 `FREQ_SIZE = 481`。`interp_band_gain()`（`src/denoise.c:140-154`）先 `memset(g, 0, FREQ_SIZE)`，只填 `[0, 400)` 区间，所以 **20 kHz–24 kHz 这 81 个频点的增益恒为 0** —— RNNoise 实际上对输出做了 20 kHz 硬低通。训练和推理都是这么处理的，不影响质量，但排查"高频没了"时要知道这一点。

#### 第 3 步：基音周期（`src/denoise.c:359-370`）

```c
RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);   // 历史左移 480
RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);              // 本帧填入尾部
rnn_pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);                          // 2 倍降采样到 24 kHz
rnn_pitch_search(pitch_buf + (PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
                 PITCH_MAX_PERIOD - 3*PITCH_MIN_PERIOD, &pitch_index);
pitch_index = PITCH_MAX_PERIOD - pitch_index;                                     // 翻转成"周期"
gain = rnn_remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
                           PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
```

- 在 24 kHz 降采样域做互相关搜索，搜索窗 20 ms；
- `rnn_remove_doubling()`（`src/pitch.h:45-46`）用来**防止倍频/半频错误**（例如把 100 Hz 判成 200 Hz）。它结合上一帧的周期和增益做时间连续性判断；
- 最终 `pitch_index` 是 **48 kHz 域的基音周期采样数**，理论范围约 [60, 768]。

#### 第 4 步：基音窗的频谱 P[] 与互相关（`src/denoise.c:371-377`）

```c
for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];   // 往回退 pitch_index 个点
apply_window(p);
forward_transform(P, p);          // P = 一个基音周期之前的同一段信号的频谱
compute_band_energy(Ep, P);       // 基音窗的频带能量
compute_band_corr(Exp, X, P);     // X 与 P 的频带互相关
for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);      // 归一化
```

`compute_band_corr()`（`src/denoise.c:115-138`）和 `compute_band_energy()` 结构几乎一样，只是把 `|X|²` 换成复数点积的实部（`src/denoise.c:126-127`）。

归一化式 `Exp[i] / sqrt(0.001 + Ex[i]·Ep[i])` 是标准的**归一化互相关（coherence）**，结果落在 [−1, 1]：`1` 表示"这个频带相隔一个基音周期后完全一样"，`0` 表示"完全无关"。`.001` 防除零。

#### 第 5 步：DCT（`src/denoise.c:160-170`）

```c
static void dct(float *out, const float *in) {
  for (i=0;i<NB_BANDS;i++) {
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) sum += in[j] * rnn_dct_table[j*NB_BANDS + i];
    out[i] = sum*sqrt(2./22);
  }
}
```

查表 `rnn_dct_table[32×32]`（`src/rnnoise_tables.c:669`）由 `src/dump_rnnoise_tables.c:91-97` 生成：

```
dct_table[i*32 + j] = cos((i+0.5) * j * π / 32)，  j == 0 时再乘 sqrt(0.5)
```

代码里索引是 `table[j*32 + i]`（`j` = 输入下标，`i` = 输出下标），所以实际计算的是

```
out[i] = Σ_j  in[j] · cos( (j+0.5) · i · π / 32 ) · sqrt(2/22)
```

`i = 0` 时 `cos(0) = 1`，即**第 0 个 DCT 系数就是 32 个值的加权和（直流/总能量）**——这也解释了为什么第 0 维需要做 −12 的中心化。

> **`sqrt(2./22)` 是个历史 bug（无害）**：22 是 2017 年初版 RNNoise 的频带数，现在 `NB_BANDS = 32`。详见 §5.2。

#### 第 6 步：对数、动态范围压缩、平移（`src/denoise.c:380-396`）

```c
logMax = -2; follow = -2;
for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));   // 下限：不低于 follow-1.5，也不低于 logMax-7
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);                   // 峰值跟随器，每帧最多衰减 1.5
    E += Ex[i];
}
...
dct(features, Ly);
features[0] -= 12;
features[1] -= 4;
```

两个限制器的作用：

- `follow`：**峰值跟随器**，每帧最多衰减 1.5（对数域，即 31.6 倍）。效果是"突发噪声导致能量骤降时，特征不跟着暴跌"，避免网络在瞬态处误判。
- `logMax - 7`：把动态范围硬性限制在 7 个数量级（70 dB）以内，避免极静音段把 `log10` 的数值拉到 −∞。

### 1.3 直觉解释（给 C 程序员）

**为什么要用 DCT？**

相邻频带的能量高度相关——第 5 个频带有能量，第 6 个大概率也有。直接把这 32 个相关数字喂给网络，网络得自己学会"哦它们是一回事"，浪费容量。

DCT 是一个**可逆的线性变换**，作用类似 MP3/JPEG：把 32 个高度相关的数重排成"从粗到细"的 32 个系数——第 0 个是总能量，第 1 个是整体的"倾斜度"，后面的才是细节。好处有二：

1. **去相关**：网络看到的每一维更独立，更好学；
2. **能量集中**：信息挤在前面几维，后面的系数接近 0。

关键是它**不损失信息**（32→32 满秩），所以这不叫降维，叫"换个更好学的坐标系"。

**为什么基音相关能帮降噪？**

浊音（元音）的频谱是**谐波梳**——能量集中在基频的整数倍上，而且相隔一个基音周期后波形几乎重复。平稳噪声（风扇、马路）没有这个性质。

所以 `Exp[i] ≈ 1` 意味着"这个频带在一个周期后没变"→ 强证据表明这是**周期性语音**，应该保留；`Exp[i] ≈ 0` → 大概率是噪声，可以放心压掉。

这给网络提供了一条**纯 DSP 算出来的先验线索**，不需要网络自己从能量里推断周期性。这是 RNNoise 用很小的网络（约 2.9 M MAC/帧）就能打过大网络的关键原因之一。

### 1.4 数值尺度与静音短路

| 量 | 典型范围 | 说明 |
|---|---|---|
| `Ex[i]`（频带能量） | 1e-2 … 1e4 | 加窗 960 点 FFT 后按频带求和，尺度随音量大幅变化 |
| `Ly[i]`（对数能量） | −2 … 4 | `log10(1e-2 + Ex)`，再被 `follow` / `logMax` 夹紧 |
| `features[0..31]` | 约 −12 … 2 | DCT 后 × `sqrt(2/22)` ≈ 0.3015，第 0/1 维再减 12 / 4 |
| `Exp[i]`（归一化基音相关） | −1 … 1 | 归一化互相关 |
| `features[32..63]` | 约 −1 … 1 | DCT 后 × 0.3015 |
| `features[64]` | −2.4 … 4.68 | `0.01 × (pitch_index − 300)`，`pitch_index ∈ [60, 768]` |

**静音短路**（`src/denoise.c:389-393`）：

```c
if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
}
```

`E = Σ Ex[i]` 是全频带总能量。低于 0.04 就认为没声音，直接返回 `silence = 1`。

后果链（`src/denoise.c:474-496`）：静音时**整个 `if (!silence)` 块被跳过**，包括 `compute_rnn()`、基音梳状滤波、增益应用。于是：

- **神经网络状态（3 个 GRU 的 1152 个 float）完全不更新**——这正是注释说的 "avoid messing up the state"。拿接近 0 的输入反复喂 GRU，隐藏状态会缓慢漂移到某个奇怪的值，等真正来语音时前几百毫秒就废了。
- 输出 = `frame_synthesis(st, out, st->delayed_X)`（`src/denoise.c:496`），即**原始上一帧频谱的重建**。因为 OLA 窗满足完美重建条件，此时输出 ≈ 原始输入（延迟 10 ms），也就是"静音时原样透传，只是没做降噪"。
- 注意 `!TRAINING`：训练时（`TRAINING=1`）不短路，因为训练需要拿到一致的特征。

### 1.5 ⚠️ `src/rnn.c:41` 的 `INPUT_SIZE 42` 是错的

```c
41:#define INPUT_SIZE 42
...
47:  /*for (int i=0;i<INPUT_SIZE;i++) printf("%f ", input[i]);printf("\n");*/
```

**这是一个残留的死宏，真实输入维度是 65。** 理由：

1. 全仓库搜索 `INPUT_SIZE`，只有 2 处命中：`src/rnn.c:41` 的定义，和 `src/rnn.c:47` 一行**已被注释掉的调试 printf**。没有任何逻辑使用它。
2. 真实维度由 `src/rnnoise_data.h:10` 的 `CONV1_IN_SIZE 65` 决定，并写死在自动生成的 `init_rnnoise()` 的 `linear_init(..., 195, 128)` 调用里（195 = 3 × 65）。
3. 训练侧也确认是 65：`torch/rnnoise/train_rnnoise.py:74` 的 `dim = 98`、`:88` 的 `data[:, :, :65]`，以及 `src/dump_features.c:487` 写入 `NB_FEATURES` 个 float。

`42` 是 2017 年初版 RNNoise 的特征维度（当时频带划分不同）。**不要用它做任何事**，包括不要用它给输入数组开空间。

> **给读者的判断规则**：判断一个常量是否生效，看它出现在 `#define` + 注释里，还是出现在函数实参里。本项目的维度真值来源只有两处：`src/rnnoise_data.h` 的宏、`init_rnnoise()` 里 `linear_init()` 的最后两个实参。

---

## 2. 网络层结构

### 2.1 层结构总表

维度全部来自自动生成的 `src/rnnoise_data.h:8-38` 与 `src/rnnoise_data.c` 的 `init_rnnoise()`（724217-724229 行）。

| 层名 | 类型 | 输入维 | 输出维 | 激活 | 权重数组前缀 | C 代码位置 | PyTorch 定义位置 |
|---|---|---:|---:|---|---|---|---|
| `conv1` | Conv1d k=3, valid | 195 (=3×65) | 128 | tanh | `conv1_` | `src/rnn.c:48` | `torch/rnnoise/rnnoise.py:66` |
| `conv2` | Conv1d k=3, valid | 384 (=3×128) | 384 | tanh | `conv2_` | `src/rnn.c:49` | `torch/rnnoise/rnnoise.py:67` |
| `gru1_input` | Linear | 384 | 1152 (=3×384) | 见 §2.3 | `gru1_input_` | `src/rnn.c:50` | `torch/rnnoise/rnnoise.py:68` |
| `gru1_recurrent` | Linear + diag | 384 | 1152 | 见 §2.3 | `gru1_recurrent_` | `src/rnn.c:50` | 同上 |
| `gru2_input` / `gru2_recurrent` | 同上 | 384 | 1152 | — | `gru2_*` | `src/rnn.c:51` | `torch/rnnoise/rnnoise.py:69` |
| `gru3_input` / `gru3_recurrent` | 同上 | 384 | 1152 | — | `gru3_*` | `src/rnn.c:52` | `torch/rnnoise/rnnoise.py:70` |
| `dense_out` | Linear | 1536 | 32 | sigmoid | `dense_out_` | `src/rnn.c:56` | `torch/rnnoise/rnnoise.py:71` |
| `vad_dense` | Linear | 1536 | 1 | sigmoid | `vad_dense_` | `src/rnn.c:57` | `torch/rnnoise/rnnoise.py:72` |

> **⚠️ 常见坑**：`torch/rnnoise/rnnoise.py:59` 的默认构造参数是 `gru_size=256`，但**发布的官方模型是 384**。
> 权威值来自 checkpoint：`torch/rnnoise/train_rnnoise.py:107` 把 `{'cond_size':..., 'gru_size':...}` 写进 `model_kwargs`，而训练脚本的默认值是 `--cond-size 128` / `--gru-size 384`（`train_rnnoise.py:48-49`）。
> 也就是说 **`rnnoise.py` 里的 256 是过时的默认值，不可信**；`train_rnnoise.py` 的 384 才是对的。

**每帧计算量（乘加数）**：

| 层 | MAC 数 |
|---|---:|
| conv1 | 195 × 128 = 24,960 |
| conv2 | 384 × 384 = 147,456 |
| GRU × 3（input + recurrent） | 3 × 2 × 384 × 1152 = 2,654,208 |
| dense_out | 1536 × 32 = 49,152 |
| vad_dense | 1536 × 1 = 1,536 |
| **合计** | **2,877,312** |

48 kHz、100 帧/秒 → 约 **288 MMAC/s（≈ 576 MFLOPS）**。little 模型（GRU 稀疏 1/3）降到约 **1,107,840 MAC/帧（≈ 111 MMAC/s）**。

### 2.2 `compute_rnn()` 逐行执行流

`src/rnn.c:44-60`

```c
44: void compute_rnn(const RNNoise *model, RNNState *rnn, float *gains, float *vad, const float *input, int arch) {
45:   float tmp[MAX_NEURONS];                       // 1024，实际只用前 384
46:   float cat[CONV2_OUT_SIZE + GRU1_OUT_SIZE + GRU2_OUT_SIZE + GRU3_OUT_SIZE];  // 384*4 = 1536
...
}
```

| 步骤 | 代码行 | 操作 | 张量变化 |
|---|---|---|---|
| 1 | `src/rnn.c:48` | `conv1` | 输入 `input[65]` + `rnn->conv1_state[130]`（2 帧历史）→ `tmp[0..127]`，**tanh** |
| 2 | `src/rnn.c:49` | `conv2` | 输入 `tmp[0..127]` + `rnn->conv2_state[256]`（2 帧历史）→ `cat[0..383]`，**tanh** |
| 3 | `src/rnn.c:50` | `gru1` | 输入 `cat[0..383]`，状态 `rnn->gru1_state[384]` → 原地更新 |
| 4 | `src/rnn.c:51` | `gru2` | 输入 = 上一步的 `gru1_state[384]`，更新 `gru2_state[384]` |
| 5 | `src/rnn.c:52` | `gru3` | 输入 = `gru2_state[384]`，更新 `gru3_state[384]` |
| 6 | `src/rnn.c:53` | 拼接 | `cat[384..767]   ← gru1_state` |
| 7 | `src/rnn.c:54` | 拼接 | `cat[768..1151]  ← gru2_state` |
| 8 | `src/rnn.c:55` | 拼接 | `cat[1152..1535] ← gru3_state` |
| 9 | `src/rnn.c:56` | `dense_out` | `cat[1536]` → `gains[32]`，**sigmoid** |
| 10 | `src/rnn.c:57` | `vad_dense` | `cat[1536]` → `vad[1]`，**sigmoid** |

`cat` 的布局：

```
 idx:  0        384       768       1152      1536
       ├────────┼─────────┼─────────┼─────────┤
       │ conv2  │  gru1   │  gru2   │  gru3   │
       │  384   │   384   │   384   │   384   │
       └────────┴─────────┴─────────┴─────────┘
```

**两个容易踩的细节：**

1. **GRU 是"串联 + 全部拼接"**，不是只取最后一层的输出。第 3 层 GRU 的输出与前两层、以及 conv2 的输出一起喂给输出层，让输出层同时看到"浅层的局部特征"和"深层的长时记忆"。
2. `compute_generic_gru()` 里有 `celt_assert(in != state)`（`src/nnet.c:81`）。步骤 3 传 `cat` 作输入、`gru1_state` 作状态是安全的；步骤 4/5 传的是 `rnn->gru1_state` / `rnn->gru2_state`，此时它们已经是更新后的状态，与 PyTorch 的 `self.gru2(gru1_out, gru2_state)`（`torch/rnnoise/rnnoise.py:104`）语义一致。

### 2.3 GRU 专题

#### 标准 GRU 公式

给 C 程序员的直觉：GRU 就是一个**带"遗忘旋钮"的状态机**。它维护一个隐状态 `h`，每帧做两件事：

1. 算一个"候选新状态" `h̃`；
2. 用旋钮 `z` 在"旧状态"和"候选新状态"之间插值：`h_new = z·h_old + (1−z)·h̃`。

`z = 1` 表示"完全保持旧状态"，`z = 0` 表示"完全换成新候选"。这个插值结构就是**梯度高速公路**，让 RNN 能记住几百毫秒前的事（普通 RNN 记不住，因为梯度连乘会消失）。

数学形式（本实现采用的版本）：

```
z  = sigmoid( W_iz·x + b_iz  +  W_hz·h_prev + b_hz )            更新门 update gate
r  = sigmoid( W_ir·x + b_ir  +  W_hr·h_prev + b_hr )            重置门 reset gate
h̃  = tanh   ( W_in·x + b_in  +  r ⊙ (W_hn·h_prev + b_hn) )     候选状态
h_new = z ⊙ h_prev + (1 − z) ⊙ h̃
```

注意 `r` 只作用于**候选状态里的递归项**（`r ⊙ (W_hn·h_prev + b_hn)`），这正是 PyTorch `nn.GRU` 的语义。

#### 本实现的 `zrh` 内存布局

`src/nnet.c:65-94`：

```c
69:   float zrh[3*MAX_RNN_NEURONS_ALL];                    // 3 × 1024
70:   float recur[3*MAX_RNN_NEURONS_ALL];
76:   N = recurrent_weights->nb_inputs;                    // N = 384（隐层大小）
77:   z = zrh;                                             // zrh[0 .. 383]     更新门
78:   r = &zrh[N];                                         // zrh[384 .. 767]   重置门
79:   h = &zrh[2*N];                                       // zrh[768 .. 1151]  候选状态
82:   compute_linear(input_weights, zrh, in, arch);        // zrh  = W_ih·x + b_ih
83:   compute_linear(recurrent_weights, recur, state, arch); // recur = W_hh·h_prev + b_hh
84:   for (i=0;i<2*N;i++) zrh[i] += recur[i];              // z、r 两段加上递归项
86:   compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID, arch);  // z、r 一起 sigmoid
87:   for (i=0;i<N;i++) h[i] += recur[2*N+i]*r[i];         // h̃ = W_in·x + r ⊙ (W_hn·h + b_hn)
89:   compute_activation(h, h, N, ACTIVATION_TANH, arch);  // h̃ = tanh(...)
90:   for (i=0;i<N;i++) h[i] = z[i]*state[i] + (1-z[i])*h[i];   // 插值
92:   for (i=0;i<N;i++) state[i] = h[i];                   // 写回状态
```

布局图（`N = 384`）：

```
zrh[]:  ┌─────────┬─────────┬─────────┐
        │    z    │    r    │    h̃    │
        │  0..383 │384..767 │768..1151│
        └─────────┴─────────┴─────────┘
         ↑ 前 2N 一起 sigmoid        ↑ 后 N 单独 tanh

recur[]: ┌─────────┬─────────┬─────────┐
        │  W_hz·h  │  W_hr·h │  W_hn·h │
        └─────────┴─────────┴─────────┘
```

#### 公式 ↔ 代码 逐步对应

| 公式 | 代码 | 说明 |
|---|---|---|
| `W_iz·x + b_iz` | `src/nnet.c:82` 得到 `zrh[0..383]` | 输入侧一次算完 3 个门 |
| `W_hz·h + b_hz` | `src/nnet.c:83` 得到 `recur[0..383]` | 递归侧一次算完 3 个门 |
| `z = σ(两者之和)` | `src/nnet.c:84`（累加前 2N）+ `:86` | **z、r 一次 sigmoid 搞定** |
| `h̃ = tanh(W_in·x + b_in + r⊙(W_hn·h + b_hn))` | `src/nnet.c:87` + `:89` | **注意 `h` 段没有在 `:84` 被累加**（只累加了 `2*N` 个），而是在 `:87` 乘上 `r` 之后再加 |
| `h_new = z·h + (1−z)·h̃` | `src/nnet.c:90` | 此处 `state[i]` 仍是旧状态（`:92` 才写回） |

**为什么前 2N 用 sigmoid、后 N 用 tanh？**

- `z`（更新门）和 `r`（重置门）在数学上是**比例系数**：`z` 是"保留旧状态的比例"，`r` 是"让旧状态参与候选计算的比例"。比例必须落在 `[0, 1]`，所以 sigmoid。
- `h̃`（候选新状态）是**内容**，不是比例。GRU 的隐状态本身是有符号的（正值和负值都携带信息），需要一个关于 0 对称、值域 `[-1, 1]` 的压缩函数，所以 tanh。

如果 `h̃` 也用 sigmoid，状态就永远非负，网络的表达能力直接减半。

#### `diag` 对角项的作用

`compute_linear_()` 末尾（`src/nnet_arch.h:153-161`）：

```c
153:   if (linear->diag) {
155:      celt_assert(3*M == N);           // M = nb_inputs = 384, N = nb_outputs = 1152
156:      for (i=0;i<M;i++) {
157:         out[i]     += linear->diag[i]     * in[i];   // 补回 z  门的对角
158:         out[i+M]   += linear->diag[i+M]   * in[i];   // 补回 r  门的对角
159:         out[i+2*M] += linear->diag[i+2*M] * in[i];   // 补回 h̃ 门的对角
160:      }
161:   }
```

`diag` 数组长度 = `nb_outputs` = 1152 = 3 × 384，存放 `W_hh` 的**对角线元素**。这些元素在导出时被**从稀疏矩阵里抽走**（`wexchange/c_export/common.py:108-124` 的 `extract_diagonal()`），矩阵里对应位置置 0。

**为什么要单独拎出来？三个理由：**

1. **精度**：对角元素是"自环增益"——第 i 个隐单元对下一帧自己的影响，直接决定记忆能保持多久。量化误差在这里会被时间放大。单独存成 float 可以绕开 int8 量化。
2. **不被剪枝**：8×4 结构化稀疏会整块（4 输入 × 8 输出）剪掉。不抽出对角的话，一个块被剪就可能顺带干掉对角元素，导致某个隐单元"失忆"。训练侧的 `keep_diagonal=True`（`torch/rnnoise/rnnoise.py:44-49`）也是同一个目的。
3. **更快**：对角项是 O(N) 的逐元素乘加，比走 8×4 稀疏内核便宜。

`celt_assert(3*M == N)` 保证 `diag` 只能用在 GRU 的**递归权重**上（`out` 是 3N 个门输出、`in` 是 N 个状态），所以 `src/nnet_arch.h:154` 的注释写的是 "Diag is only used for GRU recurrent weights."

### 2.4 Conv1d 专题

#### `kernel_size = 3` 是怎么得出的

C 侧代码里没有任何一处直接写 `kernel_size = 3`（`compute_generic_conv1d()` 只看到展开后的 `nb_inputs`），所以这个数是**推导**出来的。有四条互相印证的证据：

**① 训练侧明文定义**（`torch/rnnoise/rnnoise.py:66-67`）：

```python
self.conv1 = nn.Conv1d(input_dim, cond_size, kernel_size=3, padding='valid')
self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=3, padding='valid')
```

**② 从 C 产物的维度反推**：因为 C 侧把时间维展开成特征维，`nb_inputs = kernel_size × 每帧输入维`：

| 层 | `nb_inputs`（`init_rnnoise()` 实参） | 每帧输入维（宏） | 相除 |
|---|---:|---:|---:|
| conv1 | 195 | `CONV1_IN_SIZE` = 65（`src/rnnoise_data.h:10`） | 195 / 65 = **3** |
| conv2 | 384 | `CONV2_IN_SIZE` = 128（`src/rnnoise_data.h:18`） | 384 / 128 = **3** |

**③ 从 `STATE_SIZE` / `DELAY` 宏反推**：生成规则在 `wexchange/c_export/common.py:297-300`。注意权重已由 `:291` 的 `np.transpose(weight, (2,1,0))` 变成 `(k, in_ch, out_ch)`，所以 `weight.shape[0]` 就是 kernel size：

```python
297: #define {NAME}_OUT_SIZE   {weight.shape[2]}                             # out_channels
298: #define {NAME}_IN_SIZE    {weight.shape[1]}                             # in_channels
299: #define {NAME}_STATE_SIZE ({weight.shape[1]} * ({weight.shape[0] - 1}))  # in_ch × (k − 1)
300: #define {NAME}_DELAY     {(weight.shape[0] - 1) // 2}                   # (k − 1) // 2
```

代入 `src/rnnoise_data.h` 的真实值：

- `CONV1_STATE_SIZE = (65 * (2))`（`:12`）→ `k − 1 = 2` → **k = 3**
- `CONV1_DELAY = 1`（`:14`）→ `(k − 1) // 2 = 1` → `k ∈ {3, 4}`，与上一条取交集 → **k = 3**
- `CONV2_STATE_SIZE = (128 * (2))`（`:20`）、`CONV2_DELAY = 1`（`:22`）→ 同结论

**④ 权重元素数自洽**：`conv1_weights_float` = 24,960 = (3×65) × 128；`conv2_weights_int8` = 147,456 = (3×128) × 384。均吻合。

> ①来自训练脚本，②③④来自训练产物。产物由脚本生成，一致是必然；但 **②③④的价值在于：手头只有 `rnnoise_data.h/.c` 而没有训练代码时，依然能完整反推出 kernel size**。
> 反过来，若你改了 `rnnoise.py` 的 `kernel_size` 重新训练导出，`CONV*_STATE_SIZE` 会跟着变成 `in_ch × (k−1)`（如 k=5 则 conv1 变 `65 * (4)`），无需手改 C 代码。

#### 为什么 C 侧是"带 2 帧历史记忆的 Linear"

PyTorch 的 `nn.Conv1d(in, out, kernel_size=3, padding='valid')` 在时间维上滑窗：输出第 `t` 帧 = `Σ_{k=0..2} W[k] · x[t+k]`。训练时（离线、整段序列）这没问题；流式推理时拿不到"后面 2 帧"。

RNNoise 的做法极其简单：**把 3 帧拼成一个长向量，做一次大矩阵乘**。这就是 `compute_generic_conv1d()`（`src/nnet.c:113-123`）：

```c
113: void compute_generic_conv1d(const LinearLayer *layer, float *output, float *mem,
114:                              const float *input, int input_size, int activation, int arch) {
115:   float tmp[MAX_CONV_INPUTS_ALL];                            // 1024
118:   if (layer->nb_inputs!=input_size) RNN_COPY(tmp, mem, layer->nb_inputs-input_size);
119:   RNN_COPY(&tmp[layer->nb_inputs-input_size], input, input_size);
120:   compute_linear(layer, output, tmp, arch);
121:   compute_activation(output, output, layer->nb_outputs, activation, arch);
122:   if (layer->nb_inputs!=input_size) RNN_COPY(mem, &tmp[input_size], layer->nb_inputs-input_size);
123: }
```

以 conv1 为例（`nb_inputs = 195`，`input_size = 65`）：

```
调用前:   mem   = [ 帧 t-2 (65) | 帧 t-1 (65) ]        (= CONV1_STATE_SIZE = 130)
          input = 帧 t (65)

第 118-119 行拼装:
          tmp   = [ 帧 t-2 | 帧 t-1 | 帧 t ]           (195)

第 120 行: output = W(128×195) · tmp + bias            → 128 个输出通道

第 122 行回写:
          mem   = tmp[65 .. 195) = [ 帧 t-1 | 帧 t ]    ← 供下一帧使用
```

**这就是"卷积"的全部秘密**：3 抽头的时间卷积 = 一个输入维 ×3 的全连接层 + 一个滚动的历史缓冲。之所以能这么做，是因为 kernel 只有 3，展开后也就 195×128 的矩阵，比真正的卷积实现简单太多，且能完全复用后面的 `compute_linear_()`（包括 int8 内核）。

权重的行索引约定（与 PyTorch 对齐）：导出时 `np.transpose(w, (2,1,0))` 把 `(out, in, k)` 变成 `(k, in, out)`，再 `reshape(-1, out)`（`wexchange/c_export/common.py:290-293`），所以行号 = `t * in_channels + in_idx`，`t = 0` 对应**最旧**的帧。C 侧 `tmp` 也是"旧的在前"，两边一致。

#### `conv1_state` / `conv2_state` 存什么、多大

| 字段 | 大小 | 内容 |
|---|---:|---|
| `rnn->conv1_state` | 130 float = 520 B | 最近 2 帧的 **65 维输入特征**（即 `features[]` 的历史），不是原始音频 |
| `rnn->conv2_state` | 256 float = 1024 B | 最近 2 帧的 **128 维 conv1 输出** |

大小由 `src/rnnoise_data.h:12` 和 `:20` 给出，生成规则见 `wexchange/c_export/common.py:299`：

```python
#define {NAME}_STATE_SIZE ({in_channels} * ({kernel_size} - 1))
```

即 `每帧输入维 × (kernel_size − 1)`。改 kernel size 时这个宏会自动跟着变。

### 2.5 `LinearLayer` 结构体逐字段

`src/nnet.h:65-75`：

```c
typedef struct {
  const float     *bias;            // [nb_outputs]
  const float     *subias;          // [nb_outputs]
  const opus_int8 *weights;         // int8 量化权重
  const float     *float_weights;   // float 权重
  const int       *weights_idx;     // 8×4 稀疏索引
  const float     *diag;            // [nb_outputs] GRU 递归矩阵对角
  const float     *scale;           // [nb_outputs] int8 反量化缩放
  int              nb_inputs;
  int              nb_outputs;
} LinearLayer;
```

| 字段 | 本项目是否真用 | 说明 |
|---|---|---|
| `bias` | ✅ 全部 10 层都有 | 正常偏置。量化层在 `USE_SU_BIAS` 下会被 `subias` 顶替（`src/nnet_arch.h:144-147`） |
| `subias` | ⚠️ 仅 `conv2` + 6 个 GRU 矩阵有，且**只在 `USE_SU_BIAS` 编译时生效** | 抵消无符号输入量化引入的常数项，详见 §3.4 |
| `weights` | ✅ 量化层（conv2、6 个 GRU 矩阵） | `conv1` / `dense_out` / `vad_dense` 为 `NULL` |
| `float_weights` | ✅ 全部 10 层都有（未量化层是唯一权重） | 量化层的 float 副本是"调试用"，受 `DISABLE_DEBUG_FLOAT` 控制，详见 §3.6 |
| `weights_idx` | ⚠️ 仅 6 个 GRU 矩阵 | conv1 / conv2 / dense_out / vad_dense 为 `NULL`（稠密） |
| `diag` | ⚠️ 仅 3 个 `gru*_recurrent` | 长度 = `nb_outputs` = 1152 |
| `scale` | ⚠️ 仅量化层 | 逐输出神经元一个 float |
| `nb_inputs` / `nb_outputs` | ✅ | 见 `init_rnnoise()` 里传的最后两个参数 |

**关键优先级**（`src/nnet_arch.h:138-149`）：`float_weights` 优先于 `weights`。这直接决定走 float 内核还是 int8 内核，见 §3.6。

### 2.6 `RNNState` 各 state 的含义与真实大小

`src/rnn.h:40-46`：

```c
typedef struct {
  float conv1_state[CONV1_STATE_SIZE];   // 130
  float conv2_state[CONV2_STATE_SIZE];   // 256
  float gru1_state[GRU1_STATE_SIZE];     // 384
  float gru2_state[GRU2_STATE_SIZE];     // 384
  float gru3_state[GRU3_STATE_SIZE];     // 384
} RNNState;
```

| 字段 | 真实大小 | 含义 | 更新时机 |
|---|---:|---|---|
| `conv1_state` | 130 float = 520 B | 最近 2 帧的 65 维输入特征 | `src/nnet.c:122` |
| `conv2_state` | 256 float = 1024 B | 最近 2 帧的 128 维 conv1 输出 | `src/nnet.c:122` |
| `gru1_state` | 384 float = 1536 B | GRU1 隐状态 | `src/nnet.c:92-93` |
| `gru2_state` | 384 float = 1536 B | GRU2 隐状态 | 同上 |
| `gru3_state` | 384 float = 1536 B | GRU3 隐状态 | 同上 |
| **合计** | **1538 float = 6152 B** | | |

这 6 KB 就是"降噪器的记忆"。它不在权重里，所以：

- `rnnoise_init()` 里 `memset(st, 0, sizeof(*st))`（`src/denoise.c:286`）把它清零 → 新建实例从"零记忆"开始；
- **流式处理时绝不能重建 `DenoiseState`**，否则每帧都从头开始，前约 200 ms 会明显失真；
- 不同声道 / 不同音频流必须各用一份 `DenoiseState`。

### 2.7 C 实现 ↔ PyTorch 定义 对照表

| 项目 | PyTorch（`torch/rnnoise/rnnoise.py`） | C 实现 | 差异处理 |
|---|---|---|---|
| conv1 | `nn.Conv1d(65, 128, k=3, padding='valid')`，`weight` 形状 `(128, 65, 3)` | `LinearLayer`，`nb_inputs=195, nb_outputs=128` | 导出时 `np.transpose(w, (2,1,0))` → `(3, 65, 128)`，再 `reshape(-1, 128)`（`wexchange/c_export/common.py:290-293`）。行索引 = `t*65 + in_ch` |
| conv2 | `nn.Conv1d(128, 384, k=3)` | `nb_inputs=384, nb_outputs=384` | 同上 |
| GRU | `nn.GRU(384, 384, batch_first=True)` | 拆成 `gruN_input`（384→1152）和 `gruN_recurrent`（384→1152）两个 `LinearLayer` | 见下两行 |
| GRU 门顺序 | **r, z, n**（`weight_ih_l0` 按此顺序拼） | **z, r, h** | 导出时交换 `[0:N]` 与 `[N:2N]`（`wexchange/c_export/common.py:342-353`） |
| GRU 权重布局 | `weight_ih_l0` 形状 `(3*384, 384)` | 转置成 `(384, 1152)` 行主序 | `weight.transpose()`（`common.py:352-353`） |
| GRU `batch_first` | `True`，输入 `(B, T, 384)` | 无 batch，一次处理 1 帧，状态显式保存在 `gruN_state` | C 侧是 `seq_len = 1` 的流式展开 |
| Linear 权重 | `weight` 形状 `(out, in)` | 行主序 `(in, out)` | 导出时 `weight.transpose()`（`common.py:271-272`） |
| conv 激活 | `torch.tanh(...)`（`rnnoise.py:99-100`） | `ACTIVATION_TANH`（`src/rnn.c:48-49`） | 一致 |
| 输出层激活 | `torch.sigmoid(...)`（`rnnoise.py:107-108`） | `ACTIVATION_SIGMOID`（`src/rnn.c:56-57`） | 一致 |
| 拼接 | `torch.cat([tmp, gru1_out, gru2_out, gru3_out], dim=-1)`（`rnnoise.py:106`），此处 `tmp` 是 conv2 输出 | `cat[1536]`，见 §2.2 | 一致 |
| sigmoid / tanh | 精确数学函数 | **多项式近似** `sigmoid_approx` / `tanh_approx`（`src/vec.h:337-356`） | 有微小数值差异；`src/nnet_arch.h:83-99` 的 `HIGH_ACCURACY` 开关可切回精确版（慢） |

**`batch_first` 为什么不影响正确性**：C 侧相当于 `T` 维长度为 1 的循环，每次调用处理一个时间步，隐状态由调用方（`RNNState`）保存。PyTorch 训练时一次喂 2000 帧（`--sequence-length`，`train_rnnoise.py:52`）做 BPTT，两者数学上等价。

---

## 3. 权重数组（本文重点）

### 3.1 总览：数组从哪来

```
  ① 训练                                                        torch/rnnoise/train_rnnoise.py
     先由 dump_features 产出 features.f32（每帧 98 个 float：65 特征 + 32 增益 + 1 VAD）
                                                                src/dump_features.c:487-489
        │
        ▼
     python3 train_rnnoise.py features.f32 out_dir               →  rnnoise_50.pth
        │
        ▼
  ② 导出            python3 dump_rnnoise_weights.py --quantize rnnoise_50.pth rnnoise_c
                                                    torch/rnnoise/dump_rnnoise_weights.py
     遍历 model.named_modules()，按层类型分派（dump_rnnoise_weights.py:69-91）：
       nn.Linear  → wexchange.torch.dump_torch_dense_weights    (torch.py:249)
       nn.Conv1d  → wexchange.torch.dump_torch_conv1d_weights   (torch.py:278)
       nn.GRU     → wexchange.torch.dump_torch_gru_weights      (torch.py:180)
     三者最终都落到 wexchange/c_export/common.py 的 print_linear_layer() (:194)
        │
        ▼
  ③ C 静态数组                              rnnoise_data.c / rnnoise_data.h（由 CWriter 生成）
     - 每个数组：`static const <type> <name>[N] = {...}`
     - 量化层的 float 副本包在 `#ifndef DISABLE_DEBUG_FLOAT` 里（common.py:66, 95）
     - rnnoise_arrays[] 的每个条目包在 `#ifdef WEIGHTS_<name>_DEFINED` 里
       （c_writer.py:148-154 ← common.py:69-73）
     - init_rnnoise() 由 c_writer.py:159-166 生成
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
  ④a 直接编译进 .rodata              ④b 转 blob
     rnnoise_init(model = NULL)          write_weights 可执行程序
     → init_rnnoise(rnnoise_arrays)      → weights_blob.bin   (src/write_weights.c:71-77)
       （src/denoise.c:300）                    │
                                               ▼
                                        rnnoise_model_from_file()
                                        → parse_weights() → init_rnnoise()
                                          （src/denoise.c:291-293）
```

两条路最终汇聚到同一个 `init_rnnoise(RNNoise *model, const WeightArray *arrays)`——**它对"静态数组"和"blob 解析结果"一视同仁**，因为两者都是 `WeightArray[]`。

### 3.2 完整权重数组清单

> 数据来源：`model_version = 0a8755f8...b37` 官方模型包的 `src/rnnoise_data.c`，用脚本解析 `static const` 声明得到。
> **下表全部为实测值，非推断。** 数组总数 **50**（`grep -c "^static const"` 与 `rnnoise_arrays[]` 条目数均为 50，两者一致）。
> 行号列指向 `src/rnnoise_data.c`（该文件约 72.4 万行、78 MB）。

#### conv1（195→128，**不量化**，稠密）

| 数组名 | C 类型 | 元素数 | 字节 | 行号 |
|---|---|---:|---:|---:|
| `conv1_weights_float` | float | 24,960 | 99,840 | `:13` |
| `conv1_bias` | float | 128 | 512 | `:3143` |

24,960 = 195 × 128。

#### conv2（384→384，量化，稠密，**无 idx**）

| 数组名 | C 类型 | 元素数 | 字节 | 行号 | 备注 |
|---|---|---:|---:|---:|---|
| `conv2_weights_int8` | opus_int8 | 147,456 | 147,456 | `:3169` | 已按 8×4 重排，见 §3.5 |
| `conv2_weights_float` | float | 147,456 | 589,824 | `:21612` | **调试副本**，`DISABLE_DEBUG_FLOAT` 下不编译 |
| `conv2_subias` | float | 384 | 1,536 | `:40055` | |
| `conv2_scale` | float | 384 | 1,536 | `:40113` | |
| `conv2_bias` | float | 384 | 1,536 | `:40171` | |

#### `gru{1,2,3}_input`（384→1152，量化，稀疏，**无 diag**）

三个 GRU 结构完全相同、元素数一致，只有行号不同。

| 数组名 | C 类型 | 元素数 | 字节 | 行号（gru1 / gru2 / gru3） |
|---|---|---:|---:|---|
| `gruN_input_weights_int8` | opus_int8 | 442,368 | 442,368 | `:40229` / `:266047` / `:491865` |
| `gruN_input_weights_float` | float | 442,368 | 1,769,472 | `:95536` / `:321354` / `:547172` |
| `gruN_input_weights_idx` | int | 13,968 | 55,872 | `:150843` / `:376661` / `:602479` |
| `gruN_input_subias` | float | 1,152 | 4,608 | `:152599` / `:378417` / `:604235` |
| `gruN_input_scale` | float | 1,152 | 4,608 | `:152753` / `:378571` / `:604389` |
| `gruN_input_bias` | float | 1,152 | 4,608 | `:152907` / `:378725` / `:604543` |

#### `gru{1,2,3}_recurrent`（384→1152，量化，稀疏，**带 diag**）

| 数组名 | C 类型 | 元素数 | 字节 | 行号（gru1 / gru2 / gru3） |
|---|---|---:|---:|---|
| `gruN_recurrent_weights_diag` | float | 1,152 | 4,608 | `:153061` / `:378879` / `:604697` |
| `gruN_recurrent_weights_int8` | opus_int8 | 442,368 | 442,368 | `:153215` / `:379033` / `:604851` |
| `gruN_recurrent_weights_float` | float | 442,368 | 1,769,472 | `:208522` / `:434340` / `:660158` |
| `gruN_recurrent_weights_idx` | int | 13,968 | 55,872 | `:263829` / `:489647` / `:715465` |
| `gruN_recurrent_subias` | float | 1,152 | 4,608 | `:265585` / `:491403` / `:717221` |
| `gruN_recurrent_scale` | float | 1,152 | 4,608 | `:265739` / `:491557` / `:717375` |
| `gruN_recurrent_bias` | float | 1,152 | 4,608 | `:265893` / `:491711` / `:717529` |

#### `dense_out`（1536→32，**不量化**，稠密）

| 数组名 | C 类型 | 元素数 | 字节 | 行号 |
|---|---|---:|---:|---:|
| `dense_out_weights_float` | float | 49,152 | 196,608 | `:717683` |
| `dense_out_bias` | float | 32 | 128 | `:723837` |

#### `vad_dense`（1536→1，**不量化**，稠密）

| 数组名 | C 类型 | 元素数 | 字节 | 行号 |
|---|---|---:|---:|---:|
| `vad_dense_weights_float` | float | 1,536 | 6,144 | `:723851` |
| `vad_dense_bias` | float | 1 | 4 | `:724053` |

#### 汇总

| 类别 | 元素数 | 字节 | 占比 |
|---|---:|---:|---:|
| `float`（全部） | 2,902,817 | 11,611,268 | 78.7% |
| `opus_int8` | 2,801,664 | 2,801,664 | 19.0% |
| `int`（`*_weights_idx`） | 83,808 | 335,232 | 2.3% |
| **合计** | — | **14,748,164**（14.06 MiB） | 100% |

**但这 14 MiB 不是模型大小。** 其中 float 部分的 **11,206,656 B（10.69 MiB）是量化层的调试副本**（`conv2_weights_float` + 6 个 GRU 的 `*_weights_float`），定义 `DISABLE_DEBUG_FLOAT` 后根本不编译进去。

| 构建方式 | 实际进 `.rodata` 的大小 |
|---|---:|
| float 副本保留（`--enable-dnn-debug-float` 或手搓编译） | 14,748,164 B = 14.06 MiB |
| `DISABLE_DEBUG_FLOAT`（**autotools 默认**，见 §3.6） | **3,541,508 B = 3.38 MiB** |
| └ 其中 int8 权重 | 2,801,664 B |
| └ 其中 idx | 335,232 B |
| └ 其中 float（bias / scale / subias / diag + 3 个未量化层） | 404,612 B |

### 3.3 命名规范

`rnnoise_arrays[]`（`src/rnnoise_data.c:724061` 起，由 `wexchange/c_export/c_writer.py:147-156` 生成）里每个条目的名字由 `print_linear_layer()`（`wexchange/c_export/common.py:221-227`）按固定模板拼出：

```python
bias_name         = "{name}_bias"                 # bias 存在时
subias_name       = "{name}_subias"               # quantize 时
scale_name        = "{name}_scale"                # quantize 时
idx_name          = "{name}_weights_idx"          # sparse 时
float_weight_name = "{name}_weights_float"        # 总是
int_weight_name   = "{name}_weights_int8"         # quantize 时
diag_name         = "{name}_weights_diag"         # sparse 且 diagonal 时
```

`{name}` 由 `dump_rnnoise_weights.py:71-79` 从 `model.named_modules()` 的模块名取，`.` 换成 `_`：

| PyTorch 模块名 | 导出的 `{name}` 前缀 | C 结构字段 |
|---|---|---|
| `conv1` | `conv1_` | `model->conv1` |
| `conv2` | `conv2_` | `model->conv2` |
| `gru1` | `gru1_input_` 和 `gru1_recurrent_` | `model->gru1_input` / `model->gru1_recurrent` |
| `gru2` / `gru3` | 同上 | 同上 |
| `dense_out` | `dense_out_` | `model->dense_out` |
| `vad_dense` | `vad_dense_` | `model->vad_dense` |

**GRU 拆成两个 `LinearLayer`** 是 `wexchange/c_export/common.py:357-358` 干的：

```python
print_linear_layer(writer, name + "_input",     weight,           bias, ...)
print_linear_layer(writer, name + "_recurrent", recurrent_weight, recurrent_bias, ...)
```

**速记规则**：

```
<层名>_weights_float     ← 总是存在（未量化层的唯一权重）
<层名>_weights_int8      ← 量化层才有
<层名>_weights_idx       ← 稀疏层才有（本项目仅 6 个 GRU 矩阵）
<层名>_weights_diag      ← 稀疏且带对角的才有（仅 gruN_recurrent）
<层名>_bias              ← 总是存在
<层名>_subias            ← 量化层才有
<层名>_scale             ← 量化层才有
```

对应的 C 数组表条目（`src/rnnoise_data.c:724062` 起的模式）：

```c
#ifdef WEIGHTS_conv2_weights_int8_DEFINED
    {"conv2_weights_int8",  WEIGHTS_conv2_weights_int8_TYPE, sizeof(conv2_weights_int8), conv2_weights_int8},
#endif
```

`WEIGHTS_<name>_DEFINED` 与 `WEIGHTS_<name>_TYPE` 由 `print_vector()` 写在每个数组定义之前（`wexchange/c_export/common.py:69-73`）。类型取值见 `src/nnet.h:50-53`：

| 常量 | 值 | 本项目用到 |
|---|---:|---|
| `WEIGHT_TYPE_float` | 0 | ✅ bias / scale / subias / diag / `*_weights_float` |
| `WEIGHT_TYPE_int` | 1 | ✅ `*_weights_idx` |
| `WEIGHT_TYPE_qweight` | 2 | ❌ |
| `WEIGHT_TYPE_int8` | 3 | ✅ `*_weights_int8` |

### 3.4 int8 量化布局

#### 缩放粒度

**逐输出神经元（per-output-neuron）一个 float**，在 C 的 `(in, out)` 行主序布局里就是"逐列"。这从 `linear_init()` 的校验直接可见（`src/parse_lpcnet_weights.c:170-172`）：

```c
if (weights != NULL) {
    if ((layer->scale = find_array_check(arrays, scale, nb_outputs*sizeof(layer->scale[0]))) == NULL) return 1;
}
```

所以 `conv2_scale[384]`、`gru1_input_scale[1152]`；`conv1` / `dense_out` / `vad_dense` 没有 scale（未量化）。

#### scale 怎么算出来的

`wexchange/c_export/common.py:175-188`：

```python
def compute_scaling(weight):                                    # weight 形状 (n_in, n_out)
    weight_max_abs = np.max(np.abs(weight), axis=0)             # 逐输出列取最大绝对值
    weight_max_sum = np.max(np.abs(weight[0::2] + weight[1::2]), axis=0)   # 相邻输入两两求和后的最大绝对值
    scale_max = weight_max_abs / 127
    scale_sum = weight_max_sum / 129                            # 129 = 127 + 2，留 2 个单位余量
    scale = np.maximum(scale_max, scale_sum)
    return scale
```

两个约束的含义：

- `scale_max`：保证**单个权重**量化后落进 int8 范围；
- `scale_sum`：保证**相邻两个输入的权重之和**也在范围内（留 2 单位余量）。因为 int8 内核（`src/vec.h:269-276`）一次取 4 个输入做乘加，约束两两项之和能压住量化误差的上界。

#### 量化 / 反量化的完整链路

**导出侧**（`wexchange/c_export/common.py:126-132, 239-249`）：

```python
Aq = np.clip(np.round(weight / scale), -128, 127)      # weight_q = round(w / scale)
print_vector(writer, Aq, name + "_weights_int8", dtype='opus_int8', reshape_8x4=True)

subias      = bias - np.sum(weight_q * scale, axis=0)  # ← 见下文 USE_SU_BIAS
final_scale = scale / 127                              # ← 存进 C 的 {name}_scale
```

**C 侧**（`src/vec.h:248-281`，非 `USE_SU_BIAS` 分支）：

```c
for (i=0;i<cols;i++) x[i] = (int)floor(.5+127*_x[i]);  // 输入量化：x_q = round(127 · x)
...
y[0] += w[0]*xj0 + w[1]*xj1 + w[2]*xj2 + w[3]*xj3;     // int8 × int8 → int32 累加
...
for (i=0;i<rows;i++) out[i] *= scale[i];               // 反量化
```

代入验证：

```
out = Σ ( w_q · 127·x ) · ( scale / 127 )
    = Σ ( round(w/scale) · x · scale )
    ≈ Σ w · x                                          ✓
```

> **关键推论：`x_q = round(127·x)` 隐含假设输入落在 [−1, 1]。**
> 这正是 `unquantized` 列表的核心依据，见下文。

#### ⚠️ `SCALE (128.f*127.f)` 是什么？—— 一个死常量

它定义于 `src/vec.h:384`、`src/vec_avx.h:879`、`src/vec_neon.h:363`。用 `grep -rn "\bSCALE\b" src/` 验证：**全库只有这 3 处 `#define`，0 处使用**。

历史含义是"输入量化粒度 1/128 × 权重满量程 127"的联合缩放常数，在 LPCNet 里用于把浮点累加结果一次性还原。RNNoise 改成了**逐层、逐输出神经元**的 `{name}_scale` 数组（精度更好），但常量没删。

> **结论：读代码时看到 `SCALE` 可以直接忽略。真正的缩放完全由 `{name}_scale` 数组承载。**

#### `USE_SU_BIAS` 与 `subias`

`src/vec_avx.h:41` 和 `:881` 定义了 `USE_SU_BIAS`（AVX2 路径），此时用**无符号**输入（`src/vec.h:188`）：

```c
for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);     // 注意多了一个 127 的偏移
```

为什么要加偏移？因为 AVX2 的 `opus_mm256_dpbusds_epi32` 做的是 **unsigned × signed** 点积，把输入移到 `[0, 254]` 才能用满这条指令的吞吐。

代价是累加结果里多出一个常数项：

```
out = Σ w_q · (127 + 127·x) · (scale/127)
    = Σ w·x  +  scale · Σ w_q
                       └─────┬─────┘
                        多余的常数项
```

这个常数项正好被 `subias = bias − Σ(w_q·scale)` 抵消。切换发生在 `compute_linear_()`（`src/nnet_arch.h:144-147`）：

```c
/* Only use SU biases on for integer matrices on SU archs. */
#ifdef USE_SU_BIAS
     bias = linear->subias;
#endif
```

两条路径对照：

| 路径 | 输入量化 | 用哪个偏置 | 代码位置 |
|---|---|---|---|
| 有 `USE_SU_BIAS`（AVX2） | `127 + round(127·x)`，无符号 | `subias` | `src/vec.h:182-216` |
| 无 `USE_SU_BIAS`（C / SSE4.1 / NEON） | `round(127·x)`，有符号 | `bias` | `src/vec.h:248-311` |

所以：**`subias` 不是死数据，它是 AVX2 无符号量化方案的配套偏置**；只是 C / NEON 路径用不到它。

#### 不量化的三个层及其原因

`torch/rnnoise/dump_rnnoise_weights.py:15`：

```python
unquantized = [ 'conv1', 'dense_out', 'vad_dense' ]
```

结合前面的 `[-1, 1]` 假设，理由一目了然：

| 层 | 输入是什么 | 值域 | 能否量化 |
|---|---|---|---|
| **conv1** | 65 维原始特征（DCT 系数 + 基音） | 约 **[−12, +5]**（见 §1.4：`features[0..31]` 平移后 −12..2、`features[64]` −2.4..4.7，整体不在 [−1,1]） | ❌ **绝对不行**，×127 会严重截断 |
| **conv2** | conv1 的 tanh 输出 | **[−1, 1]** | ✅ |
| **gru*_input** | conv2 的 tanh 输出 / GRU 状态 | **[−1, 1]** | ✅ |
| **gru*_recurrent** | GRU 状态（tanh 输出的凸组合） | **[−1, 1]** | ✅ |
| **dense_out / vad_dense** | `cat` = conv2(tanh) ‖ 3×GRU 状态 | **[−1, 1]** | ✅ 可以，但没做 |

- **conv1 是唯一"必须"用 float 的层**——这是硬约束，不是性能选择。这也解释了为什么 `conv1_weights_float` 是**唯一不受 `#ifndef DISABLE_DEBUG_FLOAT` 保护**的 float 数组（它是主力，不是调试副本）。
- **dense_out / vad_dense 不量化是"性价比"取舍**：它们直接产生最终增益，量化误差会变成可听的"增益台阶"；而参数量只有 50,688 个（占总量约 1.7%），省下来的收益远小于音质损失。

> **改这里要小心**：这个列表是**硬编码**的，脚本自己的 docstring（`dump_rnnoise_weights.py:17-25`）也警告 "Modify this script manually if adjustments are needed."。给模型加新层而没更新这个列表的话，新层会被默认量化。

### 3.5 8×4 结构化稀疏布局

#### 块的含义

- **块 = 8 个输出 × 4 个输入 = 32 个 int8**（`SPARSE_BLOCK_SIZE 32`，`src/parse_lpcnet_weights.c:35`）。
- 名字里的 "8×4" 是 **(输出方向 8) × (输入方向 4)**。

⚠️ **这个方向极易搞反。** 权威依据在 `find_idx_check()`（`src/parse_lpcnet_weights.c:98-121`）：

```c
while (remain > 0) {
    nb_blocks = *idx++;                                  // 本组的块数
    if (remain < nb_blocks+1) return NULL;               // 越界检查
    for (i=0;i<nb_blocks;i++) {
      int pos = *idx++;
      if (pos+3 >= nb_in || (pos&0x3)) return NULL;      // ← pos 是【输入】下标，必须 4 对齐
    }
    nb_out -= 8;                                         // ← 每组消耗【8 个输出】
    ...
}
if (nb_out != 0) return NULL;                            // 组数必须正好 = nb_outputs / 8
```

**结论：外层循环走输出（步长 8），内层块走输入（步长 4）。**

#### `weights_idx` 的编码格式

```
weights_idx[] 布局（以 gru1_input 为例：nb_in = 384, nb_out = 1152）

  组 0（输出 0..7）        组 1（输出 8..15）           组 143（输出 1144..1151）
 ┌──────────────────┐    ┌──────────────────┐         ┌──────────────────┐
 │ 96               │    │ 96               │         │ 96               │  ← 本组的块数
 │ 0,4,8,12,...,380 │    │ 0,4,8,12,...,380 │   ...   │ 0,4,8,12,...,380 │  ← 每块的输入起始下标
 └──────────────────┘    └──────────────────┘         └──────────────────┘
  共 144 组 = 1152 / 8      每块 pos 必须 4 的倍数，且 pos+3 < 384

 总长度 = 144 × (1 + 96) = 13,968 int                     ← 与实测完全吻合 ✓
 总块数 = 144 × 96 = 13,824，每块 32 个 int8 → 442,368    ← 与 int8 数组长度吻合 ✓
```

**读取伪代码**（对应 `src/vec.h:248-281`）：

```c
idx = weights_idx;
w   = weights;
for (i = 0; i < nb_out; i += 8) {          // 144 组
    colblocks = *idx++;
    for (j = 0; j < colblocks; j++) {
        pos = *idx++;                       // 输入起始下标（4 对齐）
        for (o = 0; o < 8; o++)             // 8 个输出
            out[i+o] += w[o*4+0]*x[pos+0] + w[o*4+1]*x[pos+1]
                      + w[o*4+2]*x[pos+2] + w[o*4+3]*x[pos+3];
        w += 32;                            // 一个块 32 个 int8
    }
}
```

#### 一个块里 32 个 int8 的排布顺序

```
块 = 输出 [i .. i+7] × 输入 [pos .. pos+3]

内存顺序（连续 32 字节）：

  w[ 0.. 3]  →  输出 i+0 对 x[pos+0..pos+3] 的 4 个权重
  w[ 4.. 7]  →  输出 i+1 对 x[pos+0..pos+3] 的 4 个权重
  w[ 8..11]  →  输出 i+2 ...
  ...
  w[28..31]  →  输出 i+7 对 x[pos+0..pos+3] 的 4 个权重

即扁平下标 = 输出偏移 o × 4 + 输入偏移 k      （输出优先）
```

对应 `src/vec.h:269-276` 的展开：

```c
y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);     // y = &out[i]，8 个输出
y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
...
y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
w += 32;
```

#### ⚠️ `*_weights_float` 与 `*_weights_int8` 的块内排布**不一样**

导出侧（`wexchange/c_export/common.py:158-168`）：

```python
158:  idx = np.append(idx, j*4)                        # pos = 输入块号 × 4
162:  vblock = qblock.transpose((1,0)).reshape((-1,))  # (4输入, 8输出) → 转置成 (8,4) → o*4+k
164:  W0 = np.concatenate([W0, block.reshape((-1,))])  # (4输入, 8输出) 原样 → k*8+o
167:  if quantize: print_vector(writer, W, name + '_int8', ..., dtype='opus_int8')
168:  print_vector(writer, W0, name + '_float', ..., dtype='float', debug_float=quantize)
```

| 数组 | 块内索引 | 内核 | 内核为什么要这个顺序 |
|---|---|---|---|
| `*_weights_int8` | `w[o*4 + k]`（**输出优先**） | `sparse_cgemv8x4` / `cgemv8x4` | 4 路点积指令（`vdotq_s32` / `dpbusds`）一次吃 `a[4j..4j+3]·b[4j..4j+3]`，要求**同一输出的 4 个输入连续** |
| `*_weights_float` | `w[k*8 + o]`（**输入优先**） | `sparse_sgemv8x4` / `sgemv` | 一次加载 8 个连续 float（`w[k*8 .. k*8+7]`）直接累加到 8 个输出，要求**同一输入的 8 个输出连续** |

**这是"数据布局跟着 SIMD 内核走"的经典案例**——布局不是数学上的选择，是指令集的选择。两份数组各自适配自己的内核，**是刻意为之，不是 bug**。

（稠密的 `conv2_weights_int8` 同理：导出时 `print_vector(..., reshape_8x4=True)`（`common.py:240`），把 `(n_in, n_out)` 重排成 `(n_in/4, 4, n_out/8, 8) → (n_out/8, n_in/4, 8, 4)`（`common.py:59-61`），与 `cgemv8x4` 的 `for(i: rows step 8) for(j: cols step 4)` 完全对应。）

#### 各门的目标密度表

`torch/rnnoise/rnnoise.py:43-50`：

```python
sparse_params1 = {
    'W_hr' : (0.3, [8, 4], True),
    'W_hz' : (0.2, [8, 4], True),
    'W_hn' : (0.5, [8, 4], True),
    'W_ir' : (0.3, [8, 4], False),
    'W_iz' : (0.2, [8, 4], False),
    'W_in' : (0.5, [8, 4], False),
}
```

| 门 | 目标密度 | 稀疏块 | `keep_diagonal` | 说明 |
|---|---:|---|---|---|
| `W_ir`（输入→重置门） | 0.30 | 8×4 | False | |
| `W_iz`（输入→更新门） | 0.20 | 8×4 | False | 更新门最不重要，剪得最狠 |
| `W_in`（输入→候选状态） | 0.50 | 8×4 | False | 候选状态最重要，保留一半 |
| `W_hr`（递归→重置门） | 0.30 | 8×4 | **True** | |
| `W_hz`（递归→更新门） | 0.20 | 8×4 | **True** | |
| `W_hn`（递归→候选状态） | 0.50 | 8×4 | **True** | |
| **平均** | **1/3 ≈ 0.333** | | | |

稀疏化在训练过程中逐步进行（`torch/rnnoise/rnnoise.py:38-41`）：

```python
sparsify_start    = 6000     # 第 6000 步开始
sparsify_stop     = 20000    # 第 20000 步完成
sparsify_interval = 100      # 每 100 步剪一次
sparsify_exponent = 3        # 三次插值的指数
```

密度按 `alpha + target_density × (1 − alpha)` 从 1.0 渐进降到目标值，`alpha = ((stop − i)/(stop − start))^exponent`（`torch/sparsification/gru_sparsifier.py:119, 131`）。剪枝准则是按 **8×4 块的能量**（块内元素平方和）排序，保留能量最大的那部分块（`torch/sparsification/common.py:63-84`）。

#### ⚠️ 实测：标准模型的稀疏度是 **100%（全稠密）**

**这是最反直觉的一条。** 稀疏格式完全正确，但一个块都没剪掉。

实测证据（脚本逐组走完 6 个 `weights_idx` 数组）：

| 模型 | `weights_idx` 长度 | 组数 | 每组块数 | 总块数 | 密度 |
|---|---:|---:|---|---:|---:|
| **标准模型** | 13,968 = 144 × (1 + 96) | 144 | **恒为 96** | 13,824 | **1.000（100%）** |
| **little 模型** | 4,752 = 144 + 4,608 | 144 | 9 … 65（不等） | 4,608 | **0.333（正好 1/3）** |

自洽性验证：

```
标准模型： 13,824 × 32 = 442,368  ✓ 等于 int8 数组长度
little  ：  4,608 × 32 = 147,456  ✓ 等于 int8 数组长度
little 密度：4,608 / 13,824 = 0.3333
           与 sparse_params1 的平均值 (0.3 + 0.2 + 0.5) / 3 = 1/3 精确一致  ✓
```

原因：训练脚本的稀疏化是 **opt-in** 的（`torch/rnnoise/train_rnnoise.py:57` 的 `--sparse`，在 `:160-161` 才调用 `model.sparsify()`）。发布的标准模型（`rnnoise10Ga_12.pth`）训练时没开这个开关。

**对你的影响：**

1. 标准模型的 6 个 GRU 矩阵虽然带 `weights_idx`，但**每个块都在**，走稀疏内核不会有任何加速（反而多一次 idx 间接寻址的开销）。
2. 想真正吃到稀疏加速，请用 `rnnoise_data_little.c`（`README:121-125`：把 `rnnoise_data_little.c` 改名成 `rnnoise_data.c`），或自己训练时加 `--sparse`。
3. `README:124-125` 说 "the little model has the same size as the regular one (except for the increased sparsity)" —— 指的是**宏定义和数组个数**完全相同（所以同一个二进制可以加载两种 blob），实际 int8 数据量差 3 倍。

#### `keep_diagonal` 与 `diag` 数组

- **训练侧**：`keep_diagonal=True` 时，先把对角元素从矩阵里**减掉**，再对剩下的部分做块剪枝，最后把对角**加回**（`torch/sparsification/common.py:54-61, 84`）。保证对角永不被剪。
- **导出侧**：`extract_diagonal(A)`（`wexchange/c_export/common.py:108-124`）把 `(N, 3N)` 矩阵的 3 个 `N×N` 对角块抽出来拼成长度 `3N` 的 `diag`，原矩阵对应位置置 0。
- **C 侧**：`compute_linear_()` 末尾补回（`src/nnet_arch.h:153-161`，详见 §2.3）。

只有 `gruN_recurrent` 有 `diag`（`wexchange/c_export/common.py:358` 的 `diagonal=recurrent_sparse`），长度 1152 = 3 × 384。

### 3.6 `compute_linear_()` 的 4 条分派路径

`src/nnet_arch.h:130-162`：

```c
138:   if (linear->float_weights != NULL) {
139:     if (linear->weights_idx != NULL) sparse_sgemv8x4(out, linear->float_weights, linear->weights_idx, N, in);
140:     else                             sgemv(out, linear->float_weights, N, M, N, in);
141:   } else if (linear->weights != NULL) {
142:     if (linear->weights_idx != NULL) sparse_cgemv8x4(out, linear->weights, linear->weights_idx, linear->scale, N, M, in);
143:     else                             cgemv8x4(out, linear->weights, linear->scale, N, M, in);
144:     /* Only use SU biases on for integer matrices on SU archs. */
145: #ifdef USE_SU_BIAS
146:     bias = linear->subias;
147: #endif
148:   }
149:   else RNN_CLEAR(out, N);
```

```
                              weights_idx != NULL ?
                        ┌───────────┴───────────┐
                       YES                      NO
              ┌─────────┴─────────┐    ┌────────┴────────┐
      float_weights != NULL ?     │    float_weights != NULL ?
      ┌────────┴────────┐         │    ┌────────┴────────┐
     YES               NO         │   YES               NO
      │                 │         │    │                 │
 sparse_sgemv8x4  sparse_cgemv8x4 │  sgemv          cgemv8x4
  (路径 1)         (路径 3)        │  (路径 2)       (路径 4)
  float 稀疏       int8 稀疏       │  float 稠密     int8 稠密
```

| # | 触发条件 | 调用 | 语义 | 本项目谁走这条 |
|---|---|---|---|---|
| 1 | `float_weights != NULL` 且 `weights_idx != NULL` | `sparse_sgemv8x4()` | float 稀疏 | 6 个 GRU 矩阵（float 副本存在时） |
| 2 | `float_weights != NULL` 且 `weights_idx == NULL` | `sgemv()` | float 稠密 | conv1 / dense_out / vad_dense（**总是**）；conv2（float 副本存在时） |
| 3 | `float_weights == NULL`，`weights != NULL`，`weights_idx != NULL` | `sparse_cgemv8x4()` | int8 稀疏 | 6 个 GRU 矩阵（float 副本被裁掉时） |
| 4 | 同 3 但 `weights_idx == NULL` | `cgemv8x4()` | int8 稠密 | conv2（float 副本被裁掉时） |
| 5 | 两者都 `NULL` | `RNN_CLEAR(out, N)` | 输出全 0 | 正常情况不会发生；见 §3.8 的 `USE_WEIGHTS_FILE` 坑 |

#### ⚠️ 走哪条路取决于**构建方式**

`float_weights` 是否存在由宏 **`DISABLE_DEBUG_FLOAT`** 决定。这条链要追三层：

1. **导出脚本**：`wexchange/c_export/common.py:242` 传 `debug_float=quantize`。量化层的 float 副本因此被包进 `#ifndef DISABLE_DEBUG_FLOAT`（`common.py:66, 95`），而且 `#define WEIGHTS_<name>_DEFINED` **也在同一个 `#ifndef` 里面**（`common.py:69-73`）。
2. **数组表**：`c_writer.py:150-152` 把每个数组条目包在 `#ifdef WEIGHTS_<name>_DEFINED` 里。所以宏一开，float 数组**根本不进 `rnnoise_arrays[]`**。
3. **`linear_init()`**：`opt_array_check()`（`src/parse_lpcnet_weights.c:91-96`）找不到该名字时返回 `NULL` 且 `*error = 0`（不报错）→ `layer->float_weights = NULL` → 走 int8 路径。

而 `DISABLE_DEBUG_FLOAT` 是否定义，取决于你怎么构建（`configure.ac:80-86`）：

```m4
AC_ARG_ENABLE([dnn-debug-float],
              AS_HELP_STRING([--enable-dnn-debug-float], [Use floating-point DNN computation everywhere]),,
  enable_dnn_debug_float=no)

AS_IF([test "$enable_dnn_debug_float" = "no"], [
       AC_DEFINE([DISABLE_DEBUG_FLOAT], [1], [Disable DNN debug float])
])
```

`AC_ARG_ENABLE` 的第 4 个参数是"用户未指定时的默认值" → 默认 `no` → `AS_IF` 条件成立 → **默认就定义 `DISABLE_DEBUG_FLOAT`**。

| 构建方式 | `HAVE_CONFIG_H` | `DISABLE_DEBUG_FLOAT` | GRU / conv2 走哪条 | `.rodata` |
|---|---|---|---|---:|
| `./autogen.sh && ./configure && make`（**官方默认**） | 定义 | **定义** | **int8**（路径 3 / 4） | 3.38 MiB |
| `./configure --enable-dnn-debug-float` | 定义 | 不定义 | float（路径 1 / 2） | 14.06 MiB |
| 手搓 `cc -I src *.c`（不含 config.h） | 不定义 | 不定义 | float（路径 1 / 2） | 14.06 MiB |

> **注意**：`src/rnnoise_data.c:1-3` 是 `#ifdef HAVE_CONFIG_H / #include "config.h" / #endif`。手搓编译时若没定义 `HAVE_CONFIG_H`，`config.h` 不会被包含，`DISABLE_DEBUG_FLOAT` 自然也不生效。这是"手搓编译反而走 float"的根本原因。

**这个设计值得学习**：用"编译期宏 + NULL 指针检查"实现双实现切换，**运行时零开销**——`compute_linear_` 里没有 `if (quantized)` 的分支预测，只是两个指针判断。

> 顺带解释一个常见困惑："为什么我改了 int8 数组，输出没变化？"——因为你编译时 float 副本还在，int8 数组根本没被读。

### 3.7 装配链路

```
rnnoise_init(st, model)                                    src/denoise.c:285-309
  │
  ├── memset(st, 0, sizeof(*st))                           src/denoise.c:286  ← 清空全部 state
  │
  ├──【路径 A：外部模型】model != NULL                     src/denoise.c:288
  │     ├── parse_weights(&list, blob, blob_len)           src/denoise.c:291
  │     │     └── src/parse_lpcnet_weights.c:54-78
  │     │           循环 parse_record()（:37-52）把 blob 拆成 WeightArray[]
  │     │           末尾补一个 name==NULL 的哨兵元素（:76）
  │     ├── init_rnnoise(&st->model, list)                 src/denoise.c:293
  │     └── opus_free(list)                                src/denoise.c:294
  │           ↑ list 可以释放：linear_init 只保存了指向 blob 内部数据的指针
  │
  ├──【路径 B：内建默认模型】model == NULL 且未定义 USE_WEIGHTS_FILE
  │     └── init_rnnoise(&st->model, rnnoise_arrays)       src/denoise.c:300
  │
  └── st->arch = rnn_select_arch()                         src/denoise.c:304

init_rnnoise(model, arrays)          src/rnnoise_data.c（自动生成，724217-724229 行）
  │   依次调用 10 次 linear_init()：
  ├── linear_init(&model->conv1,          ..., "conv1_bias", NULL, NULL,
  │               "conv1_weights_float", NULL, NULL, NULL, 195, 128)
  ├── linear_init(&model->conv2,          ..., "conv2_bias", "conv2_subias", "conv2_weights_int8",
  │               "conv2_weights_float", NULL, NULL, "conv2_scale", 384, 384)
  ├── linear_init(&model->gru1_input,     ..., "gru1_input_weights_idx", NULL,
  │               "gru1_input_scale", 384, 1152)
  ├── linear_init(&model->gru1_recurrent, ..., "gru1_recurrent_weights_idx",
  │               "gru1_recurrent_weights_diag", "gru1_recurrent_scale", 384, 1152)
  ├── ... gru2 / gru3 同构 ...
  ├── linear_init(&model->dense_out,      ..., NULL, "dense_out_weights_float", NULL, NULL, NULL, 1536, 32)
  └── linear_init(&model->vad_dense,      ..., NULL, "vad_dense_weights_float", NULL, NULL, NULL, 1536, 1)

linear_init(layer, arrays, ...)                            src/parse_lpcnet_weights.c:123-176
  ├── 135-141  先把 7 个指针全部置 NULL（防御性编程：后面全靠 != NULL 分派）
  ├── 142-144  bias          ← find_array_check(期望 nb_outputs 个 float)
  ├── 145-147  subias        ← find_array_check（同上）
  ├── 148-157  【稀疏分支】weights_idx ← find_idx_check()
  │                          weights      ← 期望 32 × total_blocks 个 opus_int8
  │                          float_weights← opt_array_check()（缺失不算错）
  ├── 158-166  【稠密分支】weights      ← 期望 nb_inputs × nb_outputs 个
  │                          float_weights← opt_array_check()
  ├── 167-169  diag          ← 期望 nb_outputs 个 float
  ├── 170-172  scale         ← 仅当 weights != NULL；期望 nb_outputs 个 float
  └── 173-175  nb_inputs / nb_outputs 写入，return 0
```

`linear_init()` 有两个值得注意的设计：

1. **长度校验即契约检查。** 每个 `find_array_check` 都传入**期望字节数**。把"模型版本不匹配"这类运行时灾难提前到初始化期：`linear_init` 返回 1 → `init_rnnoise` 返回 1 → `rnnoise_init` 返回 −1 → `rnnoise_create` 返回 `NULL`（`src/denoise.c:316-320`）。**Fail fast，不放过任何静默错误。**
2. **`find_array_check` vs `opt_array_check` 的语义差异**（`src/parse_lpcnet_weights.c:85-96`）：

   | 函数 | 语义 | 谁在用 |
   |---|---|---|
   | `find_array_check` | **必需**数组。缺失 = 失败 | bias / subias / weights / diag / scale |
   | `opt_array_check` | **可选**数组。缺失 OK；存在但大小错 = 失败 | 只有 `float_weights` |

   只有 `float_weights` 用 `opt_array_check`（`:155, :163`），因为它是"调试参考副本"，可以被 `DISABLE_DEBUG_FLOAT` 裁掉。**这正是 int8 路径能生效的开关。**

#### `find_idx_check()` 在防什么

`src/parse_lpcnet_weights.c:98-121`，四道校验：

| 行 | 校验 | 防的是什么 |
|---|---|---|
| `:110` | `remain < nb_blocks+1` | idx 数组被截断（如 blob 传输不完整）时读越界 |
| `:113` | `pos+3 >= nb_in` | 内核会读 `x[pos..pos+3]`，越界会把 `x` 后面的内存当输入，产生垃圾输出甚至 segfault |
| `:113` | `pos & 0x3` | pos 必须 4 对齐。否则稀疏块会重叠或留缝，`total_blocks × 32` 与输入维对不上；也保证 SIMD 载入不跨块 |
| `:119` | `nb_out != 0` | 走完全部数据后组数必须严格等于 `nb_outputs / 8`（GRU：1152/8 = 144）。多了少了都拒绝 |

**为什么值得这么严？** 因为 blob 可以被 `rnnoise_model_from_file()` / `rnnoise_model_from_filename()` 从**外部文件**加载（`include/rnnoise.h:96-118`）。这是一条安全边界：损坏或恶意构造的 blob 不能让库读越界。

### 3.8 blob 二进制格式

#### 结构

`src/nnet.h:55-62`：

```c
typedef struct {
  char head[4];        // magic "DNNw"
  int version;         // WEIGHT_BLOB_VERSION = 0
  int type;            // WEIGHT_TYPE_*
  int size;            // 有效数据字节数
  int block_size;      // 补齐到 64 倍数的块大小
  char name[44];       // 数组名，必须 NUL 结尾
} WeightHead;
```

```
偏移               内容                                 大小
─────────────────────────────────────────────────────────────
0x00   ┌───────────────────────────────────────────┐
       │ WeightHead                                │
       │   char head[4]      = "DNNw"              │   4
       │   int  version      = 0                   │   4
       │   int  type         = WEIGHT_TYPE_*       │   4
       │   int  size         = 实际字节数           │   4
       │   int  block_size   = ceil(size/64)*64    │   4
       │   char name[44]     = NUL 结尾             │  44
0x40   ├───────────────────────────────────────────┤  64 = WEIGHT_BLOCK_SIZE
       │ data（size 字节）                          │
       │ zero padding（block_size − size 字节）     │  block_size
       ├───────────────────────────────────────────┤
       │ 下一个 record ...                          │
       └───────────────────────────────────────────┘

无总目录，顺序扫描，直到 len 耗尽
```

`celt_assert(sizeof(h) == WEIGHT_BLOCK_SIZE)`（`src/write_weights.c:63`）保证头部精确 64 字节：`4 + 4×4 + 44 = 64` ✓。选 64 是为了对齐典型 cache line，让后续数据块 64 字节对齐，便于 SIMD 对齐加载和 mmap 后直接访问。

**标准模型 blob 大小估算**：

```
数据总量          14,748,164 B
50 个头部         × 64  =    3,200 B
50 个块的对齐填充 ≤ 50 × 63 =    3,150 B（实测：6 个 idx 各补 48，dense_out_bias 补 32，vad_dense_bias 补 63 = 383 B）
────────────────────────────────────
合计              ≈ 14,751,747 B ≈ 14.07 MiB
```

#### 写出（`src/write_weights.c:46-69`）

```c
memcpy(h.head, "DNNw", 4);
h.version    = WEIGHT_BLOB_VERSION;
h.type       = list[i].type;
h.size       = list[i].size;
h.block_size = (h.size+WEIGHT_BLOCK_SIZE-1)/WEIGHT_BLOCK_SIZE*WEIGHT_BLOCK_SIZE;   // 向上取整到 64
RNN_CLEAR(h.name, sizeof(h.name));
strncpy(h.name, list[i].name, sizeof(h.name));
h.name[sizeof(h.name)-1] = 0;                      // 强制 NUL 结尾
fwrite(&h, 1, WEIGHT_BLOCK_SIZE, fout);
fwrite(list[i].data, 1, h.size, fout);
fwrite(zeros, 1, h.block_size-h.size, fout);       // 补零
```

注意数组名最长 **43 字符**（`name[44]` 留一个 `'\0'`）；`src/write_weights.c:52-54` 会打印警告，`parse_record` 的 `:42` 会直接拒绝。

`src/write_weights.c:38-44` 有个 hack 值得一提：

```c
/* This is a bit of a hack because we need to build nnet_data.c and plc_data.c without USE_WEIGHTS_FILE,
   but USE_WEIGHTS_FILE is defined in config.h. */
#undef HAVE_CONFIG_H
#ifdef USE_WEIGHTS_FILE
#undef USE_WEIGHTS_FILE
#endif
#include "rnnoise_data.c"
```

`write_weights` 这个工具**需要**静态数组作为转换源，但 `config.h` 里定义了 `USE_WEIGHTS_FILE` 会把数组 `#ifndef` 掉，所以在 include 之前手工 `#undef`。

#### 解析（`src/parse_lpcnet_weights.c:37-78`）

**零拷贝设计**（`:44, :47`）：`array->name` 和 `array->data` **都直接指向 blob 内的地址**，不复制数据。

```c
array->name = h->name;                                              // 指向 blob 内部
array->data = (void*)((unsigned char*)(*data) + WEIGHT_BLOCK_SIZE); // 指向 blob 内部
```

这带来一条硬性约束——**blob 缓冲区必须在 `DenoiseState` 生命周期内保持有效**。`README:107-115` 和 `include/rnnoise.h:96-102` 都强调了这点：

- `rnnoise_model_from_buffer()`（`src/denoise.c:235-242`）只存指针，不拷贝 → **调用方负责生命周期**；
- `rnnoise_model_from_file()`（`src/denoise.c:252-269`）会 `malloc` 一份拷贝（`:262`），`rnnoise_model_free()` 时释放（`:273`）；
- `rnnoise_model_from_filename()`（`src/denoise.c:244-250`）会 `fopen`，`rnnoise_model_free()` 时 `fclose`（`:272`）。

**安全加固**（`:39-43`），5 项校验全部针对"恶意/损坏的 blob"：

```c
if (*len < WEIGHT_BLOCK_SIZE) return -1;                  // 不够一个头
if (h->block_size < h->size) return -1;                   // block_size 必须 >= size
if (h->block_size > *len-WEIGHT_BLOCK_SIZE) return -1;    // 数据区超出剩余长度
if (h->name[sizeof(h->name)-1] != 0) return -1;           // 名字必须 NUL 结尾
if (h->size < 0) return -1;
```

特别注意第 4 项：防止 `find_array_entry()`（`:80-83`）里的 `strcmp` 读到数组外。

#### 静态数组 ↔ blob 的互转关系

```
              rnnoise_data.c（14 MiB 的 .c 源文件）
                            │
            ┌───────────────┴────────────────┐
            │                                │
    编译时链接进 .rodata               write_weights 程序
    （未定义 USE_WEIGHTS_FILE）        （write_weights.c:71-77）
            │                                │
            │                         weights_blob.bin
            │                                │
            │                         parse_weights()
            │                         (parse_lpcnet_weights.c:54)
            │                                │
            └──────────► WeightArray[] ◄─────┘
                                │
                        init_rnnoise()
                                │
                    RNNoise model（10 个 LinearLayer）
```

两者是**同一批数据的两种载体，语义完全等价**（`linear_init` 只认 `WeightArray` 表）。

| | 静态数组（默认） | blob |
|---|---|---|
| 优点 | 编译进可执行文件，无 IO、无解析开销、无外部依赖 | 模型可热更新、可多模型共存、可执行体小 |
| 代价 | 二进制大 14 MiB | 启动时要读文件 + 解析（解析零拷贝，很快） |
| 编译期开关 | `USE_WEIGHTS_FILE`（`src/rnnoise_data.c:7, 724060, 724214`） | |

> ⚠️ **`USE_WEIGHTS_FILE` 的坑**（`src/denoise.c:298-303`）：如果定义了 `USE_WEIGHTS_FILE` 且调用时 `model == NULL`，则**两条分支都被编译掉**，`init_rnnoise` 根本不会被调用 → `st->model` 全是 NULL → `compute_linear_` 走 `src/nnet_arch.h:149` 的 `RNN_CLEAR(out, N)` → **输出全 0（静音）**。所以 `USE_WEIGHTS_FILE` 构建下**必须**传 model。

---

## 4. 输入预处理与输出后处理

### 4.1 预处理链

#### ① 高通滤波（DC 阻断）— `src/denoise.c:471` → `:409-419`

```c
static const float a_hp[2] = {-1.99599, 0.99600};
static const float b_hp[2] = {-2, 1};
...
rnn_biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
```

`rnn_biquad()`（`src/denoise.c:409-419`）用的是直接 II 型转置结构。把递推式转成 z 域可得：

```
H(z) = (1 + b0·z⁻¹ + b1·z⁻²) / (1 + a0·z⁻¹ + a1·z⁻²)
     = (1 − 2z⁻¹ + z⁻²) / (1 − 1.99599z⁻¹ + 0.996z⁻²)
     = (1 − z⁻¹)² / (1 − 1.99599z⁻¹ + 0.996z⁻²)
```

- **分子 `(1 − z⁻¹)²`**：在 `z = 1`（即 0 Hz / DC）有**两个零点**，所以直流增益严格为 0。
- **分母**：判别式 `1.99599² − 4×0.996 = 3.9839761 − 3.984 < 0` → 一对共轭极点，`|z| = √0.996 ≈ 0.998`，角度 `θ = arccos(1.99599 / (2×0.998)) ≈ 2.4×10⁻³ rad` → 转折频率 `f = θ·fs/(2π) ≈ 20 Hz`（末位系数敏感，量级是 20 Hz 左右）。

**为什么要这一步？** 去掉直流偏移和极低频（麦克风底噪、风噪、呼吸声）。这很重要，因为最低的两个频带极窄：`eband20ms[1] = 2`（`src/denoise.c:65`），第一个频带只覆盖 2 个 FFT bin（约 100 Hz 宽）。一点点 DC 漂移就会把 `log10` 特征整个抬高，让网络误判。

#### ② 重叠加窗 — `src/denoise.c:332-338`

```c
RNN_COPY(x, st->analysis_mem, FRAME_SIZE);            // 前半 = 上一帧
for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i]; // 后半 = 本帧
RNN_COPY(st->analysis_mem, in, FRAME_SIZE);           // 记住本帧
apply_window(x);
```

50% 重叠：`WINDOW_SIZE = 2 × FRAME_SIZE`。这是时频分析的标配——单帧 480 点做 FFT 频率分辨率太粗（100 Hz），拼成 960 点后分辨率翻倍（50 Hz），同时重叠保证时间上不漏信息。

`apply_window()`（`src/denoise.c:219-225`）乘 Vorbis 窗，抑制频谱泄漏。窗函数数据由 `src/dump_rnnoise_tables.c:84-89` 生成，存在 `src/rnnoise_tables.c:570`。

#### ③ FFT — `src/denoise.c:186-198`

960 点实数 FFT，只保留前 481 个正频率点（负频率是共轭镜像，冗余）。

### 4.2 后处理链

调用顺序（`src/denoise.c:474-495`）：

| 步 | 代码行 | 干什么 |
|---|---|---|
| 1 | `:476` | `compute_rnn()` → `g[32]`（频带增益）+ `vad` |
| 2 | `:478` | `rnn_pitch_filter()` —— **基音梳状滤波**（后滤波），用 `g` 做门限 |
| 3 | `:479-487` | 增益时间平滑（RT60 = 135 ms）+ 能量补偿 |
| 4 | `:488` | `interp_band_gain()` 把 32 个频带增益插值成 481 个频点增益 `gf[]` |
| 5 | `:490-493` | `delayed_X[i] *= gf[i]`（实部虚部同乘） |
| 6 | `:496` | `frame_synthesis()`：IFFT + 加窗 + 重叠相加 → `out[480]` |
| 7 | `:498-502` | 把本帧的 `X / P / Ex / Ep / Exp` 存进 `delayed_*`，供下一帧用 |

#### 基音梳状滤波 `rnn_pitch_filter()`（`src/denoise.c:421-455`）

```c
429:  for (i=0;i<NB_BANDS;i++) {
435:    if (Exp[i]>g[i]) r[i] = 1;
436:    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
437:    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
439:    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
440:  }
441:  interp_band_gain(rf, r);
442:  for (i=0;i<FREQ_SIZE;i++) {
443:    X[i].r += rf[i]*P[i].r;          // ← 加上"一个基音周期前"的频谱
444:    X[i].i += rf[i]*P[i].i;
445:  }
446:  compute_band_energy(newE, X);
447:  for (i=0;i<NB_BANDS;i++) norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
450:  interp_band_gain(normf, norm);
451:  for (i=0;i<FREQ_SIZE;i++) { X[i].r *= normf[i]; X[i].i *= normf[i]; }
```

**直觉**：`src/denoise.c:443` 做的 `X += rf · P` 在频域上就是**梳状滤波**——把"一个基音周期之前"的频谱按频带比例加回来。因为 `P` 是 `X` 延迟一个基音周期后的版本，两者相加会在基频的整数倍处**相长干涉**、在谐波之间**相消干涉**，结果是：谐波被加强，谐波之间的噪声被抵消。

这是从 CELT / Opus 继承来的经典技巧（pitch postfilter），效果是让浊音听起来更"干净"，代价是可能引入轻微的"金属感"（所以 `rf` 要保守）。

`r[i]` 怎么定（`:435-439`）：

- `Exp[i]` 是实测的**基音相关性**，`g[i]` 是网络给出的**增益**。
- 如果 `Exp[i] > g[i]`：说明"实测周期性比网络想保留的还多" → 网络压过头了 → `r = 1`，全额补回周期性成分。
- 否则按一个 MMSE 形式的比值（赔率比）计算部分补偿量。
- `sqrt(Ex/Ep)`（`:439`）修正当前帧与延迟帧的能量差，避免叠加后音量翻倍。
- 最后 `:447-453` 逐频带把能量**重新归一化回 `Ex[i]`**，保证这个滤波器只改频谱形状、不改响度。

#### 增益时间平滑与能量补偿（`src/denoise.c:479-487`）

```c
480:      float alpha = .6f;
481:      /* Cap the decay at 0.6 per frame, corresponding to an RT60 of 135 ms.
482:         That avoids unnaturally quick attenuation. */
483:      g[i] = MAX16(g[i], alpha*st->lastg[i]);
486:      st->lastg[i] = MIN16(1.f, g[i]*(st->delayed_Ex[i]+1e-3)/(Ex[i]+1e-3));
```

**RT60 = 135 ms 的由来**：

```
α = 0.6 每帧，帧长 = 10 ms
RT60 的定义是"衰减 60 dB 所需的时间"，60 dB 对应幅度比 10^(-60/20) = 0.001
求 n： 0.6^n = 0.001
     n = ln(0.001) / ln(0.6) = -6.9078 / -0.51083 = 13.52 帧
     t = 13.52 × 10 ms = 135.2 ms        ✓
```

**为什么需要它**：网络可能这一帧说"这个频带是噪声"（g = 0.02），下一帧又说"是语音"（g = 0.9）。如果增益直接跳变，会听到"咔咔"的抽水机声（musical noise）。这行限制**下降速度每帧最多乘 0.6**，上升不限（要跟上语音起音）。这是经典的 "gain smoothing / minimum statistics" 技巧。

**能量补偿（`:486`）**：`lastg` 存的不是上一帧的 `g`，而是**按能量比修正过**的 `g`。直觉：假设上一帧能量 `Ex = 1`、增益 0.5，这一帧能量突然涨到 `Ex = 10`（来了一个瞬态）。若还用 0.5，噪声会跟着一起放大漏出来。所以按 `delayed_Ex / Ex` 缩放，让"绝对噪声底"保持连续。`1e-3` 防除零，`MIN16(1.f, ...)` 保证增益不超过 1。

#### 频带 → 频点插值（`src/denoise.c:140-154`）

`interp_band_gain()` 把 32 个频带值插值成 481 个频点值，用的是与 `compute_band_energy()` 的软分带**互逆**的线性插值（`:149` 的 `(1-frac)*bandE[i-1] + frac*bandE[i]`）。首尾两个频带做常数外推（`:152-153`）。

`memset(g, 0, FREQ_SIZE)`（`:142`）后只填 `[0, eband20ms[NB_BANDS]=400)`，所以 **20 kHz 以上增益恒为 0**（见 §1.2 的注解）。

### 4.3 ⚠️ 专节：为什么增益乘的是 `delayed_X` 而不是 `X`

这是整份代码里最值得单独讲的一处设计。

```c
478:    rnn_pitch_filter(st->delayed_X, st->delayed_P, st->delayed_Ex, st->delayed_Ep, st->delayed_Exp, g);
...
490:    for (i=0;i<FREQ_SIZE;i++) {
491:      st->delayed_X[i].r *= gf[i];
492:      st->delayed_X[i].i *= gf[i];
493:    }
496:  frame_synthesis(st, out, st->delayed_X);
498:  RNN_COPY(st->delayed_X, X, FREQ_SIZE);
```

**注意：`g` 是用【本帧】的特征算出来的，却作用于【上一帧】的频谱。**

#### 索引算术（把延迟算清楚）

设第 `n` 次调用时输入的是帧 `n`（`in[0..479]` = 帧 `n`）。

```
第 k 次调用时：
  analysis_mem 里是帧 k-1
  → 分析窗 x = [帧 k-1 | 帧 k]
  → X_k = FFT([帧 k-1, 帧 k])

第 n 次调用时 delayed_X = X_{n-1} = FFT([帧 n-2, 帧 n-1])

frame_synthesis 里：
  x = IFFT(X_{n-1}) → 960 个采样，x[0] 对齐到帧 n-2 的起点，
                      x[480] 对齐到帧 n-1 的起点，x[960] 对齐到帧 n 的起点
  out[i] = x[i]·w[i] + synthesis_mem[i]
  synthesis_mem = 上一次的 x[480..960]，对齐到帧 n-2

→ 两项都对齐到帧 n-2
→ 第 n 次调用输出的是【帧 n-2 的降噪版本】
```

所以：

| 组成 | 延迟 | 来源 |
|---|---:|---|
| 50% 重叠分析窗固有的延迟 | **10 ms（1 帧）** | 必须等窗的后半段（新帧）到齐，才能发射前半段（旧帧） |
| `delayed_X` 额外引入的延迟 | **10 ms（1 帧）** | 增益作用于上一帧的频谱 |
| **合计算法延迟** | **20 ms（2 帧）** | |

#### 为什么要"故意"多延迟一帧？—— 换未来信息（look-ahead）

关键在于**增益是在什么信息下算出来的**：

- 第 `n` 次调用算出的 `g`，来自 `features`，而 `features` 来自 `X_n = FFT([帧 n-1, 帧 n])`——**它已经看到了帧 `n` 的全部内容**。
- 但这个 `g` 被用于重建**帧 `n-2`**。

也就是说：**网络在决定"帧 n-2 该怎么降噪"时，已经看完了帧 n-1 和帧 n（比输出帧晚 20 ms 的音频）。**

如果直接把 `g` 乘到 `X`（本帧的频谱）上，那么输出帧 `n-1` 的增益只用到 `[帧 n-1, 帧 n]` 的信息，look-ahead 只有 10 ms。

**多延迟一帧 = 多换来 10 ms 的"未来信息"。** 对降噪来说这非常值：语音起音（onset）的前 10 ms 决定了后续几十毫秒是不是语音，能提前看到就能避免"起音被削掉"这个最伤听感的问题。

#### 对实时系统的影响

| 影响 | 说明 |
|---|---|
| **延迟 20 ms** | 对 VoIP / 会议是**可接受**的（单向预算通常 40–150 ms）。但对**实时耳返（monitoring）**、**乐器效果器**场景，20 ms 已经能被人感知到（>10 ms 开始有"回声感"），需要额外注意。 |
| **不能并行/乱序** | `delayed_X` / `delayed_P` / `delayed_Ex` / `delayed_Ep` / `delayed_Exp`（`src/denoise.c:83-86`）是有状态的，必须严格按帧序调用 `rnnoise_process_frame()`。不能拆成多线程并行处理不同帧。 |
| **不能中途重建 `DenoiseState`** | 重建会把 `delayed_*` 清零，输出会先静音再逐渐恢复。 |
| **与编解码器联合设计时要注意** | 若前面还有 AAC/Opus 之类的算法延迟，总预算要一起算。 |
| **`vad` 也延迟了** | 返回的 `vad_prob`（`src/denoise.c:503`）对应的是当前帧的特征，但作用于上一帧的输出——用它做 VAD 驱动别的逻辑时要注意这 20 ms 的偏移。 |

### 4.4 `inverse_transform()` 的倒序输出与 `WINDOW_SIZE` 归一化因子

`src/denoise.c:200-217`：

```c
200: static void inverse_transform(float *out, const kiss_fft_cpx *in) {
204:   for (i=0;i<FREQ_SIZE;i++) x[i] = in[i];
207:   for (;i<WINDOW_SIZE;i++) {
208:     x[i].r = x[WINDOW_SIZE - i].r;
209:     x[i].i = -x[WINDOW_SIZE - i].i;
211:   rnn_fft(&rnn_kfft, x, y, 0);
213:   out[0] = WINDOW_SIZE*y[0].r;
214:   for (i=1;i<WINDOW_SIZE;i++) {
215:     out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
```

#### 第一步：构造共轭对称（`:204-210`）

```c
for (i=0;i<FREQ_SIZE;i++) x[i] = in[i];          // x[0..480] = X[0..480]
for (;i<WINDOW_SIZE;i++) {                        // i = 481..959
    x[i].r =  x[WINDOW_SIZE - i].r;               // x[960-i] 的实部
    x[i].i = -x[WINDOW_SIZE - i].i;               // 虚部取反
}
```

`i` 从 481 到 959，`WINDOW_SIZE - i` 从 479 降到 1。所以：

```
x[k]      = X[k]            for k = 0 .. 480   （含 Nyquist 点 x[480] = X[480]，实信号下它为实数）
x[N - k]  = conj(x[k])      for k = 1 .. 479   （N = 960）
```

这就是实序列 IFFT 要求的 **Hermitian 对称**。虚部取反即取共轭——变换后 `y` 的虚部会全部相消，只剩实部（所以 `:213-215` 只取 `.r`）。

#### 第二步：用正向 FFT 做反变换（`:211`）

代码调的是 `rnn_fft(..., x, y, 0)`——**正向** FFT，不是 IFFT。为什么能行？

设 `X[k]` 满足 Hermitian 对称（`X[N−k] = conj(X[k])`），我们要的时域序列是 IDFT：

```
s[n] = Σ_{k=0}^{N-1} X[k] · e^{+2πikn/N}
```

而正向 DFT 是：

```
F[n] = Σ_{k=0}^{N-1} X[k] · e^{-2πikn/N}
```

两者只差一个符号：`s[n] = F[−n mod N]`。对 `n = 0` 是 `F[0]`；对 `n = 1..N−1` 是 `F[N − n]`。

**这就是 `:213-215` "倒序输出"的数学来源**：

```c
out[0] = WINDOW_SIZE * y[0].r;                          // s[0] = F[0]
out[i] = WINDOW_SIZE * y[WINDOW_SIZE - i].r;            // s[i] = F[N - i]
```

> 用一次正向 FFT 完成反变换，是"省一个 IFFT 内核"的经典 trick——只要先构造共轭对称、再把输出倒着读就行。

#### 第三步：`WINDOW_SIZE` 因子的推导

kiss_fft 的正向变换**自带 `1/nfft` 归一化**：`src/kiss_fft.c:453-459` 设置 `st->scale = 1.f/nfft`，在 `src/kiss_fft.c:582` 的 `fout[...] = SHR32(MULT16_32_Q16(scale, x.r), scale_shift)` 应用。

所以 `y[n] = (1/N) · F[n]`（N = 960 = `WINDOW_SIZE`），而我们要的是 `s[n] = F[N−n]`。于是：

```
s[n] = F[N−n] = N · y[N−n]
```

**`WINDOW_SIZE` 这个乘数就是 kiss_fft 的 `1/nfft` 的倒数。** 它和反向 FFT 自己的 `1/N` 正好抵消，保证 `forward_transform` → `inverse_transform` 的往返是恒等的（在加窗前）。

#### 第四步：OLA 完整推导（`src/denoise.c:400-407`）

```c
inverse_transform(x, y);       // x = s[0..959]，幅度已还原
apply_window(x);               // 第二次加窗
for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
```

窗被乘了两次（分析一次、合成一次），实际权重是 `w²`。Vorbis 窗满足：

```
w[i]² + w[479−i]² = 1        （i = 0 .. 479）
```

（可用 `src/dump_rnnoise_tables.c:85` 的公式验证：例如 `w[0] ≈ 4.2e-6`、`w[479] ≈ 1.0`，平方和 ≈ 1；`w[240]² + w[239]² ≈ 0.5015 + 0.4983 ≈ 1`。）

在 OLA 里：

```
out[i] = x_当前[i]·w[i]  +  x_上一次[i+480]·w[479−i]
         └── 窗的前半段 ┘   └── 窗的后半段，权重反序 ┘
```

两项对应的都是**同一段输入采样**（见 §4.3 的对齐分析），权重平方和为 1，所以完美重建。

> **一句话总结这一节的三个"魔法数字"**：
> - 虚部取反 → 构造 Hermitian 对称 → 输出是实序列；
> - 输出倒序读 → 用正向 FFT 算 IDFT；
> - 乘 `WINDOW_SIZE` → 抵消 kiss_fft 的 `1/nfft` 归一化。

---

## 5. 附录

### 5.1 关键文件速查表

| 想看什么 | 看哪个文件:行号 |
|---|---|
| 整条处理链路 / 主循环 | `src/denoise.c:457-504` |
| 65 维特征怎么算 | `src/denoise.c:347-398` |
| 频带边界表 | `src/denoise.c:63-65` |
| 频带能量 / 互相关 | `src/denoise.c:90-113` / `:115-138` |
| DCT | `src/denoise.c:160-170`；查表 `src/rnnoise_tables.c:669` |
| 正/反 FFT + 加窗 | `src/denoise.c:186-225` |
| 重叠相加合成 | `src/denoise.c:400-407` |
| 基音梳状后滤波 | `src/denoise.c:421-455` |
| 网络主链（10 行） | `src/rnn.c:44-60` |
| 死宏 `INPUT_SIZE 42` | `src/rnn.c:41` |
| GRU 前向（含 zrh 布局） | `src/nnet.c:65-94` |
| Conv1d 退化成 Linear | `src/nnet.c:113-123` |
| 内核分派（float/int8 × 稠密/稀疏） | `src/nnet_arch.h:130-162` |
| `diag` 补回 | `src/nnet_arch.h:153-161` |
| `LinearLayer` / `WeightArray` / `WeightHead` | `src/nnet.h:65-75` / `:43-48` / `:55-62` |
| 激活函数（含近似实现） | `src/nnet_arch.h:79-127`；近似 `src/vec.h:337-381` |
| int8 稀疏内核 | `src/vec.h:248-281`（C）、`src/vec_avx.h:778-877`（AVX2） |
| float 稀疏内核 | `src/vec.h:123-180` |
| 权重装配 | `src/parse_lpcnet_weights.c:123-176` |
| 稀疏索引合法性校验 | `src/parse_lpcnet_weights.c:98-121` |
| blob 解析 / 写出 | `src/parse_lpcnet_weights.c:37-78` / `src/write_weights.c:46-69` |
| **真实网络维度（权威）** | `src/rnnoise_data.h:8-38` |
| **真实层配置（权威）** | `src/rnnoise_data.c` 的 `init_rnnoise()`（724217-724229 行） |
| 模型加载入口 | `src/denoise.c:285-309` |
| 对外 API | `include/rnnoise.h:57-125` |
| PyTorch 模型定义 | `torch/rnnoise/rnnoise.py:58-109` |
| PyTorch → C 导出脚本 | `torch/rnnoise/dump_rnnoise_weights.py` |
| 数组命名与量化/稀疏布局（**生成逻辑在这**） | `torch/weight-exchange/wexchange/c_export/common.py:32-364` |
| 稀疏化训练调度 | `torch/rnnoise/rnnoise.py:38-50`、`torch/sparsification/gru_sparsifier.py` |
| 训练脚本参数 | `torch/rnnoise/train_rnnoise.py:40-57` |
| DCT 表 / 窗表的生成 | `src/dump_rnnoise_tables.c:84-101` |
| 训练特征提取（必须与推理一致） | `src/dump_features.c:363, 487-489` |

### 5.2 代码中的历史包袱 / 反直觉之处

| # | 位置 | 现象 | 说明 |
|---|---|---|---|
| 1 | `src/rnn.c:41` | `#define INPUT_SIZE 42` | **死宏**。真实输入是 65。只在一行被注释掉的调试 printf（`:47`）里出现。详见 §1.5 |
| 2 | `src/rnn.c:58` | 注释里 `for (int i=0;i<22;i++)` | 另一个旧 `NB_BANDS`（22）的残留，现在 gains 是 32 维 |
| 3 | `src/denoise.c:168` | `sum*sqrt(2./22)` | 22 是旧频带数，现在 32。这个错的归一化因子已被"烤进"训练好的权重，**改成 `sqrt(2./NB_BANDS)` 会让模型精度崩掉**（除非重训） |
| 4 | `src/vec.h:384`、`vec_avx.h:879`、`vec_neon.h:363` | `#define SCALE (128.f*127.f)` | **0 处使用的死常量**，从 LPCNet 继承。真正的缩放由每层 `*_scale` 承载 |
| 5 | `src/nnet.c:54` + `src/nnet_arch.h:105-110` | `SOFTMAX_HACK` | softmax 被退化成 `RNN_COPY`（恒等映射）。RNNoise 不用 softmax，无害 |
| 6 | `src/nnet.h:78-85`、`src/parse_lpcnet_weights.c:178-201`、`src/nnet_arch.h:169-251` | `Conv2dLayer` / `conv2d_init` / `compute_conv2d_` | **RNNoise 完全没用到**，是从 LPCNet 共享代码库继承的死代码。`RNNoise` 结构体里一个 `Conv2dLayer` 都没有 |
| 7 | `src/nnet.h:34-39` | `ACTIVATION_LINEAR/RELU/SOFTMAX/SWISH` | RNNoise 只用 `SIGMOID` 和 `TANH`，其余 4 个是通用库能力 |
| 8 | `src/nnet.h:52` | `WEIGHT_TYPE_qweight` | 未使用 |
| 9 | `src/nnet_arch.h:60` | `MAX_ACTIVATIONS 4096` | 只被 `vec_swish()`（`nnet_arch.h:62-70`）用到，RNNoise 不用 swish |
| 10 | `src/nnet.h:89-106` | 一串 `#define compute_linear_c rnn_compute_linear_c` | 加 `rnn_` 前缀是为了和 Opus 的同名符号共存（注释见 `src/nnet.h:88`）。不是为了可读性 |
| 11 | `src/nnet_arch.h:36-39` | `CAT_SUFFIX2` / `CAT_SUFFIX` 两层拼接宏 | 两层间接是**必须的**：只写一层的话 `RTCD_ARCH` 不会被展开。C 预处理器的经典陷阱 |
| 12 | `src/denoise.c:142-153` | 20 kHz 以上增益恒为 0 | `eband20ms[33] = 400 < FREQ_SIZE = 481`，剩下的 81 个频点保持 memset 的 0。训练集同处理，所以不影响质量 |
| 13 | `src/parse_lpcnet_weights.c:91-96` | `opt_array_check` 只对 `float_weights` 用 | 这个"可选数组"语义是 `DISABLE_DEBUG_FLOAT` 开关的实现基础 |
| 14 | `torch/rnnoise/rnnoise.py:59` | `gru_size=256` 默认值 | **过时**，真实模型是 384。权威值看 `src/rnnoise_data.h:24` 和 `train_rnnoise.py:49` |
| 15 | `torch/rnnoise/dump_rnnoise_weights.py:64, 67` | 非量化层传 `scale=1/128` | 冗余：非量化层不生成 `*_scale` 数组（`common.py:223` 的 `scale_name = ... if quantize else "NULL"`，`common.py:231` 只在 `quantize` 时用 scale） |
| 16 | `src/denoise.c:271-275` | `rnnoise_model_free()` 先 `fclose` 再 `free(blob)` | 顺序上略可疑（blob 里的 `WeightArray.name` 指向文件映射/缓冲），严格说应先释放引用者。多数 libc 下 `fclose` 不立即 unmap 所以没炸 |
| 17 | `src/write_weights.c:38-44` | `#undef HAVE_CONFIG_H` 再 `#include "rnnoise_data.c"` | 作者自称的 hack：工具需要静态数组作转换源，但 `config.h` 里的 `USE_WEIGHTS_FILE` 会把数组 `#ifndef` 掉 |

### 5.3 如果要改模型维度 / 增删层，需要同步改动的位置

#### A. 自动生成的（**不要手改**，改训练脚本后重新导出即可）

| 位置 | 生成规则 |
|---|---|
| `src/rnnoise_data.h:8-38` 的 `*_IN_SIZE` / `*_OUT_SIZE` / `*_STATE_SIZE` / `*_DELAY` | `wexchange/c_export/common.py:276, 297-300, 361-362` |
| `src/rnnoise_data.c` 的 `init_rnnoise()` 里 10 个 `linear_init()` 调用 | `wexchange/c_export/common.py:255-256` + `c_writer.py:159-166` |

新增层只要在 PyTorch 模型里加上，`dump_rnnoise_weights.py:60-91` 会遍历 `named_modules()` 自动导出，C 侧的 `RNNoise` 结构体和 `init_rnnoise()` 会自动跟着变。

#### B. 手写的上界宏（**改大维度时必须检查**）

| 宏 | 当前值 | 约束 | 当前余量 |
|---|---:|---|---|
| `MAX_NEURONS`（`src/rnn.h:37`） | 1024 | ≥ `max(CONV1_OUT_SIZE, CONV2_OUT_SIZE)` | 需要 384 ✓ |
| `MAX_RNN_NEURONS_ALL`（`src/nnet.c:63`） | 1024 | ≥ `gru_size`（数组开 3×，见 `src/nnet.c:69-70`） | 需要 384 ✓ |
| `MAX_CONV_INPUTS_ALL`（`src/nnet.c:111`） | 1024 | ≥ `max(conv 的 nb_inputs)` | 需要 384 ✓ |
| `MAX_INPUTS`（`src/vec.h:45`） | 2048 | ≥ `max(nb_inputs)`（int8 内核的输入量化缓冲） | 需要 384 ✓ |

`src/rnn.c:46` 的 `cat[]` 大小是宏表达式，自动跟随，**不需要改**。

#### C. 与模型语义绑定（**改了必须重训**）

| 位置 | 内容 |
|---|---|
| `src/denoise.h:34-35` | `NB_BANDS` / `NB_FEATURES` |
| `src/denoise.c:63-65` | `eband20ms[]` 频带边界表 |
| `src/rnnoise_tables.c:669` | `rnn_dct_table`（由 `src/dump_rnnoise_tables.c:91-97` 生成，随 `NB_BANDS` 变） |
| `src/denoise.c:395-396` | `features[0] -= 12` / `features[1] -= 4` 的经验平移量 |
| `src/denoise.h:38-41` | 基音搜索常数 |
| `src/dump_features.c:363, 487-489` | 训练特征提取，每帧写 98 个 float |
| `src/denoise.c:389` | 静音门限 `E < 0.04` |

#### D. 训练侧（**必须同步改**）

| 位置 | 内容 |
|---|---|
| `torch/rnnoise/rnnoise.py:59` | 构造参数默认值（`input_dim` / `output_dim` / `cond_size` / `gru_size`） |
| `torch/rnnoise/rnnoise.py:66-72` | 层定义 |
| `torch/rnnoise/rnnoise.py:106` | `out_cat` 的拼接方式（`4*gru_size`） |
| `torch/rnnoise/train_rnnoise.py:48-49` | `--cond-size` / `--gru-size` 默认值 |
| `torch/rnnoise/train_rnnoise.py:74, 88` | `dim = 98` 与 `data[:, :, :65] / [:, :, 65:-1] / [:, :, -1:]` 的切片 |
| `torch/rnnoise/dump_rnnoise_weights.py:15` | `unquantized` 列表（**新增层若不想被量化，必须加进来**） |
| `torch/rnnoise/rnnoise.py:43-50` | `sparse_params1` 各门密度 |
| `torch/rnnoise/rnnoise.py:38-41` | 稀疏化调度（start / stop / interval / exponent） |

#### E. 换模型（不改维度）的操作清单

1. 训练得到新的 `.pth`；
2. `python3 dump_rnnoise_weights.py --quantize rnnoise_XX.pth rnnoise_c`；
3. 把 `rnnoise_data.c` / `rnnoise_data.h` 拷到 `src/`（`README:101`）；
4. 重新 `./autogen.sh && ./configure && make`；
5. 若要用 blob 形式：用 `write_weights` 程序导出 `weights_blob.bin`（`README:107-115`）。

> **注意 `README:117-118`**：如果加了 `-DUSE_WEIGHTS_FILE`，运行时**必须**传 model，否则输出全静音（见 §3.8 末尾）。
> **注意 `README:121-125`**：`rnnoise_data_little.c` 与 `rnnoise_data.c` 的宏完全相同，可以直接改名替换，同一个二进制也能加载两种 blob。



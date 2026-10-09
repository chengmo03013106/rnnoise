# RNN_UNIT 整网误差分析 — 任务跟进

> 本文是**执行用的跟进文档**：每完成一项就改成 `- [x]`。
> 「阶段 0」没走完（待决问题没答案）之前，后面任何一步都不要开工。

## 0. 目标（用户原文要点）

1. 把 `gru_scratch.py` 里的 `_check_sigmoid_diff` / `_rnnoise_sigmoid_approx` / `_rnnoise_tanh_approx` 拆到**单独文件**。
2. `gru_scratch.py` **改名 `gru.py`**。
3. 原 `gru_scratch.py`、`rnn_unit.py` 里**所有用 tanh / sigmoid 的地方**都支持高/低精度，用开关区分。
4. Python `rnn_unit` 与 C `rnn_unit` 做误差分析：
   - 输入 = `conv.json` 的 `conv1.inputs`（16 帧 × 4）
   - 四组对比：`py_high vs c_high` / `py_low vs c_low` / `py_high vs py_low` / `c_high vs c_low`
   - **每一帧、所有层输出**都要比、记录误差
   - 对误差做**根因分析**
   - 并回答：**「直接通过输出来对比」这种方法科学吗？有没有更好的方式？**
5. Agent A 拆解 + 生成本文；Agent B review 本文 / 代码改动 / 精度报告。
6. 两个 agent 都不能确定的 → 问用户，不许推测。
7. 指出用户考虑不足的地方。

---

## 1. 现状核对（已逐行读过源码，含已发现的缺陷）

### 1.1 代码

| 位置 | 现状 |
|---|---|
| `examples/rnn_unit.py:296` | `DenseLayer.active = torch.sigmoid` — **硬编码** |
| `examples/rnn_unit.py:309` | `Conv1D.active = torch.tanh` — **硬编码** |
| `examples/rnn_unit.py` | `gru_demo()` 经 `GRUMo` 已支持 `--acc`；**上一行两个类不支持** |
| `examples/rnn_unit.py:rnnoise_demo()` | 用户的 WIP。**帧循环末尾 `break` → 只跑第 0 帧**；`emit('vad', "out", frame, gains[0])` **误用了 gains**；协议名 `gains.out`/`vad.out`（C 是 `dense.gains`/`vad`）；**没有** emit `dense.in`；中间夹着非协议 `print(f'conv1 {frame}:...')` |
| `examples/rnn_unit.c:conv_gru_unit()` | `all` 单元。串 `conv1→conv2→gru1/2/3→concat→dense/vad` ✓。但打的是 `[gruN.state#i]` **72 个**，而 `compute_generic_gru()` 只写了前 24 个（`N=CASE_GRU_N`），后 48 个是数组里的陈旧值；`gru_unit()` 那边打的是 **24**，两侧长度不一致。另外 `gruK_state` 声明成了真实模型的 `GRU_STATE_SIZE`(384) |
| `examples/rnn_unit.c:conv_gru_unit()` | 调的是 `src/nnet.c` 的**正版** `compute_generic_gru()`，所以链路上**没有** `zrh_recur / sigmoid / recur_tanh / h` 这些内部量，只有最终 `state` |
| `examples/gru_scratch.py:_check_sigmoid_diff()` | 内部调用了**并不存在的 `_rnnoise_sigmoid`**（一调用就 `NameError`），且**没有任何调用者** |
| `examples/gru_scratch.py:gru_demo()` | 坏的：`net(inputs_data, hidden_0, gru)` 里 `gru` 未定义 |

### 1.2 用例规模（真实模型尺寸 //16）

```
conv1  每帧输入 4  → 线性 12 → 输出 8      （conv.json 里 只有 conv1 有 inputs：16 帧 × 4）
conv2  每帧输入 8  → 线性 24 → 输出 24     （conv.json 里 conv2 没有 inputs）
gruN   每帧输入 24 → 线性 72 ；N=24       （gru.json 里 只有 gru1 有 inputs：16 帧 × 24）
dense_out / vad_dense  输入 96 → 2 / 1     （dense.json 只有参数）
```

### 1.3 可复用的基础设施

- 协议：`[<层名>.<输出名>#<帧>] %.8e ...`；C 侧 `print_layer_item()`，py 侧 `emit()`。
- C 精度开关：`make rnn_unit RNN_UNIT_HIGH_ACCURACY=-DHIGH_ACCURACY`（`Makefile.am` 里的变量）。
  **切档前必须 `touch src/nnet_default.c`**，并靠运行时 `[Notice] Now high/low accuracy used` 校验
  —— automake 不会因为 CFLAGS 变化就重编 `.o`。
- Python 精度开关：`--acc high|low`（缺省 `low`）。
- 统计设施（**2026-10-02 起**）：`scripts/err_stats/collect_all.sh` 采集 +
  `err_all.py` / `teacher_force.py` / `bench_act.py` / `amplify.py` / `business_check.py` / `trace_value.py`。
  `err_stat.py` 是它们共用的**数值工具模块**：`f32()`（十进制 → float32 还原）、`parse()`（按 key 判重）、
  `mx` / `mean` / `e`。
  > 这里原本还列着 `{collect.sh, make_cases.py}` —— 那两个属于**第一版单层 GRU 任务**的工具链，
  > 已于 2026-10-02 连同 `raw/` 和第一版报告一起删除（详见 §阶段 8）。
- 基线：`uv run scripts/unit_check.py` → **272 项全过**。

### 1.4 两个必须记住的坑

1. **文本打印只有 9 位有效数字**（`%.8e`）。float32 恰好需要 9 位才能唯一还原，所以读数据时
   **必须先把十进制还原成 float32 再比**；直接当 double 用会凭空多出 ~5e-10 假差异。
   自检判据：**两个 float32 之差只能是 0 或 ≥1 ulp**，出现中间值就是解析口径错了。
2. **别在 C 块注释里写 `**// 16**`** 这类含 `*/` 的文字。

---

## 2. 阶段拆解（逐项勾选）

### 阶段 0｜需求澄清（阻塞后面全部）

- [x] Q1 `rnnoise_demo()` 怎么处理 → **授权改最小必要三处**（去 `break`、修 `vad` 用错变量、统一命名），其余写法/注释/结构不动
- [x] Q2 `all` 链路的 C 侧要不要输出 gru 内部量 → **不要**，只比 `state`
- [ ] Q3 方法论：是否接受「逐层 teacher forcing 隔离 + 单算子微基准 + 四组 pairwise」→
      用户在问这三个名词分别是什么、为什么做、结果能表达什么（2026-09-30 追问中）
- [x] Q4 `gru.py` 里剩下的 RNN 教学代码去留 → **暂时留在 `gru.py`**，加一行 TODO 说明它不是 GRU
- [x] Q5 `_check_sigmoid_diff` → 搬过去并**修好悬空引用**，保留 C 的 `0.5+0.5*tanh_approx(0.5x)` 版本做对照
- [x] Q6 新文件名 → `examples/rnnoise_activation.py`
- [x] Q7 报告位置 → 新建 `doc/RNN_UNIT整网误差分析.md`
- [x] Q8 协议命名 → 以 **C 侧**为准（`conv1.out`/`conv2.out`/`gruN.state`/`dense.in`/`dense.gains`/`vad`）
- [x] Q9 `all` 单元 → 纳入 `scripts/unit_check.py` 覆盖（给它一个只有 `unit`/`tolerance` 的 `all.json`）
- ✓ Q3 有答案后才算完成

> **Q1 答案带来的简化**：既然可以动 `rnnoise_demo()`，就不必再另建一份镜像 harness 当主路径。
> 主路径 = **修好后的 `rnnoise_demo()`**（跑"自回归/free-running"的端到端链路）；
> teacher forcing 隔离因为要"从外部注入 C 的输入"，仍然需要一个**独立的小脚本**（Python 侧天然支持注入：
> `Conv1D.forward(data, mem)` / `GRUMo.__call__(x)` 都是从外面喂的，`mem` / `hidden` 也可注入）。

### 阶段 1｜拆文件 + 改名（纯重构，零行为变化）—— **2026-09-30 完成**

- [x] 新建 `examples/rnnoise_activation.py`（116 行），移入三个函数并改名：
      `_rnnoise_tanh_approx` → `tanh_approx`、`_rnnoise_sigmoid_approx` → `sigmoid_approx`、
      `_check_sigmoid_diff` → `check_sigmoid_diff`；另加 `sigmoid_from_tanh()`（C 的
      `0.5+0.5*tanh_approx(0.5x)` 写法，与 `sigmoid_approx` 数学等价）和 `pick(low_accuracy)`
- [x] 修 `check_sigmoid_diff` 的悬空引用：原版调的 `_rnnoise_sigmoid` 并不存在。
      **顺手改了签名**：`(zrh, hidden_size, recur)` → `(pre_act, out)`（原来那个签名要求调用方
      先做切片相加，用它当通用工具很别扭）
- [x] `examples/gru_scratch.py` → `examples/gru.py`
- [x] `rnn_unit.py` 的 import 改成 `from gru import GRUMo as GRUMo`
- [x] 全局搜 `gru_scratch` 残留：`doc/学习工具搭建手册.md`（树 + 4.3 表）、
      `doc/GRU计算过程误差对比统计.md`（4 处）、`.codebuddy/memory/MEMORY.md` 都已同步；
      只剩 `rnnoise_activation.py` 模块头里那份"原名对照"是故意留的
- ✓ 验收：`uv run scripts/unit_check.py` → **272/272** ✓
- ✓ 验收：冒烟 `import gru, rnnoise_activation` 成功、`check_sigmoid_diff` 能出数 ✓

**这一步顺手修的两个"外部改动/历史遗留"（都先报备）**：

1. **`from torch import nn` 被删掉了**（git 里还在，工作区没了）→
   `gru.py:get_gru_params_from_net(net: nn.GRUCell)` 的**函数注解在 import 时就求值**，
   于是整个模块 import 失败、`unit_check` 从 272 项掉到 **0 项通过**。
   **这是在我动手之前就坏的**（外部编辑导致），已恢复该 import。
   → 教训：**Python 的类型注解也是运行时求值的**，`nn` 没导入连模块都加载不了。
2. `gru_demo()` 原来是坏的：`net(inputs_data, hidden_0, gru)` 里 `gru` 未定义；
   而且它按"教材版 9 组参数 + 批量"写，与现在 `GRUMo`（RNNoise 的 [z|r|h] 布局、
   **一次一帧**）根本不兼容 —— 即使补上 `gru` 也会 `ValueError: too many values to unpack`。
   已把它接到 `do_gru()`（教材版整段递推，与 `get_gru_params()` 是一对）上，`uv run examples/gru.py` 现在能跑。
   `GRUMo` 不再被 `gru_demo` 使用，但仍是 `rnn_unit.py` 的入口。

### 阶段 2｜tanh / sigmoid 开关全覆盖 —— **2026-09-30 完成**

- [x] `DenseLayer.__init__(..., low_accuracy=True)` → `self.active = pick(low_accuracy)[0]`
- [x] `Conv1D.__init__(..., low_accuracy=True)` → `self.active = pick(low_accuracy)[1]`
- [x] `conv1d_demo()` 把 `low_accuracy` 传给 `Conv1D`；删掉"conv1d 不受 --acc 影响"那段注释
- [x] `rnnoise_demo()` 的 4 个构造点各加 `low_accuracy=low_accuracy`
- [x] `gru.py` 的教学 `rnn()` / `RNNScratch` / `do_gru()` 也纳管（加 `low_accuracy` 参数 + `pick()`）
- ✓ 验收：`uv run scripts/unit_check.py` → **272/272** ✓
- ✓ 验收（探针）：`all` 单元在两档下都是 16 帧 × 8 项 = 128 行，键集一致，
  **8 项 × 16 帧全部不同**（不止 conv1，conv2/dense.gains/dense.in/vad.out/gruN.state 也都不同）✓

**默认档定成 `low`**（与 C 不定义 `HIGH_ACCURACY` 对齐）。注意这改变了 `conv1d` 单元在
`unit_check` 里的语义：原来是"Py 高 vs C 低"，现在两边都是低精度（`unit_check` 仍 272 项全过）。

### 阶段 3｜C / Py 协议对齐 —— **2026-09-30 完成**

- [x] `conv_gru_unit()` 的 `gruN.state` 长度 **`CASE_GRU_OUT`(72) → `CASE_GRU_N`(24)**，和 `gru_unit()` 一致
      （原来那 72 个里有 48 个是数组里的陈旧值 —— 是数据污染源，不是"多打点更保险"）
- [x] 顺带把 `gru1/2/3_state` 的声明从真实模型的 `GRU_STATE_SIZE`(384) 改成 `CASE_GRU_N`(24)
- [x] C 的 `[vad#i]` 改成 `[vad.out#i]`（`print_item("vad", ...)` → `print_layer_item("vad","out",...)`），
      与 `[conv1.out#i]` 统一；Python 侧对齐成 `emit('vad', "out", ...)`
- [x] Python 侧命名对齐：`gains.out` → `dense.gains`；新增 `dense.in`；`vad` 改用 `vad[0]`
      （原来错用了 `gains[0]`）
- [x] 按 Q2 决定：`all` 链路**不**补 gru 内部量。
      ★ **2026-10-08 起演进**：曾用 `detail` 开关控制打几项，现已彻底简化 ——
      `gru.py` **不打印任何协议行**：中间量由 `GRUMo.forward()` 随返回值一起给出
      （`h_out, detail = net(x, h_in)`），打哪几项由调用方决定（`rnn_unit.py: gru_demo()`）。
      原因：trace / `torch.onnx.export` 的图里不能混进 Python 打印这种副作用。
- [x] `rnnoise_demo()` 去掉帧循环末尾的 `break`（原来只跑第 0 帧）—— 这是你授权的"最小必要三处"之一
- ✓ 验收：两侧都是 **128 个 key**，**键集完全相同**（`set(c) == set(py)` 为 True），
  **值个数不一致的 key = 0** ✓

> **还没做的核对**：逐行比对 `conv_gru_unit()` 与 `rnnoise_demo()` 的计算图（concat 顺序、
> `cat[]` 的偏移、隐状态初值、喂给下一层的到底是 `conv2.out` 还是 `gruN.state`）。
> 阶段 3 的"键集/个数对齐"只证明**接口**对齐，不证明**语义**对齐。
> 这一条**移入阶段 4 的第一步**：采集时先做一次 frame 0 的逐层数值抽查（高精度下两侧应在 1 ulp 量级）。

### 阶段 3.5｜语义对齐实证 —— **2026-09-30 完成（抓到一个严重 bug）**

思路：光对齐 key 不够，得用**数值**证明两边算的是同一件事。做法是 C 与 Py 都切高精度
（此时激活几乎无误差），跑同一条 `all` 链路逐项比 —— 期望落在 **1~2 ulp**。

**第一次跑就炸了**：`conv1.out` 的 `max|Δ|` 到了 **3.7e-01**，而第 0 帧却是 `0.000e+00`。
用 float64 参考一查：**Python 对（与参考差 7.3e-09），C 错（差 4.5e-02）**。

**根因：`padding_rn()` 拷输入序列时行步长不匹配。**

```c
float inputs[CASE_NB_FRAMES][CASE_CONV1_IN];            /* 目标：行步长 4 */
memcpy(inputs, cc[0].inputs, CASE_NB_FRAMES*CASE_CONV1_IN*sizeof(float));
/*            ^^^^^^^^^^^^ 但 Conv1dCase.inputs 的声明是
 *            [CASE_NB_FRAMES][CASE_CONV_MAX_IN]，行步长 = CASE_CONV2_IN = 8 */
```

整块按"步长 4"拷 64 个 float 的结果是：

```
inputs[0] = 第 0 帧（对）
inputs[1] = 第 0 帧的填充区（那 4 个 float 从没被写过，恒为 0）
inputs[2] = 第 1 帧
inputs[3] = 第 1 帧的填充区（0）
...
→ 隔行变 0，而且 json 里第 8..15 帧根本用不上
```

坐实方式：按"mem 正确 + 输入全 0"算出来的 8 个数，与 C 打印的 `conv1.out#1` **逐位相同**。

**修法**（逐行拷，用正确的源步长）：

```c
for (int f = 0; f < CASE_NB_FRAMES; ++f)
  memcpy(inputs[f], cc[0].inputs[f], CASE_CONV1_IN*sizeof(float));
```

**修复后（高精度、16 帧、C vs Py）**：

| 项 | max\|Δ\| | mean\|Δ\| |
|---|---|---|
| `conv1.out` | 1.490e-08 | 1.164e-10 |
| `conv2.out` | 1.490e-08 | 1.940e-10 |
| `gru1.state` | 5.960e-08 | 8.764e-09 |
| `gru2.state` | 7.451e-08 | 1.703e-08 |
| `gru3.state` | 1.192e-07 | 2.072e-08 |
| `dense.in` | 1.192e-07 | 1.168e-08 |
| `dense.gains` | 2.980e-08 | 1.863e-08 |
| `vad.out` | 5.960e-08 | 1.863e-08 |

`conv1.out` 里 15/16 帧**逐位完全相同**（只有帧 8 差 1 ulp）。
→ **语义对齐确认**：1~2 ulp 就是 float32 累加顺序的正常量级，GRU 那几层略大是因为递推累积。

- [x] 高精度下逐层数值抽查（这一步证明了"计算图镜像"而不只是"接口对齐"）
- [x] 顺带扫同类隐患：`rnn_unit.c` 里涉及 `inputs` 的 `memcpy` 只有两处，
      另一处 `memcpy(inputs, gc.inputs, sizeof(inputs))` 两侧类型/步长一致，安全
- [x] `unit_check` 回归 **272/272**

> **为什么值得记下来**：这个 bug 的"接口"完全正确 —— key 对、长度对、帧数对，
> 只是喂进去的**数**错了。如果直接进阶段 5 采集，得到的会是一张"误差 3.7e-01、看起来像
> C 实现烂透了"的假报告，而根因其实在用例装填这一层。

### 阶段 4｜数据采集（16 帧 × 全层 × 4 组）—— **2026-09-30 完成**

- [x] 语义对齐抽查（见阶段 3.5）
- [x] **C 切档脚本化**：新增 `scripts/err_stats/collect_all.sh` —— 每次切档先
      `touch src/nnet_default.c` 重编，再**断言**运行时的
      `[Notice] Now high|low accuracy used`，断言不过直接 `exit 1`
      （防"编译没生效 → 采到假同精度数据"）
- [x] 采集四份到 `scripts/err_stats/raw_all/`：`c_high` / `c_low` / `py_high` / `py_low`
      （各 128 行 = 16 帧 × 8 项；另有 `.all` 完整 stdout 和 `.err`）
- [x] 统计脚本 `scripts/err_stats/err_all.py`，复用 `err_stat.py` 的 `parse()`（**先还原 float32 再比**）
- ✓ 验收：项名集合四组一致；每个 key 都有 16 帧、无重复；**值个数四组一致**；无 NaN/Inf
      → **全部通过**
- 注意：脚本跑完后磁盘上的 C 二进制是**低精度**那份（最后一次编译）

> **2026-10-02 说明**：`doc/RNN_UNIT整网误差分析.md` 已**整体重写**（用户要求：删除推测内容、
> 统一业界术语、补数值溯源、去掉历史补全痕迹）。重写后**章节编号全部变了**，
> 本文件下面各处对报告章节号的引用可能已失效，以报告本身为准。
> 新增 `scripts/err_stats/trace_value.py`（单个数值的完整溯源）。

### 阶段 5｜误差报告 + 根因分析 —— **2026-09-30 完成**（报告：`doc/RNN_UNIT整网误差分析.md`）

- [x] 四组 pairwise 的逐项 `max|Δ|` / `mean|Δ|`（见下）
- [x] 逐帧误差增长曲线（`err_all.py: sec_per_frame`）
      → 第 0 帧就已经有 4.9e-05（**根本没有分叉可言**），16 帧后只到 1.12e-04（2.3×）
      → **误差的"种子"是每帧激活近似本身**；放大倍数**取决于对比组**（2.3× vs 4.3×）
- [x] 逐层 teacher forcing 隔离（`scripts/err_stats/teacher_force.py`）
      → 四列 `A_self`（rcp 隔离）/ `A_swap`（同输入换激活）/ `B_free` / `D_free`
      → `A_self ≥ A_swap` **每项成立**；`B_free/A_self` = 0.78~1.54（种子主导）；
        `D_free/A_swap` = 1.00~6.52（放大主导）
- [x] 单算子微基准（`./rnn_unit activation` + `scripts/err_stats/bench_act.py`）
      → sigmoid：C高 **0.000e+00** / C低 1.446e-04 / Py高 5.960e-08 / Py低 2.933e-05
      → tanh：C高 **0.000e+00** / C低 2.769e-04 / Py高 1.490e-08 / Py低 5.966e-05
      → **必须分两个区间看**：全网格 `[−8,+8]` vs 网络实际 pre-act 区间 `[−1.01,+0.68]`
        （不分区看会把量级说大 2 倍）
      → linear（`dense_out` 96→2）：**1 ulp**（5.957e-08）
- [x] 根因分类与归因（见报告 §5.3 归因表）
- [x] **关闭 5.2 的疑点**（见下）
- ✓ 验收：每个 >1 ulp 的偏差都归到了某一类根因；报告里每个数字都能被脚本重跑复现

### 阶段 6｜Agent B 独立复核 —— **2026-09-30 完成**

- [x] 复核跟进文档的勾选真实性 —— 通过
- [x] 复核代码 diff —— 通过（`padding_rn` 修复被判正确；`detail` 开关被认为必要而非过度设计；★ 2026-10-08 该开关已按需求移除，见上文）
- [x] 复核精度报告的结论是否被数据支撑 —— **判"需修正"**，B 抓到了本次报告**最站不住的一条**：
      > 报告原写"低精度误差的主体是每帧激活近似、**不是轨迹分叉**"。
      > B 指出这与自身数据矛盾：`D_free/A_swap` = 2.55 / 4.49 / **6.52**，
      > 说明"换激活"那一组的末层量级**恰恰由轨迹放大主导**。
      > 正确表述应是「种子在每帧激活、量级 = 种子 × 放大倍数，放大倍数取决于对比组」。
- **已按 B 的意见修正**（见下"复核后的改动"）
- ✓ 结论：**通过（修正后）**

#### 复核后的改动（2026-09-30）

| # | B 的意见 | 裁定与处置 |
|---|---|---|
| 1 | 核心结论"不是轨迹分叉"过强 | **采纳**。报告 §0/§4 全部改成"种子 × 放大倍数、取决于对比组" |
| 2 | "共模相消"与微基准 `C低−Py低=1.385e-04` 自相矛盾 | **部分采纳**。矛盾来自口径：网格最差值用的是人为铺满的 `[−8,+8]`，而网络 pre-act 只在 `[−1.01,+0.68]`。补了"两区间"对照后完全自洽（网络区间上 sigmoid `4.700e-05`、tanh `1.726e-04`，与端到端同量级）。**并补上可加性铁证**：`C低−torch ≈ (C低−Py低) + (Py多项式−精确)`，sigmoid 和 tanh 两组都成立（tanh 精确到 4 位）→ 共模相消**成立** |
| 3 | "C高=0"表述不清 | **采纳**。改成"在 1601 点网格上逐位等于 float32 精确值，即**正确舍入**，不是数学真值零误差" |
| 4 | 缺分位数 / 只报 max | **采纳**（列入报告 §8 遗留），本轮未做 |
| 5 | `gru.py` 的 `low_accuracy` 默认值未确认 | **驳回**。实测 `rnn()` / `RNNScratch.__call__` / `do_gru()` / `GRUMo`（构造参数）**默认值全是 `True`**，与 C 缺省对齐 |
| 6 | `check_sigmoid_diff` 是死代码 | **采纳**。给 `rnnoise_activation.py` 加了 `__main__`，现在 `uv run examples/rnnoise_activation.py` 可独立跑 |
| 7 | 272/272 没抓到 `padding_rn` 的 bug，要补回归断言 | **采纳**。`err_all.py` 加了护栏：**高精度 `conv1.out` 必须 ≤1e-6**，否则判不通过（实测 1.490e-08 → 通过） |
| 8 | C dump 契约改过（72→24 等），要确认下游同步 | **已核实**。`collect_all.sh` / `err_all.py` / `teacher_force.py` 都只按值个数 zip，不依赖长度；自检的"值个数四组一致"为"是" |
| 9 | 是否把 C 低精度对齐到 Py 的多项式 | **不适用**。C 是上游参考实现，`_mm256_rcp_ps` 是 RNNoise 为速度刻意选的；本任务的目标是**量差异、找根因**，不是消除差异 |
| 10 | 增益/VAD 的听感/阈值验收 | **采纳为遗留**（报告 §8 第 3 条）。这是用户确实没提、但业务上真正重要的验收维度 |
| 11 | 调试用 `activation` 单元是否留主线 | **保留**。它被 `collect_all.sh` 调用、是归因的刻度尺，不是死代码 |

#### 5.1 四组 pairwise（`Δ = 前者 − 后者`，已 `f32()` 还原）

| 打印项 | 维度 | **py高 vs C高** | **py低 vs C低** | **py高 vs py低** | **C高 vs C低** |
|---|---|---|---|---|---|
| `conv1.out` | 8 | 1.490e-08 | 5.144e-05 | 4.455e-05 | 6.366e-05 |
| `conv2.out` | 24 | 1.490e-08 | 5.886e-05 | 5.972e-05 | 1.047e-04 |
| `gru1.state` | 24 | 5.960e-08 | 6.312e-05 | 9.389e-05 | 1.100e-04 |
| `gru2.state` | 24 | 7.451e-08 | 1.122e-04 | 1.532e-04 | 1.488e-04 |
| `gru3.state` | 24 | 1.192e-07 | 8.339e-05 | 1.977e-04 | 1.926e-04 |
| `dense.in` | 96 | 1.192e-07 | 1.122e-04 | 1.977e-04 | 1.926e-04 |
| `dense.gains` | 2 | 2.980e-08 | 1.314e-05 | 1.380e-05 | 1.687e-05 |
| `vad.out` | 1 | 5.960e-08 | 1.955e-05 | 1.901e-05 | 3.833e-05 |

（表里是 `max|Δ|`；`mean|Δ|` 见 `err_all.py` 的输出。）

**第一眼读出来的东西**：

1. **高精度那一列全部 ≤ 1.2e-07（≈2 ulp）** —— 两个实现数学等价，这一列是"地板"。
2. **三列低精度都落在 `1e-05 ~ 2e-04`**，比地板高 **3 个数量级** —— 差异全部来自激活近似。
3. **误差沿着链路单调放大**：`conv1.out`(4.5e-05) → `gru1`(9.4e-05) → `gru3`(2.0e-04)。
   符合"递推系统把每帧的激活误差累积起来"的预期。
4. **`dense.gains` / `vad.out` 反而变小**（1.3e-05 / 1.9e-05，比 `dense.in` 的 1.1e-04 小一个量级）——
   因为最后那层是 sigmoid，它把输入压到 (0,1) 并**压缩**了小误差。
5. `dense.in` 与 `gru3.state` 的 `max|Δ|` 完全相同（都是 1.977e-04 / 1.926e-04），
   因为 `dense.in` 就是"conv2.out + 三个 gru state"的拼接，最大值由其中最大的那项决定。

#### 5.2 未解疑点（随本阶段推进逐条关闭）

两个疑点**都已在阶段 7 关闭**，见下。

### 阶段 7｜放大机制深挖 + 业务层验收 —— **2026-10-01 完成**

（用户追加的两项：q-0「先关递归定位置、再拆单激活定算子」、q-1「做阈值层面的检查」）

- [x] **关递归对照**（`amplify.py: exp_b`）——**放大来自跨帧状态反馈**
      | 层 | 正常 帧15 | 关递归 帧15 |
      |---|---|---|
      | `gru1.state` | 9.389e-05 | **4.363e-05** |
      | `gru2.state` | 1.405e-04 | **4.759e-05** |
      | `gru3.state` | 1.977e-04 | **3.567e-05** |
      关掉后差**不再随帧增长**，且**正好回落到「正常模式第 0 帧」水平**；
      `conv1`/`conv2`（本就无状态）两种模式**逐值完全相同** —— 天然对照 ✓
- [x] **拆单激活**（`amplify.py: exp_c`，靠 `pick()` 新增的 `"sigmoid"`/`"tanh"` 细粒度取值）
      ——**tanh 是罪魁**：只换 tanh 帧 15 得 `1.876e-04`（两个都换是 `1.977e-04`），
      只换 sigmoid 只有 `1.334e-05`。
      `conv1`/`conv2` 只换 sigmoid 的差**恒为 0**（那两层不用 sigmoid）—— 又一个天然对照 ✓
- [x] **系统性 vs 逐帧随机**（`amplify.py: exp_d`）——rcp 同号占比中位 **50%**、符号乱跳；
      换 tanh **54%~67%**、帧 4 起连续 12 帧同号 → **rcp 是随机扰动（不被放大），
      换 tanh 是系统性偏差（相干累积）**
- [x] **关闭疑点 1（反直觉点）**：不是矛盾。真相 = 「两组注入误差的统计性质不同」
      × 「放大只对系统性偏差起作用」。**事实 1**：隔离量上 `A_self ≥ A_swap` 每项成立，
      本来就不反直觉。
- [x] **关闭疑点 2（`dense.in` 的 max 与四分量最大值一致）**：
      解释是"`dense.in` = `conv2.out` + 三个 `gruN.state` 的拼接，最大值由四个分量里最大的决定"。
      `err_all.py: sec_dense_in_check` 抽查 4 组 × 4 帧，**16/16 全部为"是"**。
      （注意不是"永远 gru3 最大"—— 低精度组有些帧是 `gru2.state` 最大。）
- [x] **业务层验收**（`business_check.py`）——判据全部取自源码：
      | 对外量 | 误差 | 相关阈值/台阶 | 判定 |
      |---|---|---|---|
      | `vad.out` | 1.9e-05 | 门限 `> 0.5`（`dump_rnn_io.c:173` + 序列级 Viterbi 平滑） | 比最小裕量（7.757e-02）小 **4083×**，四组判决**完全一致**；真实数据 500 帧里落在门限 ±1.9e-05 的有 **0 帧** |
      | `dense.gains` | 1.3e-05 | 限速台阶 `MAX16(g,.6*lastg)`（`denoise.c:483`） | **数学上 `max|Δ| ≤ δ`，台阶不放大** |
      | `dense.gains` | 相对 2.798e-05 | 16-bit LSB = 3.052e-05 | **0.92 LSB / −91 dBFS**，不可感知 |
- ✓ 验收：§8 遗留第 1、3 条关闭；报告 §5.4 / §9 新增；`unit_check` 仍 **272/272**

#### 阶段 7 的第二轮复核（Agent B，2026-10-01）与处置

B 判「需修正」，抓到 4 条硬的。逐条裁定：

| # | B 的意见 | 裁定与处置 |
|---|---|---|
| 1 | **D 用"同号占比"不显著** —— 24 样本、16 帧下标准误就有 ±10%，58% 与 50% 的差别在 1σ 内 | **采纳（最重要的一条）**。改用**游程检验**：rcp `R=8`（期望 8.88，**p=0.645**，与独立无显著差异）；换 tanh `R=2`（期望 7.00，**p=0.0004**，极显著）。判据硬了 3 个数量级 |
| 2 | **"rcp 是随机扰动"是错的** —— rcp 是确定性近似，同输入两次求值必然相同，没有随机源 | **采纳**。全文改成"符号逐帧**不相干**"，并明确写出"确定性"。这是措辞错误，不是结论错误 |
| 3 | **"tanh 是罪魁"不能外推到输出层** —— `dense.gains` 上 sigmoid 6.318e-06 + tanh 7.093e-06 = 两个都换 1.341e-05（可加） | **采纳**。报告改成"**在 GRU 隐状态上** tanh 主导；到最后一层 `dense.gains`，两者**各占一半**；`vad.out` 上还有相消" |
| 4 | **B 的实验证据边界**：`gru2/gru3` 在关递归模式下上游轨迹也变了，不能单独归因 | **采纳**。报告加了"证据边界"说明：**`gru1` 才是干净证据**（输入 `conv2.out` 不依赖任何 GRU，两模式逐值相同）；gru2/gru3 只能证明"差不再随帧增长" |
| 5 | **业务验收**：不该用"实测最大 1.9e-05"比裕量；缺端到端 PCM 比对；台阶论证只覆盖单帧 | **部分采纳**。①补了**保守上界**（单算子最坏 5.966e-05 × 最大放大 6.52 ≈ 3.9e-04，仍比裕量小 200×）；②端到端 PCM 列入遗留（需要真实音频）；③台阶论证改写成「单帧 1-Lipschitz + 跨帧系数相同的线性递推」，并**明确写出**没做"实测跨台阶帧数" |
| 6 | `pick()` 该改名 `activation_mode` | **不采纳（保留原名）**。它的实现已经是显式分派 + `raise`（B 要的健壮性已满足）；改名要动约 30 个调用点，收益只是命名清晰度。已在报告 §8 第 11 条写明这个取舍 |
| 7 | 补 `pick()` 的 4 种取值单测（含旧 bool 回归） | **采纳**。`rnnoise_activation.py` 加了 `selftest()`（6 种入参 + 5 个非法值），`uv run examples/rnnoise_activation.py` 可跑 |

### 阶段 8｜报告重写 + 清理第一版任务 —— **2026-10-02 完成**

**起因**：用户对报告提出 5 条整改要求 —— 不许有推测/自造术语、要求数值可溯源、
去掉"历史补全痕迹"、重新整理。随后要求清理第一版（单层 GRU）任务及其产物。

**8.1 报告重写**（`doc/RNN_UNIT整网误差分析.md`，~700 行 → **510 行**）

新结构：摘要 → 术语与度量 → 数据流（含每个变量的出处）→ **单个数值的完整溯源** →
测量结果 → 误差归因 → **测试设计评估与结论适用范围** → 复现 → 附（修掉的代码问题）。

术语整改（自造/含糊 → 业界通用）：

| 弃用 | 改用 |
|---|---|
| 分叉 | 误差累积 / 轨迹发散（error accumulation） |
| 端到端 | **自回归推理**（autoregressive inference / free-running）vs **教师强制**（teacher forcing） |
| 种子 / 地板 | 局部误差 / float32 舍入量级（1 ulp 量级） |
| 隔离量 / 隔离差 | 教师强制下的逐层误差（layer-wise error） |
| 逐帧曲线 / 放大倍数 | 逐帧误差 / **增长因子**（error growth factor） |

新增 `scripts/err_stats/trace_value.py`：把一个打印值从 input 开始逐环节溯源
（默认 `conv1.out` 帧 8 分量 5）。**该值不依赖任何 GRU 状态**，
链路是 `conv.json#conv1.inputs[6..8]` → `tmp1`(12) → `y=W1·tmp1+b1` → `tanh` → `conv1.out`；
那 1 ulp 的出处是 **`sgemv` 与 `torch.matmul` 的累加顺序不同**。

**8.2 【重要】修掉一个被掩盖的过时数字**

报告里"网络实际区间 `[−1.01, +0.68]`"这一行（含由它推出的刻度值
2.265e-05 / 4.700e-05 / 6.849e-05 与"可加性精确到 4 位"的结论）
**出自第一版的过时数据**：那时的 `gru` 单元只跑**单层、N=4**（`zrh_recur` 每帧 8 个值）；
当前是**三层、N=24**（每帧 48 个值）。

处置：
- `collect_all.sh` 增加采集 `./rnn_unit gru`（→ `raw_all/c_gru_{high,low}.txt`），
  把数据源并进新工具链，**不再依赖旧目录**。
- `bench_act.py` 里那个硬编码区间改成从 `gru1/2/3.zrh_recur` **读出来算**
  （写死的区间换用例后会静默失效）。实测当前区间为 **`[−2.68, +2.80]`**。
- 报告按当前数据重算：sigmoid 前置激活区间上 `Py多项式−精确` 2.933e-05、
  `C低−Py低` 9.722e-05、`C低−torch` 1.113e-04；tanh 用 sigmoid 的区间作**代理**
  （tanh 的输入 `zrh[2N:]+recur[2N:]*r` 当前没有打印），5.871e-05 / 2.599e-04 / 2.667e-04。
- 删掉"可加性精确到 4 位"那段（那是把 `max` 当可加的误用）；
  改成**构造性论证**：两侧共用同一多项式（`bench_act.py` 实测两套写法在 1601 点上逐位相同），
  所以差值里多项式误差同等地两份、相减即消，剩下的只可能是除法方式不同，
  并检查实测值落在 `_mm256_rcp_ps` 的 3.66e-04 上界内。

**8.3 清理第一版任务**

| 删除 | 原因 |
|---|---|
| `doc/GRU计算过程误差对比统计.md`（371 行） | 内容已被整网报告覆盖；且其中"网络实际区间"一行源自过时数据 |
| `scripts/err_stats/collect.sh` | 功能已被 `collect_all.sh` 覆盖 |
| `scripts/err_stats/make_cases.py` | 只被 `collect.sh` 使用 |
| `scripts/err_stats/raw/`（42 个文件，216K） | 单层 N=4 的过时数据 |

`err_stat.py` **不能整体删**（6 个脚本 import 它的 `parse`/`f32`）——
已拆成纯**数值工具模块**，删掉了它内部的第一版报告生成器（`main()` 与 6 个 `sec_*`）。

**随之失去的一项检查（用户已决定不保留）**：第一版的「1/2/3 帧递推一致性校验」
（截断用例重跑，验证输出与完整 16 帧的前 n 帧逐字节相同）。
它随 `make_cases.py` + `err_stat.py: sec_stage` 一起删除，当前工具链**没有**这项检查。

> **2026-10-02 用户裁定：不保留。** 该项作为待办关闭 —— 不要迁回，
> 也不要在后续报告/文档里把它列成"遗留"。
> 它的历史结论（四组前 1/2/3 帧与 16 帧前缀逐字节一致）记在
> `.codebuddy/memory/2026-09-28.md`，仅作事实留存，不再复现。

> **这个清理本身发现了一个"清理的理由"**：被删的旧数据**已经是过时的**
> （第一版单层 N=4 vs 当前三层 N=24），而报告里有一行引用了它却没标注。
> 留着过时数据 + 不标注出处 = 静默污染。


---

## 3. 待决问题（**没有答案就不要开工**）

| # | 问题 | 为什么不能猜 | 我的建议默认值 | 结论 |
|---|---|---|---|---|
| Q1 | `rnnoise_demo()` 你明确说过不许我碰，但它现在只跑第 0 帧、`vad` 用错变量、命名与 C 不一致 —— 而需求 4 要"每帧全层" | 这是**硬冲突**：不动它就做不出需求 4 | (c) 不碰它，另建镜像 C 计算图的 harness | **已定：授权改最小必要三处**（去 `break`、修 `vad` 变量、统一命名），其余不动 |
| Q2 | `all` 链路的 C 侧要不要补 gru 的**内部量**（`zrh_recur`/`sigmoid`/`recur_tanh`/`h`）？现在只有 `state` | 要的话得把 `conv_gru_unit` 里的 `compute_generic_gru` 换成 `local_compute_generic_gru`；不要的话 gru 内部量只能在上个任务的 `gru` 单元里比，而**那里的 gru1 输入是 gru.json 自己的 inputs，不是 conv2 输出** | (a) 要，改用本地版本（**不动 `src/`**） | **已定：不要**，`all` 只比 `state` |
| Q3 | 方法论：是否接受「逐层 teacher forcing + 单算子微基准 + 四组 pairwise」，而不是做一份全链路 float64 真值？ | 全链路 float64 成本高、收益递减；但这是"根因分析"能做多深的分水岭 | 接受 | **待定（用户在追问名词含义）** |
| Q4 | `gru.py` 里剩下的 RNN 教学代码（`rnn()` / `RNNScratch` / `rnn_demo()`）怎么办？ | 挪走等于改你的学习路径 | 暂时留在 `gru.py`，加 TODO | **已定：暂时留在 `gru.py`**（加一行 TODO） |
| Q5 | `_check_sigmoid_diff` 搬过去时修还是删？（它引用了不存在的 `_rnnoise_sigmoid`，且没人调用） | 它是死代码 | 搬过去并**修好**，让它真能跑（同时保留 C 的 `0.5+0.5*tanh_approx(0.5x)` 版本做对照） | 默认（用户未反对） |
| Q6 | 新文件名 | — | `examples/rnnoise_activation.py` | 默认（用户未反对） |
| Q7 | 报告写哪 | — | 新建 `doc/RNN_UNIT整网误差分析.md`（旧的 `GRU计算过程误差对比统计.md` 只管 gru 单元） | 默认（用户未反对） |
| Q8 | 协议命名统一到哪套 | — | 以 **C 侧**为准（`conv1.out`/`conv2.out`/`gruN.state`/`dense.in`/`dense.gains`/`vad`） | 默认（用户未反对） |
| Q9 | `all` 单元要不要纳入 `scripts/unit_check.py` 的覆盖？ | 它没有 `all.json`，`discover_units()` 只看 json 的 `unit` 字段，所以现在根本没覆盖 | 纳入（给它一个 `all.json` 只放 `unit`/`tolerance`） | 默认（用户未反对） |

---

## 4. 风险与对策

| 风险 | 对策 |
|---|---|
| automake 不因 CFLAGS 变化重编 `.o` → 采到"假同精度"数据 | 切档脚本里强制 `touch src/nnet_default.c`，并断言 `[Notice]` 行 |
| 9 位有效数字不足以还原 float32 | 一律走 `f32()` 还原；自检"差只能是 0 或 ≥1 ulp" |
| `gruN.state` 打 72 但只写了 24 → 误差表里会出现 1e0 量级假误差 | 阶段 3 先修长度，再进阶段 4 |
| 改名/拆分打破 272 项基线 | 阶段 1 结束立刻跑 `unit_check.py`，不过不进下一阶段 |
| `conv2` / `gru2` / `gru3` 在 json 里没有 `inputs` | 它们只能**级联**跑，不能独立跑；"每帧每层"必须按级联后的定义 |
| `py_low` 现在不是真的 low | 阶段 2 不完成，阶段 4 的四组对比**没有一组可信** |
| GRU 是递推系统，第 15 帧误差大可能只是轨迹分叉 | 用 teacher forcing 隔离 + 逐帧误差曲线区分 |

---

## 5. 我认为用户考虑不足的地方

1. **「16 帧」不是"每层各 16 帧"。** `conv.json` 只有 `conv1` 有 `inputs`，`conv2` 没有；
   `gru.json` 只有 `gru1` 有。`conv2` / `gru2` / `gru3` 的输出**完全依赖级联喂入**，
   独立跑是 **0 帧**。所以"每帧所有层"只能在**级联链路**里定义，不能按"每层各跑 16 帧"理解。

2. **`py_low` 现在根本不是 low。** Python 侧 `conv1` / `conv2` / `dense_out` / `vad_dense` 四层
   恒用精确的 `torch.tanh` / `torch.sigmoid`，而 C 侧这四层在低精度档走的是多项式近似。
   也就是说**需求 3 不做完，需求 4 的四组对比没有一组是可信的**（需求 3 正是需求 4 的前提）。

3. **四组互相比较，在没有"真值"的情况下定位不了根因。** 缺 float64/float32 精确参考时，
   `py_high vs py_low` 差得大只能说明"近似有代价"，说不清是 sigmoid 多项式、tanh 多项式、
   还是 GRU 递归放大的。而且 GRU 是递推系统，第 k 帧的差里混着前 k-1 帧的传播。

4. **`conv_gru_unit()` 打 72 个 state 不是"多打点更保险"，是数据污染源** —— 48 个陈旧值会被解析进误差表。

5. **协议不一致不只是"名字不同"**：py 缺 `dense.in`、多一个 `gains.out`、只跑 1 帧，
   再加上 `rnnoise_demo()` 里混着的非协议 `print`，会让任何自动解析**静默丢帧**（不报错，直接少数据）。

6. **`all` 单元当前完全不在测试覆盖里**（`discover_units()` 只看 json 的 `unit` 字段，而 `all` 没有
   自己的 json），所以"整网链路"这件事至今没有任何回归保护 —— 改坏了不会有人告诉你。

7. **C 侧 ABI 级别的隐式约束没写下来**：`conv2` 的输出要复用同一块 `cat[]` 缓冲（`conv2` 写前 24 个，
   `gru` 的三个 state 写在 24/48/72 处），这种"一块缓冲多处写"的写法一旦偏移算错就是静默错值。
   建议在报告里把每层输出的**偏移表**明确列出来，或用独立数组 + 最后 `memcpy` 拼。

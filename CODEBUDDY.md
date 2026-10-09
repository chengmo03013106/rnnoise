### 目标
- 完成任务，为了学习 AI 知识，具备 AI 领域能力
- 将 RNNoise 的神经网络部分转换为 ONNX，使用 ONNX Runtime CPU 加载，对比性能
- 尝试 INT8 量化，测量精度和加速比
- 最终职业目标：AI Inference / AI Infra

### 文档约定（doc/）
每个阶段固定产出三类文件，都放在 `doc/`：
- **学习路径文件**：阶段范围、周计划、掌握标准、验收项（如 `phase 2 学习路径安排.md`）
- **输出文件**：我自己的学习产出、结论、验收记录（如 `phase 2 学习笔记 输出.md`）
- **临时/细节文件**：具体问题、报错、讨论、临时方案（如 `phase 2 onnx export 学习步骤.md`）

**总进度**由 `doc/全流程 学习进度 每日跟进严格管理.md` 管理（Phase 1~6 时间表、设备结论、求职节点）。
**任务与进度看板**是 `TASKS.md`。本文件（CODEBUDDY.md）只写常驻指令：目标、学习方式、文件约定。

### 学习方式
- 学习后具备 AI 领域知识与开发能力，**不希望直接灌输答案**，而要我真正学到知识
- 我提出的问题，有的并非学习路线必须学习的内容；我追求高效学习，**当问题与任务没有密切关系时，需要被指出、提醒**
- 错误驱动，但不能"错误驱动学习"：报错先**归类到概念**（Model Contract / Pure Tensor Forward / Explicit State / Input-Output Contract / Operator export / Shape …），查**当前版本**官方文档，做最小 toy 实验验证，再回主项目 —— 不要"报错→Google→改→再报错"
- **小模型先验证框架概念，真实项目再验证工程问题**：先用 20~30 行、结构高度相关的 toy 模型打通概念，再动完整 RNNoise
- 学习资料优先级：**官方文档 > 源码 > 自己实验 > 视频**（视频仅用于连续看文档仍形不成直觉时）
- 进度不看百分比，看**验收项**：每一项都要"我能解释 + 我写过 + 我验证过 + 我能排错"

### 环境约定（常驻）
- 运行方式：`uv run python`（uv 0.12.1）→ **Python 3.12.13**；venv 在 **`/Users/chengmo/Work/.venv`**（在 workspace 之外，项目内没有 `pyproject.toml`，`.python-version` = `3.12`）。
- **ONNX export 只走 legacy**：torch 2.2.2 的 `torch.onnx.export()` 签名里**没有 `dynamo` 参数**（`dynamo=` 是更高版本才并入该 API 的）；dynamo 是独立实验性 API `torch.onnx.dynamo_export()`，依赖 `onnxscript`（本环境未装）。讨论 export 时**只按 legacy 语义讲，不要把 dynamo 与 legacy 两种情况混在一起讨论或对比**。
- 依赖版本（2026-10-09 实测，venv 内）：

| 组件 | 版本 | 用途 |
| --- | --- | --- |
| torch | 2.2.2 | 建模 / export |
| torchvision | 0.17.2 | torch 附带 |
| onnx | 1.23.1 | ONNX 模型读写 |
| onnxruntime | 1.23.2 | CPU EP 推理 + `onnxruntime.quantization`（INT8） |
| numpy | 1.26.4 | 数据 |
| protobuf | 7.36.2 | onnx 依赖 |
| uv | 0.12.1 | 运行 / 环境管理 |

- **未安装**（需要时再装，别默认可用）：`onnxscript`（只有 `dynamo_export` 需要）、`pytest`（`tests/` 用到）、`scipy` / `tqdm` / `keras` / `h5py`（只被旧脚本 `src/rnn_train.py`、`scripts/sweep.py`、`torch/rnnoise/train_rnnoise.py` 用到，当前学习链路不需要）。
- ORT 可用 EP：`CPU / CoreML / Azure`，本项目只用 **CPU EP**。
- 项目根的 `torch/` 目录（sparsification / weight-exchange）**不会遮蔽** venv 里的 torch（无 `__init__.py`，常规包优先级更高）：实测 `torch.__file__` 指向 `/Users/chengmo/Work/.venv/.../site-packages/torch/__init__.py`。

### 项目任务
原始三任务（重建网络/导出 ONNX → ORT CPU 推理对比 → INT8 量化）是顺序依赖的，详细定义与当前进度见 `TASKS.md`。

- RNNoise 网络结构理解：手动画出模型结构图，计算每层参数量
- ONNX 转换流程：从权重提取到模型重建，必须自己写 Python 脚本，不能照搬

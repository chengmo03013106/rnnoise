/*
 * rnn_unit 公共头：宏定义 + 用例结构体 + 跨文件共享声明
 * filename: examples/rnn_unit.h
 *
 * 本文件按"只放一类东西"的原则拆分而来：
 *     - 宏定义 + 用例结构体 -> 本文件 (rnn_unit.h)
 *     - 辅助函数            -> examples/rnn_unit_util.h / .c
 *     - 单元实现            -> examples/rnn_unit.c
 */
#ifndef RNN_UNIT_H
#define RNN_UNIT_H

#include <stdint.h>

/* ===================== 宏定义 =====================
 *
 * 用例规模 = 真实模型尺寸 // 16（统一比例，这样层与层之间的接口宽度才对得上）。
 * 真实尺寸见 src/rnn.c:48-52 与 src/rnnoise_data.h。
 *
 * 真实链路（每帧）                    线性层(nb_inputs -> nb_outputs)
 *     conv1 : 65  -> 128              195 = 3x65     -> 128
 *     conv2 : 128 -> 384              384 = 3x128    -> 384   (输出 = 输入)
 *     gruN  : 384 -> 384 = N          384            -> 3N = 1152
 *
 * //16 之后（本文件用的用例规模）:
 *     conv1 : 每帧输入 4  -> 输出 8    线性层 12 -> 8
 *     conv2 : 每帧输入 8  -> 输出 24   线性层 24 -> 24
 *     gruN  : 每帧输入 24 -> N 24      线性层 24 -> 3N = 72
 *
 * 三个接口关系（两侧代码和 json 都必须满足）:
 *     1) conv2 的线性宽度 = 3 x conv1 的输出宽；
 *     2) gru 的每帧输入宽 = conv2 的输出宽；gru 的隐状态 N 也等于它。
 *
 * 宏名统一带 CASE_ 前缀：src/rnnoise_data.h 里已经有真实模型的
 * CONV1_IN_SIZE(65) / CONV2_IN_SIZE(128)，不加前缀会撞名，撞上之后
 * 谁后包含谁生效，维数会静默变成错的。
 */
#define CASE_NB_FRAMES 16                                    /* 用例允许的最大帧数，两侧必须一致 */

#define CASE_CONV1_IN 4
#define CASE_CONV1_OUT  8
#define CASE_CONV2_IN 8
#define CASE_CONV2_OUT  24

/* 两层共用同一份实现，数组按两层的上限开；实际维度放结构体里，运行时给 */
#define CASE_CONV_MAX_IN  CASE_CONV2_IN
#define CASE_CONV_MAX_OUT CASE_CONV2_OUT

/* GRU：每帧输入宽 = conv2 输出宽 = N。
 * 三层（gru1/gru2/gru3）真实规模相同，所以共用一套宏。
 * src/nnet.c:74-76 的硬约束：nb_outputs = 3N，且 N = recurrent 权重矩阵的 nb_inputs */
#define CASE_GRU_N          (24)
#define CASE_GRU_IN (24)
#define GRU_STATE_SIZE (CASE_GRU_IN)
#define CASE_GRU_OUT     (3*CASE_GRU_N)

/* 整网输出层（对应 src/rnnoise_data.h 的 dense_out / vad_dense）:
 *   输入宽 = 拼接后的宽度 = conv2 输出 + gru1/2/3 输出 = 24 + 3x24 = 96（真实 1536）
 *   dense_out 32 -> 2（//16），vad_dense 1 -> 1（单标量输出，不再缩）
 */
#define CASE_DENSE_IN    (CASE_CONV2_OUT + 3*CASE_GRU_N)
#define CASE_DENSE_OUT     2
#define CASE_VAD_OUT 1

/* 两层 dense 共用同一份实现，数组按上限开 */
#define CASE_DENSE_MAX_IN  CASE_DENSE_IN
#define CASE_DENSE_MAX_OUT CASE_DENSE_OUT

/* ===================== 跨文件共享声明 ===================== */

/* 被测算子使用的架构编号（0 = 纯 C 参考实现），定义在 examples/rnn_unit.c */
extern const int arch;

/* ===================== 用例数据结构 =====================
 *
 * 权重布局和 C 的 sgemv 一致：weights[输入 j][输出 i]，
 * 行数 = 3 × nb_in（mem 2 帧 + 本帧），列数 = nb_out。
 * 见 src/nnet_arch.h:140 的 sgemv(..., N, M, N, in)。
 *
 * layer/nb_in/nb_out 是运行时字段：conv1d 单元要同时跑两层，两层规模不同，
 * 所以数组按 CASE_CONV_MAX_* 开，具体用多少由 load_conv1d_case() 填。
 */

/* ---- conv1d 用例 ----
 *   bias          : 长度 = nb_out
 *   float_weights : 长度 = (3*nb_in) * nb_out
 *   int8_weights  : 同形状的量化版。当前用例 JSON 里**没有**这一项（整网只走 float 那条路），
 *                   数组留着是为了不动 conv1d_unit 的代码；`float_weights` 非 NULL 时
 *                   compute_linear 走 sgemv(float)（src/nnet_arch.h:138），int8 那条路不会被执行。
 *   inputs        : 逐帧输入，共 nb_frames 帧，每帧 nb_in 个数。
 *                   **可选**：conv2 在整网里由 conv1 的输出喂，用例里就没有这一项，
 *                   这时 nb_frames = 0；有这一项但格式不对，仍然报错。
 */
typedef struct {
  const char *layer;                          /* 层名（conv1 / conv2），用于拼 JSON 路径和输出名 */
  int    nb_in;
  int    nb_out;
  int    nb_frames;
  float  bias[CASE_CONV_MAX_OUT];
  float  float_weights[3*CASE_CONV_MAX_IN*CASE_CONV_MAX_OUT];
  int8_t int8_weights[3*CASE_CONV_MAX_IN*CASE_CONV_MAX_OUT];
  float  inputs[CASE_NB_FRAMES][CASE_CONV_MAX_IN];
} Conv1dCase;

/* ---- gru 用例（gru1 / gru2 / gru3 共用同一份结构）----
 *   input_weights     : nb_inputs = CASE_GRU_IN, nb_outputs = CASE_GRU_OUT
 *   recurrent_weights : nb_inputs = CASE_GRU_N,          nb_outputs = CASE_GRU_OUT
 *   inputs            : 逐帧输入，共 nb_frames 帧，每帧 CASE_GRU_IN 个数。
 *                       **可选**：gru2 / gru3 在整网里由上一层的隐状态喂，用例里就没有这一项，
 *                       这时 nb_frames = 0；有这一项但格式不对，仍然报错。
 *   state             : 初始隐状态，长度 = CASE_GRU_N（两侧递推前都会清零，这里只做完整性校验）
 *   input_subias / recurrent_subias : 当前用例 JSON 里**没有**这一项（它只在
 *                   int8 权重 + USE_SU_BIAS 那条路上生效），字段留着是为了不动 gru_unit 的代码。
 */
typedef struct {
  const char *layer;                          /* 层名（gru1 / gru2 / gru3） */
  int    nb_frames;
  float  input_weights[CASE_GRU_IN*CASE_GRU_OUT];
  float  input_bias[CASE_GRU_OUT];
  float  input_subias[CASE_GRU_OUT];
  float  recurrent_weights[CASE_GRU_N*CASE_GRU_OUT];
  float  recurrent_bias[CASE_GRU_OUT];
  float  recurrent_subias[CASE_GRU_OUT];
  float  inputs[CASE_NB_FRAMES][CASE_GRU_IN];
  float  state[CASE_GRU_N];
} GruCase;

/* ---- dense 用例（dense_out / vad_dense 共用同一份结构）----
 *   bias          : 长度 = nb_out
 *   float_weights : 长度 = nb_in * nb_out
 *   dense 没有记忆、没有输入序列，所以这里只有参数，没有 inputs
 */
typedef struct {
  const char *layer;                          /* 层名（dense_out / vad_dense） */
  int    nb_in;
  int    nb_out;
  float  bias[CASE_DENSE_MAX_OUT];
  float  float_weights[CASE_DENSE_MAX_IN*CASE_DENSE_MAX_OUT];
} DenseCase;

/* ---- 用例读取（实现见 examples/rnn_unit_util.c） ----
 * 严格模式：path 为空、文件打不开、解析失败、或任何必需字段缺失/长度不对，
 * 都打印错误并返回 -1（调用方据此退出程序），**不再退回内置保底数据**。
 * 字段名和校验规则必须和 examples/rnn_unit.py 的 _load_*_case() 完全一致。
 *
 * 唯一例外：`inputs` 是可选的 —— conv2 由 conv1 的输出喂、gru2 / gru3 由上一层的隐状态喂，
 * 用例里干脆不写这一项（这时 nb_frames = 0）。写了但格式不对，仍然报错。
 */
int load_conv1d_case(Conv1dCase *c, const char *path, const char *layer, int nb_in, int nb_out);
int load_gru_case(GruCase *c, const char *path, const char *layer);
int load_dense_case(DenseCase *c, const char *path, const char *layer, int nb_in, int nb_out);

#endif /* RNN_UNIT_H */

/*
 * rnn_unit 公共头：宏定义 + 跨文件共享声明
 * filename: examples/rnn_unit.h
 *
 * 本文件按"只放一类东西"的原则拆分而来：
 *     - 宏定义      -> 本文件 (rnn_unit.h)
 *     - 用例参数    -> examples/rnn_unit_data.h
 *     - 辅助函数    -> examples/rnn_unit_util.h / .c
 *     - 单元实现    -> examples/rnn_unit.c
 */
#ifndef RNN_UNIT_H
#define RNN_UNIT_H

/* ===================== 宏定义 ===================== */

/* 用例规模：conv1d 单元的输入/输出维度与帧数 */
#define NB_INPUTS  4
#define NB_OUTPUTS 8
#define NB_FRAMES  16

/* gru1 用例规模：N = 4 个隐层单元，输入宽 4，三个门拼成的输出宽 3N = 12。
 * src/nnet.c:74-76 的硬约束：nb_outputs = 3N，且 N = recurrent 权重矩阵的 nb_inputs。 */
#define GRU1_N          (4)
#define GRU1_INPUT_SIZE (4)
#define GRU1_NB_OUT     (3*GRU1_N)

/* ===================== 跨文件共享声明 ===================== */

/* 被测算子使用的架构编号（0 = 纯 C 参考实现），定义在 examples/rnn_unit.c */
extern const int arch;

/* ---- gru1 用例：结构体 + 读取函数（实现见 examples/rnn_unit.c） ----
 * 权重布局和 C 的 sgemv 一致：weights[输入 j][输出 i]，
 * 数组长度 = nb_inputs * nb_outputs（见 src/nnet_arch.h:140 的 sgemv(..., N, M, N, in)）。
 *   input_weights     : nb_inputs = GRU1_INPUT_SIZE, nb_outputs = GRU1_NB_OUT
 *   recurrent_weights : nb_inputs = GRU1_N,          nb_outputs = GRU1_NB_OUT
 *   inputs            : 逐帧输入，共 nb_frames 帧，每帧 GRU1_INPUT_SIZE 个数
 *   state             : 初始隐状态，长度 = GRU1_N（逐帧递推时会被原地更新）
 */
typedef struct {
  float input_weights[GRU1_INPUT_SIZE*GRU1_NB_OUT];
  float input_bias[GRU1_NB_OUT];
  float input_subias[GRU1_NB_OUT];
  float recurrent_weights[GRU1_N*GRU1_NB_OUT];
  float recurrent_bias[GRU1_NB_OUT];
  float recurrent_subias[GRU1_NB_OUT];
  float inputs[NB_FRAMES][GRU1_INPUT_SIZE];
  int   nb_frames;
  float state[GRU1_N];
} GruCase;

/* 逐字段读取 + 逐字段校验，不合格的字段单独退回保底值。
 * 规则必须和 examples/rnn_unit.py 的 load_case() 完全一致。
 * 这里刻意不是 static：gru_unit() 还没写，非 static 可以避免 -Wunused-function。 */
void load_gru_case(GruCase *c, const char *path);

#endif /* RNN_UNIT_H */

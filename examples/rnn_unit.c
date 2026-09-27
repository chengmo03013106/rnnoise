/*
 * 个人练习使用
 * 调用神经网络相关的 c 函数，做单元测试；输出与 PyTorch 实现逐项对比
 * filename: examples/rnn_unit.c
 *
 * 输出协议（一行一项，方便脚本比对）:
 *     [<unit>.<output>#<frame>] <v0> <v1> ...
 * 例:
 *     [conv1d.out#0] -8.39260000e-02 8.15400000e-03 ...
 *
 * 用例数据来源（和 py 侧完全一致），按顺序尝试:
 *     ① argv[2] 显式指定的 JSON
 *     ② 按约定自动定位 <仓库根>/scripts/unit_cases/<unit>.json
 *     ③ 都拿不到，才用编译进来的保底数据（此时会打醒目警告）
 *
 * 用法:
 *     ./rnn_unit conv1d                                   # 自动找用例文件
 *     ./rnn_unit conv1d scripts/unit_cases/conv1d.json    # 显式指定
 *
 * 构建: make rnn_unit   （源码列表见 Makefile 的 rnn_unit_SOURCES）
 * 对比: uv run scripts/unit_check.py [unit...]
 *
 * 文件结构（本文件只留"主流程 + 各单元实现"，其余按类别拆出去）:
 *     rnn_unit.h       宏定义 + 跨文件共享声明
 *     rnn_unit_data.h  保底参数数据
 *     rnn_unit_util.h  辅助函数声明
 *     rnn_unit_util.c  辅助函数实现（JSON 读取 / 输出协议 / 用例定位）
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rnnoise.h"
#include "common.h"
#include "denoise.h"
#include "nnet.h"

#include "rnn_unit.h"
#include "rnn_unit_data.h"
#include "rnn_unit_util.h"

const int arch = 0;

/* ===================== 单元: conv1d =====================
 * 对应 src/nnet.c: compute_generic_conv1d()
 * 它做的事: tmp = [mem, input] 拼起来 -> 过一层线性 -> 过激活 -> 把 tmp 的尾部存回 mem
 */

typedef struct {
  float  bias[NB_OUTPUTS];
  float  float_weights[NB_OUTPUTS*NB_INPUTS*3];
  int8_t int8_weights[NB_OUTPUTS*NB_INPUTS*3];
  float  inputs[NB_FRAMES][NB_INPUTS];
  int    nb_frames;
} Conv1dCase;

/* 逐字段读取 + 逐字段校验，不合格的字段单独退回保底值。
 * 规则必须和 examples/rnn_unit.py 的 load_case() 完全一致，否则两边会读到不同的数据。 */
static void load_conv1d_case(Conv1dCase *c, const char *path) {
  static UjDoc doc;
  static float tmp_inputs[NB_FRAMES][NB_INPUTS];
  const char *src_bias = "builtin", *src_w = "builtin", *src_in = "builtin";
  int i, j, nb_tmp = 0;

  /* 1. 先铺满编译进来的保底数据 */
  memcpy(c->bias, default_bias, sizeof(c->bias));
  memcpy(c->float_weights, default_float_weights, sizeof(c->float_weights));
  memcpy(c->int8_weights, default_int8_weights, sizeof(c->int8_weights));
  memcpy(c->inputs, default_inputs, sizeof(default_inputs));
  c->nb_frames = (int)(sizeof(default_inputs) / sizeof(default_inputs[0]));

  /* 2. 能读到 JSON 就逐字段覆盖；每个字段各自做长度校验 */
  if (path != NULL) {
    int st = uj_load(&doc, path);
    if (st < 0) fprintf(stderr, "[rnn_unit] json error: parse failed\n");
    if (st == 1) {
      if (uj_get_floats(&doc, "bias", c->bias, NB_OUTPUTS)) src_bias = "json";
      if (uj_get_floats(&doc, "float_weights", c->float_weights, NB_OUTPUTS*NB_INPUTS*3)) src_w = "json";
      uj_get_int8(&doc, "int8_weights", c->int8_weights, NB_OUTPUTS*NB_INPUTS*3);

      /* 帧要么全部合格，要么整段不采纳，否则会读到半截数据。
       * 这条规则必须和 py 侧 _is_frames() 一致。 */
      for (i = 0; i < NB_FRAMES; i++) {
        char key[32];
        int got = 0;
        const double *v;
        snprintf(key, sizeof(key), "inputs[%d]", i);
        v = uj_get(&doc, key, &got);
        if (v == NULL) break;                        /* 数组到头，正常结束 */
        if (got != NB_INPUTS) { nb_tmp = 0; break; } /* 帧内元素数不对 */
        for (j = 0; j < NB_INPUTS; j++) tmp_inputs[i][j] = (float)v[j];
        nb_tmp++;
      }
      if (nb_tmp == NB_FRAMES) {                     /* 可能被 NB_FRAMES 截断了 */
        char key[32];
        int got = 0;
        snprintf(key, sizeof(key), "inputs[%d]", NB_FRAMES);
        if (uj_get(&doc, key, &got) != NULL) {
          fprintf(stderr, "[rnn_unit] json error: 帧数超过上限 %d，inputs 字段不采纳\n", NB_FRAMES);
          nb_tmp = 0;
        }
      }
      if (nb_tmp > 0) {
        memcpy(c->inputs, tmp_inputs, sizeof(tmp_inputs));
        c->nb_frames = nb_tmp;
        src_in = "json";
      }
    }
  }

  fprintf(stderr, "[rnn_unit] case data: bias=%s weights=%s inputs=%s%s%s\n",
          src_bias, src_w, src_in, path ? " path=" : "", path ? path : "");

  if (!strcmp(src_bias, "builtin") && !strcmp(src_w, "builtin") && !strcmp(src_in, "builtin")) {
    fprintf(stderr, "[rnn_unit] 注意: 三个字段全部用了内置保底数据，本次对比只验证代码通路，不验证真实用例数据。\n");
    if (path == NULL)
      fprintf(stderr, "[rnn_unit] 没找到用例文件 scripts/unit_cases/<unit>.json\n");
  }
}

static int conv1d_unit(const char *case_path) {
  LinearLayer linear;
  static Conv1dCase sc;
  float out[NB_OUTPUTS];
  float mem[NB_INPUTS*2];        /* 必须 >= layer->nb_inputs - input_size */
  int f;

  load_conv1d_case(&sc, case_path);

  memset(&linear, 0, sizeof(linear));
  linear.bias = sc.bias;
  linear.float_weights = sc.float_weights;   /* 非 NULL 时 compute_linear 走 sgemv(float) */
  linear.weights = sc.int8_weights;
  linear.nb_inputs = NB_INPUTS * 3;
  linear.nb_outputs = NB_OUTPUTS;

  memset(mem, 0, sizeof(mem));
  for (f = 0; f < sc.nb_frames; f++) {
    compute_generic_conv1d(&linear, out, mem, sc.inputs[f], NB_INPUTS, ACTIVATION_TANH, arch);
    print_item("conv1d.out", f, out, NB_OUTPUTS);
    print_item("conv1d.mem", f, mem, NB_INPUTS*2);
  }
  return 0;
}

/* ===================== 单元: gru =====================
 * 对应 src/nnet.c: compute_generic_gru()
 * 它做的事:
 *   zrh   = W_in  * x     + b_in          // 输入投影
 *   recur = W_rec * state + b_rec         // 递归投影
 *   z,r   = sigmoid(zrh[0:2N]   + recur[0:2N])
 *   h     = tanh(   zrh[2N:3N]  + recur[2N:3N] .* r)
 *   state = z*state + (1-z)*h
 * 用例规模: N = GRU1_N(4)、输入宽 GRU1_INPUT_SIZE(4)、三个门拼成的输出宽 3N = 12。
 */

/* 逐字段读取 + 逐字段校验，不合格的字段单独退回保底值。
 * 规则必须和 examples/rnn_unit.py 的 load_case() 完全一致，否则两边会读到不同的数据。 */
void load_gru_case(GruCase *c, const char *path) {
  static UjDoc doc;
  static float tmp_inputs[NB_FRAMES][GRU1_INPUT_SIZE];
  const char *src_iw = "builtin", *src_ib = "builtin", *src_is = "builtin";
  const char *src_rw = "builtin", *src_rb = "builtin", *src_rs = "builtin";
  const char *src_in = "builtin", *src_st = "builtin";
  int i, j, nb_tmp = 0;

  /* 1. 先铺满编译进来的保底数据 */
  memcpy(c->input_weights, default_gru1_input_weights, sizeof(c->input_weights));
  memcpy(c->input_bias, default_gru1_input_bias, sizeof(c->input_bias));
  memcpy(c->input_subias, default_gru1_input_subias, sizeof(c->input_subias));
  memcpy(c->recurrent_weights, default_gru1_recurrent_weights, sizeof(c->recurrent_weights));
  memcpy(c->recurrent_bias, default_gru1_recurrent_bias, sizeof(c->recurrent_bias));
  memcpy(c->recurrent_subias, default_gru1_recurrent_subias, sizeof(c->recurrent_subias));
  memcpy(c->inputs, default_gru1_inputs, sizeof(default_gru1_inputs));
  c->nb_frames = (int)(sizeof(default_gru1_inputs) / sizeof(default_gru1_inputs[0]));
  memcpy(c->state, default_gru1_state, sizeof(c->state));

  /* 2. 能读到 JSON 就逐字段覆盖；每个字段各自做长度校验 */
  if (path != NULL) {
    int st = uj_load(&doc, path);
    if (st < 0) fprintf(stderr, "[rnn_unit] json error: parse failed\n");
    if (st == 1) {
      if (uj_get_floats(&doc, "gru1_input_weights_float", c->input_weights,
                        GRU1_INPUT_SIZE*GRU1_NB_OUT)) src_iw = "json";
      if (uj_get_floats(&doc, "gru1_input_bias", c->input_bias, GRU1_NB_OUT)) src_ib = "json";
      if (uj_get_floats(&doc, "gru1_input_subias", c->input_subias, GRU1_NB_OUT)) src_is = "json";
      if (uj_get_floats(&doc, "gru1_recurrent_weights_float", c->recurrent_weights,
                        GRU1_N*GRU1_NB_OUT)) src_rw = "json";
      if (uj_get_floats(&doc, "gru1_recurrent_bias", c->recurrent_bias, GRU1_NB_OUT)) src_rb = "json";
      if (uj_get_floats(&doc, "gru1_recurrent_subias", c->recurrent_subias, GRU1_NB_OUT)) src_rs = "json";
      if (uj_get_floats(&doc, "state", c->state, GRU1_N)) src_st = "json";

      /* 帧要么全部合格，要么整段不采纳，否则会读到半截数据。
       * 这条规则必须和 py 侧 _is_frames() 一致。 */
      for (i = 0; i < NB_FRAMES; i++) {
        char key[32];
        int got = 0;
        const double *v;
        snprintf(key, sizeof(key), "inputs[%d]", i);
        v = uj_get(&doc, key, &got);
        if (v == NULL) break;                              /* 数组到头，正常结束 */
        if (got != GRU1_INPUT_SIZE) { nb_tmp = 0; break; } /* 帧内元素数不对 */
        for (j = 0; j < GRU1_INPUT_SIZE; j++) tmp_inputs[i][j] = (float)v[j];
        nb_tmp++;
      }
      if (nb_tmp == NB_FRAMES) {                           /* 可能被 NB_FRAMES 截断了 */
        char key[32];
        int got = 0;
        snprintf(key, sizeof(key), "inputs[%d]", NB_FRAMES);
        if (uj_get(&doc, key, &got) != NULL) {
          fprintf(stderr, "[rnn_unit] json error: 帧数超过上限 %d，inputs 字段不采纳\n", NB_FRAMES);
          nb_tmp = 0;
        }
      }
      if (nb_tmp > 0) {
        memcpy(c->inputs, tmp_inputs, sizeof(tmp_inputs));
        c->nb_frames = nb_tmp;
        src_in = "json";
      }
    }
  }

  fprintf(stderr, "[rnn_unit] case data:"
                  " gru1_input_weights_float=%s gru1_input_bias=%s gru1_input_subias=%s"
                  " gru1_recurrent_weights_float=%s gru1_recurrent_bias=%s gru1_recurrent_subias=%s"
                  " inputs=%s state=%s%s%s\n",
          src_iw, src_ib, src_is, src_rw, src_rb, src_rs, src_in, src_st,
          path ? " path=" : "", path ? path : "");

  if (!strcmp(src_iw, "builtin") && !strcmp(src_ib, "builtin") && !strcmp(src_is, "builtin") &&
      !strcmp(src_rw, "builtin") && !strcmp(src_rb, "builtin") && !strcmp(src_rs, "builtin") &&
      !strcmp(src_in, "builtin") && !strcmp(src_st, "builtin")) {
    fprintf(stderr, "[rnn_unit] 注意: 八个字段全部用了内置保底数据，本次对比只验证代码通路，不验证真实用例数据。\n");
    if (path == NULL)
      fprintf(stderr, "[rnn_unit] 没找到用例文件 scripts/unit_cases/<unit>.json\n");
  }
}

/* ===================== 待补单元 =====================
 * conv1d 跑通后，按同样的模式往下加：
 *   - linear_unit  : compute_linear    (sgemv / cgemv8x4 / sparse_*)
 *   - dense_unit   : compute_generic_dense
 *   - gru_unit     : compute_generic_gru   （load_gru_case() 已写好，还缺 xxx_unit + main 注册）
 *   - activation_unit : compute_activation (tanh_approx / sigmoid_approx)
 * 每个单元: 一个 load_xxx_case() + 一个 xxx_unit(case_path)，并在 main 里注册。
 *
 * 注意: gru_unit 目前只是**占位骨架**，还没实现，作用是让整个文件先编译得过。
 *       GruCase 的字段含义、两个 LinearLayer 各要填什么规格，见 examples/rnn_unit.h。
 */


static int gru_unit( const char * case_path) {

  LinearLayer gru_input;
  memset(&gru_input,0,sizeof(gru_input));

  LinearLayer gru_recurrent;
  memset(&gru_recurrent,0,sizeof(gru_recurrent));

  static GruCase gc;
  load_gru_case(&gc, case_path);
  
  gru_input.bias = gc.input_bias;
  gru_input.subias = gc.input_subias;
  gru_input.float_weights = gc.input_weights;
  gru_input.nb_inputs = GRU1_INPUT_SIZE;
  gru_input.nb_outputs = GRU1_NB_OUT ;

  gru_recurrent.bias = gc.recurrent_bias;
  gru_recurrent.subias = gc.recurrent_subias;
  gru_recurrent.float_weights = gc.recurrent_weights;   // 非 NULL 时 compute_linear 走 sgemv(float)
  gru_recurrent.nb_inputs = GRU1_N;
  gru_recurrent.nb_outputs = GRU1_NB_OUT;

  static float gru_state[GRU1_N];
  // memcpy(gru_state, gc.state, sizeof(float)*GRU1_N);
  
  for(int i = 0; i < NB_FRAMES; ++i) {
    float * inputs = gc.inputs[i];
    compute_generic_gru(&gru_input, &gru_recurrent, gru_state, inputs, 0);
    print_item("gru.state", i, gru_state, NB_INPUTS*2);
  }
  

  return 0;
}

int main(int argc, char **argv) {
  const char *unit = (argc > 1) ? argv[1] : "conv1d";
  const char *case_path = (argc > 2) ? argv[2]
                                    : auto_case_path(argc > 0 ? argv[0] : NULL, unit);

  if (!strcmp(unit, "conv1d")) return conv1d_unit(case_path);
  if (!strcmp(unit, "gru")) return gru_unit(case_path);

  fprintf(stderr, "unknown unit: %s\n", unit);
  return 2;
}

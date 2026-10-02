/*
 * 个人练习使用
 * 调用神经网络相关的 c 函数，做单元测试；输出与 PyTorch 实现逐项对比
 * filename: examples/rnn_unit.c
 *
 * 输出协议（一行一项，方便脚本比对）:
 *     [<层名>.<输出名>#<帧>] <v0> <v1> ...
 * 例:
 *     [conv1.out#0] -8.39260000e-02 8.15400000e-03 ...
 *
 * 注意第一段是**层名**而不是单元名：src/rnn.c 里 conv1/conv2 共用 compute_generic_conv1d、
 * gru1/gru2/gru3 共用 compute_generic_gru，所以一个单元会依次跑好几层，
 * 靠层名把同名的输出项区分开（py 侧 emit() 的写法完全一致）。
 *
 * 用例数据来源（和 py 侧完全一致）:
 *     ① argv[2] 显式指定的 JSON
 *     ② 按约定自动定位 <仓库根>/scripts/unit_cases/<用例文件名>.json
 * 严格模式：读不到 / 必需字段对不上，直接报错并返回非 0（不再退回内置保底数据）。
 *
 * 用法:
 *     ./rnn_unit conv1d                                # 自动找用例文件（= conv.json）
 *     ./rnn_unit conv1d scripts/unit_cases/conv.json   # 显式指定
 *     ./rnn_unit gru                                   # = gru.json
 *
 * 构建: make rnn_unit   （源码列表见 Makefile 的 rnn_unit_SOURCES）
 * 对比: uv run scripts/unit_check.py [unit...]
 *
 * 文件结构（本文件只留"主流程 + 各单元实现"，其余按类别拆出去）:
 *     rnn_unit.h       宏定义 + 用例结构体 + 跨文件共享声明
 *     rnn_unit_util.h  辅助函数声明
 *     rnn_unit_util.c  辅助函数实现（用例读取 / JSON 读取 / 输出协议 / 用例定位）
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rnnoise.h"
#include "rnnoise_data.h"
#include "common.h"
#include "denoise.h"
#include "nnet.h"

#include "rnn_unit.h"
#include "rnn_unit_util.h"

const int arch = 0;

/* 按 "<层名>.<输出名>#<帧>" 打一项。
 * 名字格式必须和 examples/rnn_unit.py 的 emit() 逐字节一致，脚本靠它配对。 */
static void print_layer_item(const char *layer, const char *suffix, int frame,
                             const float *v, int n) {
  char name[64];
  snprintf(name, sizeof(name), "%s.%s", layer, suffix);
  print_item(name, frame, v, n);
}

/* ===================== 单元: conv1d =====================
 * 对应 src/nnet.c: compute_generic_conv1d()
 * 它做的事: tmp = [mem, input] 拼起来 -> 过一层线性 -> 过激活 -> 把 tmp 的尾部存回 mem
 *
 * 同一份实现被 src/rnn.c 的 conv1 / conv2 两层复用，两层规模不同（见 rnn_unit.h 的宏），
 * 所以这里按层循环：每层各自读用例、各自跑、各自打 [<层名>.out#帧] / [<层名>.mem#帧]。
 *
 * 注意：conv2 在用例里**没有 inputs**（整网里它的每帧输入是 conv1 的输出），
 * 所以那一轮循环跑 0 帧、不产输出；conv2 的串联要等整网那条路（conv_gru_unit）覆盖。
 */

/* 一层的用例规模：层名 + 每帧输入宽 + 输出宽 */
typedef struct {
  const char *layer;
  int nb_in;
  int nb_out;
} ConvLayerSpec;

static const ConvLayerSpec CONV_LAYERS[] = {
  {"conv1", CASE_CONV1_IN, CASE_CONV1_OUT},
  {"conv2", CASE_CONV2_IN, CASE_CONV2_OUT},
};

static int conv1d_unit(const char *case_path) {
  static Conv1dCase sc;
  float out[CASE_CONV_MAX_OUT];
  float mem[CASE_CONV_MAX_IN*2];      /* 必须 >= nb_inputs - input_size = 2*nb_in */
  size_t li;
  int f;

  for (li = 0; li < sizeof(CONV_LAYERS)/sizeof(CONV_LAYERS[0]); li++) {
    const ConvLayerSpec *spec = &CONV_LAYERS[li];
    LinearLayer linear;

    if (load_conv1d_case(&sc, case_path, spec->layer, spec->nb_in, spec->nb_out) < 0)
      return 1;

    memset(&linear, 0, sizeof(linear));
    linear.bias = sc.bias;
    linear.float_weights = sc.float_weights;   /* 非 NULL 时 compute_linear 走 sgemv(float) */
    linear.weights = sc.int8_weights;          /* 用例里暂时没有，留零；float 路径优先 */
    linear.nb_inputs = 3 * sc.nb_in;
    linear.nb_outputs = sc.nb_out;

    memset(mem, 0, sizeof(mem));
    for (f = 0; f < sc.nb_frames; f++) {
      compute_generic_conv1d(&linear, out, mem, sc.inputs[f], sc.nb_in, ACTIVATION_TANH, arch);
      print_layer_item(spec->layer, "out", f, out, sc.nb_out);
      print_layer_item(spec->layer, "mem", f, mem, 2*sc.nb_in);
    }
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
 *
 * src/rnn.c 里 gru1 / gru2 / gru3 三层共用这一份实现，且三层真实规模相同
 * （见 rnn_unit.h 的宏），所以这里按层循环，用例规模是固定的一套。
 *
 * 注意：输入序列只有 gru1 那一段有（gru2 / gru3 在整网里由上一层的隐状态喂），
 * 所以这里三层共用 gru1 的输入跑；三层真正串起来要等整网那条路（conv_gru_unit）覆盖。
 */

#define MAX_RNN_NEURONS_ALL 1024

static void local_compute_generic_gru(const LinearLayer *input_weights,
                                      const LinearLayer *recurrent_weights,
                                      float *state, const float *in, int arch,
                                      const char *layer, int frame)
{
  int i;
  int N;
  float zrh[3*MAX_RNN_NEURONS_ALL];
  float recur[3*MAX_RNN_NEURONS_ALL];
  float *z;
  float *r;
  float *h;
  celt_assert(3*recurrent_weights->nb_inputs == recurrent_weights->nb_outputs);
  celt_assert(input_weights->nb_outputs == recurrent_weights->nb_outputs);
  N = recurrent_weights->nb_inputs;
  z = zrh;
  r = &zrh[N];
  h = &zrh[2*N];
  celt_assert(recurrent_weights->nb_outputs <= 3*MAX_RNN_NEURONS_ALL);
  celt_assert(in != state);
  compute_linear(input_weights, zrh, in, arch);
  compute_linear(recurrent_weights, recur, state, arch);
  for (i=0;i<2*N;i++)
     zrh[i] += recur[i];
  /* frame 走循环变量，不再硬编码 0（多帧时同一 key 会互相覆盖）。 */
  print_layer_item(layer, "zrh_recur", frame, zrh, 2*N);
  compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID, arch);
  print_layer_item(layer, "sigmoid", frame, zrh, 2*N);

  for (i=0;i<N;i++)
     h[i] += recur[2*N+i]*r[i];
  compute_activation(h, h, N, ACTIVATION_TANH, arch);
  print_layer_item(layer, "recur_tanh", frame, h, N);

  for (i=0;i<N;i++)
     h[i] = z[i]*state[i] + (1-z[i])*h[i];
  print_layer_item(layer, "h", frame, h, N);

  for (i=0;i<N;i++)
     state[i] = h[i];
}

static const char *const GRU_LAYERS[] = {"gru1", "gru2", "gru3"};

static int gru_unit(const char *case_path) {
  static GruCase gc;
  static float inputs[CASE_NB_FRAMES][CASE_GRU_IN];
  static float gru_state[CASE_GRU_N];
  int nb_frames, i;
  size_t li;

  /* 输入序列只存在 gru1 那一段里（gru2 / gru3 在整网里由上一层的隐状态喂）。
   * 这个单元测的是"同一份输入 × 三套权重"，所以先把 gru1 的输入单独读出来三层共用；
   * 必须和 py 侧 gru_demo() 取的 cases["gru1"]["inputs"] 是同一份。 */
  if (load_gru_case(&gc, case_path, GRU_LAYERS[0]) < 0) return 1;
  nb_frames = gc.nb_frames;
  memcpy(inputs, gc.inputs, sizeof(inputs));

  for (li = 0; li < sizeof(GRU_LAYERS)/sizeof(GRU_LAYERS[0]); li++) {
    const char *layer = GRU_LAYERS[li];
    LinearLayer gru_input;
    LinearLayer gru_recurrent;

    if (load_gru_case(&gc, case_path, layer) < 0) return 1;
    /* 输入序列统一用 gru1 那份，别的层再写一份就是数据和代码对不上，直接报错 */
    if (li > 0 && gc.nb_frames > 0) {
      fprintf(stderr, "[rnn_unit] json error: %s 的用例里不该有 inputs"
                      "（输入序列统一用 %s 那份）\n", layer, GRU_LAYERS[0]);
      return 1;
    }

    memset(&gru_input, 0, sizeof(gru_input));
    memset(&gru_recurrent, 0, sizeof(gru_recurrent));

    gru_input.bias = gc.input_bias;
    gru_input.float_weights = gc.input_weights;
    gru_input.nb_inputs = CASE_GRU_IN;
    gru_input.nb_outputs = CASE_GRU_OUT;

    gru_recurrent.bias = gc.recurrent_bias;
    gru_recurrent.float_weights = gc.recurrent_weights;  /* 非 NULL 时走 sgemv(float) */
    gru_recurrent.nb_inputs = CASE_GRU_N;
    gru_recurrent.nb_outputs = CASE_GRU_OUT;

    #ifdef USE_SU_BIAS
    gru_input.subias = gc.input_subias;
    gru_recurrent.subias = gc.recurrent_subias;
    #endif

    /* 初始隐状态显式置零：不用用例 json 里的 state（py 侧同样用 zeros），
     * 两侧从同一起点递推，且每次运行的起点都一样。 */
    memset(gru_state, 0, sizeof(gru_state));

    for (i = 0; i < nb_frames; ++i) {
      local_compute_generic_gru(&gru_input, &gru_recurrent, gru_state,
                                inputs[i], 0, layer, i);
      print_layer_item(layer, "state", i, gru_state, CASE_GRU_N);
    }
  }
  return 0;
}

/* ===================== 整网：把用例参数填进 RNNoise =====================
 *
 * 对标 src/rnnoise_data.h 的 RNNoise 结构 + src/rnnoise_data_little.c 末尾的 init_rnnoise()：
 * 那边用 WeightArray 表 + src/parse_lpcnet_weights.c 的 linear_init() 把真实权重挂到 10 个
 * LinearLayer 上；这里换成「用例 JSON」，所以不走 linear_init（它只认 WeightArray 表），
 * 改成逐层展开、手动把每个字段填好 —— 好处是每层的 nb_inputs / nb_outputs 一眼可见。
 *
 * 真实模型 -> 用例（一律 //16）:
 *     conv1           195 -> 128      =>  12 -> 8       每帧输入 4
 *     conv2           384 -> 384      =>  24 -> 24      每帧输入 8（= conv1 输出）
 *     gruN_input      384 -> 1152     =>  24 -> 72
 *     gruN_recurrent  384 -> 1152     =>  24 -> 72      nb_inputs = 隐状态 N
 *     dense_out       1536 -> 32      =>  96 -> 2       96 = conv2 输出 + 3 x gru 输出
 *     vad_dense       1536 -> 1       =>  96 -> 1
 *
 * 只用得上 bias / float_weights；subias / weights / weights_idx / diag / scale 都留 NULL，
 * 它们服务的是 int8 量化路径，用例 JSON 里没有。
 */

/* auto_case_path() 返回的是它内部的静态缓冲区，连着调会互相覆盖，所以先拷出来 */
static int copy_case_path(char *dst, size_t n, const char *unit) {
  const char *p = auto_case_path(NULL, unit);
  if (p == NULL) {
    fprintf(stderr, "[rnn_unit] error: 找不到 %s 的用例文件"
                    "（在仓库根目录跑，或看 scripts/unit_cases/）\n", unit);
    return 0;
  }
  snprintf(dst, n, "%s", p);
  return 1;
}

/* inputs 是 [帧][CASE_CONV1_IN] 的二维数组（conv1 的输入序列），
 * 所以形参类型要写成"指向数组的指针"，不能写成 float **。 */
static int padding_rn( RNNoise * rn, float (*inputs)[CASE_CONV1_IN]) {
  static Conv1dCase cc[2];        /* cc[0] = conv1  cc[1] = conv2 */
  static GruCase    gc[3];        /* gc[0] = gru1   gc[1] = gru2   gc[2] = gru3 */
  static DenseCase  dc[2];        /* dc[0] = dense_out  dc[1] = vad_dense */
  char conv_path[4096], gru_path[4096], dense_path[4096];

  /* 1. 三份用例文件：conv.json / gru.json / dense.json */
  if (!copy_case_path(conv_path, sizeof(conv_path), "conv1d")) return -1;
  if (!copy_case_path(gru_path, sizeof(gru_path), "gru")) return -1;
  if (!copy_case_path(dense_path, sizeof(dense_path), "dense")) return -1;
  fprintf(stderr, "[rnn_unit] padding_rn 用例: %s %s %s\n", conv_path, gru_path, dense_path);

  /* 2. 逐层读参数。这里只关心权重，用例里的 inputs 读不读都行
   *    （conv2 本来就没有 inputs；conv1 的 inputs 是给 conv1d 单元用的）。 */
  if (load_conv1d_case(&cc[0], conv_path, "conv1", CASE_CONV1_IN, CASE_CONV1_OUT) < 0) return -1;
  if (load_conv1d_case(&cc[1], conv_path, "conv2", CASE_CONV2_IN, CASE_CONV2_OUT) < 0) return -1;
  if (load_gru_case(&gc[0], gru_path, "gru1") < 0) return -1;
  if (load_gru_case(&gc[1], gru_path, "gru2") < 0) return -1;
  if (load_gru_case(&gc[2], gru_path, "gru3") < 0) return -1;
  if (load_dense_case(&dc[0], dense_path, "dense_out", CASE_DENSE_IN, CASE_DENSE_OUT) < 0) return -1;
  if (load_dense_case(&dc[1], dense_path, "vad_dense", CASE_DENSE_IN, CASE_VAD_OUT) < 0) return -1;

  /* 3. 逐层挂到 RNNoise 上（其它字段保持 NULL） */
  memset(rn, 0, sizeof(RNNoise));
  /* 逐行拷 conv1 的输入序列。
   * 不能整块 memcpy：Conv1dCase.inputs 的声明是 [CASE_NB_FRAMES][CASE_CONV_MAX_IN]
   * （行步长 = CASE_CONV2_IN = 8），而这里的目标是 [CASE_NB_FRAMES][CASE_CONV1_IN]
   * （行步长 4）。整块按"步长 4"拷的话，每行的填充区（那 4 个没被写过的 float）会被
   * 当成下一帧的数据，结果是输入序列错位、隔行变 0、且后 8 帧永远用不上。 */
  for (int f = 0; f < CASE_NB_FRAMES; ++f)
    memcpy(inputs[f], cc[0].inputs[f], CASE_CONV1_IN*sizeof(float));
  /* ---- conv1：每帧输入 4 -> 线性 12 -> 输出 8 ---- */
  rn->conv1.bias          = cc[0].bias;
  rn->conv1.float_weights = cc[0].float_weights;
  rn->conv1.nb_inputs     = 3 * CASE_CONV1_IN;
  rn->conv1.nb_outputs    = CASE_CONV1_OUT;

  /* ---- conv2：每帧输入 8（= conv1 输出）-> 线性 24 -> 输出 24 ---- */
  rn->conv2.bias          = cc[1].bias;
  rn->conv2.float_weights = cc[1].float_weights;
  rn->conv2.nb_inputs     = 3 * CASE_CONV2_IN;
  rn->conv2.nb_outputs    = CASE_CONV2_OUT;

  /* ---- gru1：输入投影 24 -> 72，递归投影 24(=N) -> 72 ---- */
  rn->gru1_input.bias          = gc[0].input_bias;
  rn->gru1_input.float_weights = gc[0].input_weights;
  rn->gru1_input.nb_inputs     = CASE_GRU_IN;
  rn->gru1_input.nb_outputs    = CASE_GRU_OUT;

  rn->gru1_recurrent.bias          = gc[0].recurrent_bias;
  rn->gru1_recurrent.float_weights = gc[0].recurrent_weights;
  rn->gru1_recurrent.nb_inputs     = CASE_GRU_N;
  rn->gru1_recurrent.nb_outputs    = CASE_GRU_OUT;

  /* ---- gru2 ---- */
  rn->gru2_input.bias          = gc[1].input_bias;
  rn->gru2_input.float_weights = gc[1].input_weights;
  rn->gru2_input.nb_inputs     = CASE_GRU_IN;
  rn->gru2_input.nb_outputs    = CASE_GRU_OUT;

  rn->gru2_recurrent.bias          = gc[1].recurrent_bias;
  rn->gru2_recurrent.float_weights = gc[1].recurrent_weights;
  rn->gru2_recurrent.nb_inputs     = CASE_GRU_N;
  rn->gru2_recurrent.nb_outputs    = CASE_GRU_OUT;

  /* ---- gru3 ---- */
  rn->gru3_input.bias          = gc[2].input_bias;
  rn->gru3_input.float_weights = gc[2].input_weights;
  rn->gru3_input.nb_inputs     = CASE_GRU_IN;
  rn->gru3_input.nb_outputs    = CASE_GRU_OUT;

  rn->gru3_recurrent.bias          = gc[2].recurrent_bias;
  rn->gru3_recurrent.float_weights = gc[2].recurrent_weights;
  rn->gru3_recurrent.nb_inputs     = CASE_GRU_N;
  rn->gru3_recurrent.nb_outputs    = CASE_GRU_OUT;

  /* ---- dense_out：输入 96（= conv2 输出 + 3 x gru 输出）-> 输出 2（gains）---- */
  rn->dense_out.bias          = dc[0].bias;
  rn->dense_out.float_weights = dc[0].float_weights;
  rn->dense_out.nb_inputs     = CASE_DENSE_IN;
  rn->dense_out.nb_outputs    = CASE_DENSE_OUT;

  /* ---- vad_dense：输入 96 -> 输出 1（VAD）---- */
  rn->vad_dense.bias          = dc[1].bias;
  rn->vad_dense.float_weights = dc[1].float_weights;
  rn->vad_dense.nb_inputs     = CASE_DENSE_IN;
  rn->vad_dense.nb_outputs    = CASE_VAD_OUT;

  return 0;
}

static int conv_gru_unit(const char *case_path) {
  // conv1->conv2->gru1->gru2->gru3
  (void)case_path;      /* 这个单元自己不收 case_path：三份用例由 padding_rn 按约定去找 */
  RNNoise m;
  float inputs[CASE_NB_FRAMES][CASE_CONV1_IN] = {{0.0}};
  if (padding_rn(&m, inputs) < 0) {
    /* 读不到参数就別往下跑：下面的 LinearLayer 全是 NULL，跑起来只会段错误 */
    fprintf(stderr, "[rnn_unit] error: all 单元初始化失败（用例参数没读全），见上面的错误信息\n");
    return 1;
  }
  RNNoise * model = &m;

  float conv1_state[CASE_CONV1_IN*3] = {0.0};
  float conv2_state[CASE_CONV1_OUT*3] = {0.0};
  /* 隐状态长度 = CASE_GRU_N(24)，不是真实模型的 GRU_STATE_SIZE(384) ——
   * compute_generic_gru() 只会写 N = recurrent->nb_inputs 个元素。 */
  float gru1_state[CASE_GRU_N] = {0.0};
  float gru2_state[CASE_GRU_N] = {0.0};
  float gru3_state[CASE_GRU_N] = {0.0};
  float gains[(CASE_GRU_IN*4)/(1536/32)] = {0.0};
  float vad[1] = {0.0};

  float tmp[MAX_RNN_NEURONS_ALL]={0.0};
  /* 拼接缓冲：conv2 输出(CASE_CONV2_OUT) + gru1/2/3 各一份隐状态(CASE_GRU_N) = CASE_DENSE_IN */
  float cat[CASE_DENSE_IN] = {0.0};

  for (int i = 0; i < CASE_NB_FRAMES; ++i) {
    float *input = inputs[i];

    /*for (int i=0;i<INPUT_SIZE;i++) printf("%f ", input[i]);printf("\n");*/
    compute_generic_conv1d(&model->conv1, tmp, conv1_state, input, CASE_CONV1_IN, ACTIVATION_TANH, arch);
    print_layer_item("conv1","out",i,tmp,CASE_CONV1_OUT);
    compute_generic_conv1d(&model->conv2, cat, conv2_state, tmp, CASE_CONV2_IN, ACTIVATION_TANH, arch);
    print_layer_item("conv2","out",i,cat,CASE_CONV2_OUT);
    compute_generic_gru(&model->gru1_input, &model->gru1_recurrent, gru1_state, cat, arch);
    print_layer_item("gru1","state",i,gru1_state,CASE_GRU_N);
    compute_generic_gru(&model->gru2_input, &model->gru2_recurrent, gru2_state, gru1_state, arch);
    print_layer_item("gru2","state",i,gru2_state,CASE_GRU_N);

    compute_generic_gru(&model->gru3_input, &model->gru3_recurrent, gru3_state, gru2_state, arch);
    print_layer_item("gru3","state",i,gru3_state,CASE_GRU_N);

    /* 三份隐状态各有 CASE_GRU_N 个数（= 24），不是 CASE_GRU_OUT(72)，更不是真实的 GRU1_OUT_SIZE(384) */
    RNN_COPY(&cat[CASE_CONV2_OUT], gru1_state, CASE_GRU_N);
    RNN_COPY(&cat[CASE_CONV2_OUT+CASE_GRU_N], gru2_state, CASE_GRU_N);
    RNN_COPY(&cat[CASE_CONV2_OUT+CASE_GRU_N+CASE_GRU_N], gru3_state, CASE_GRU_N);

    print_layer_item("dense","in",i,cat,CASE_CONV2_OUT+CASE_GRU_N*3);

    compute_generic_dense(&model->dense_out, gains, cat, ACTIVATION_SIGMOID, arch);
    print_layer_item("dense","gains", i, gains, sizeof(gains)/sizeof(float));
    compute_generic_dense(&model->vad_dense, vad, cat, ACTIVATION_SIGMOID, arch);
    print_layer_item("vad","out", i, vad, sizeof(vad)/sizeof(float));
    
  }

  return 0;
}

/* ===================== 单元: activation =====================
 * 单算子微基准（micro-benchmark）：在一段固定网格上把 C 的 sigmoid / tanh 打出来。
 *
 * 为什么要单独做：整网里每一层的差都是「激活近似 + 线性累加 + 上游传播」揉在一起的，
 * 没法回答"这个 1e-4 到底是谁贡献的"。把算子单独拎出来在一段网格上量一遍，
 * 就得到一把**独立的刻度尺**（不受网络、不受递推污染）。
 *
 * 网格点也一并打印（act.x），Python 侧直接读它来算参考值 ——
 * 这样两侧用的是**逐位相同**的输入，避免各自造网格造出假差异。
 *
 * 对应统计脚本：scripts/err_stats/bench_act.py
 * 这个单元不吃用例文件。
 */
#define BENCH_N 1601
static int activation_unit(const char *case_path) {
  static float x[BENCH_N];
  static float y[BENCH_N];
  int i;

  (void)case_path;
  for (i = 0; i < BENCH_N; i++) x[i] = -8.0f + 0.01f * (float)i;

  print_item("act.x", 0, x, BENCH_N);

  compute_activation(y, x, BENCH_N, ACTIVATION_SIGMOID, 0);
  print_item("act.sigmoid", 0, y, BENCH_N);

  compute_activation(y, x, BENCH_N, ACTIVATION_TANH, 0);
  print_item("act.tanh", 0, y, BENCH_N);

  return 0;
}

/* 
 *  * 每个单元: 一个 load_xxx_case() + 一个 xxx_unit(case_path)，并在 main 里注册。
 */

static int is_high_accuracy() {
  float x = 0.371234;
  float exact = 1.0f / (1.0f + exp(-x));
  float rn = 0.0;
  compute_activation(&rn, &x, 1, ACTIVATION_SIGMOID, 0);
  
  printf("x=%+.10f rn=%+.10f exact=%+.10f diff=%+.10e\n",
       x, rn, exact, rn-exact);
  return !!(rn-exact == 0.0);
}

/* ===================== 用法 =====================
 * unit 不认识、用例文件找不到、或者跑失败时都会打这一份，省得再去翻源码。
 * bad_unit 非 NULL 表示是"单元名不认识"这一类错误。
 */
static void usage(const char *argv0, const char *bad_unit) {
  fprintf(stderr,
    "\n用法: %s <unit> [case.json]\n"
    "\n"
    "  unit        要跑的单元（缺省 conv1d）:\n"
    "                conv1d  conv1 / conv2 两层各自独立跑，打 [convN.out#帧] / [convN.mem#帧]\n"
    "                gru     gru1 / gru2 / gru3 三层，打 [gruN.zrh_recur|sigmoid|recur_tanh|h|state#帧]\n"
    "                all     整条链路 conv1 -> conv2 -> gru1 -> gru2 -> gru3\n"
    "                        参数分别取自 scripts/unit_cases/ 下的\n"
    "                        conv.json / gru.json / dense.json，所以它不收 case.json\n"
    "                activation  单算子微基准：在 x = -8..+8（步长 0.01）的固定网格上\n"
    "                        打 act.x / act.sigmoid / act.tanh，给误差归因当刻度尺\n"
    "                        也不收 case.json（对应 scripts/err_stats/bench_act.py）\n"
    "  case.json   可选，显式指定用例文件；不写就按约定找\n"
    "              <仓库根>/scripts/unit_cases/<用例文件名>.json\n"
    "              （conv1d -> conv.json、gru -> gru.json）\n"
    "\n"
    "  -h, --help  只看这份说明\n"
    "\n"
    "例:\n"
    "  %s conv1d\n"
    "  %s gru scripts/unit_cases/gru.json\n"
    "  %s all\n"
    "\n"
    "和 PyTorch 逐项对比: uv run scripts/unit_check.py [unit...]\n"
    "构建:               make rnn_unit\n",
    argv0, argv0, argv0, argv0);
  if (bad_unit != NULL)
    fprintf(stderr, "\n[rnn_unit] 不认识的 unit: %s\n", bad_unit);
}

int main(int argc, char **argv) {
  const char *argv0 = (argc > 0 && argv[0] != NULL) ? argv[0] : "rnn_unit";
  const char *unit = (argc > 1) ? argv[1] : "conv1d";
  const char *case_path;
  int rc;

  if (!strcmp(unit, "-h") || !strcmp(unit, "--help") || !strcmp(unit, "help")) {
    usage(argv0, NULL);
    return 0;
  }

  case_path = (argc > 2) ? argv[2]
                         : auto_case_path(argc > 0 ? argv[0] : NULL, unit);

  printf("[Notice] Now %s accuracy used !!!!!!!\n", is_high_accuracy() == 1? "high":"low");
  /* 打一行用例来源，驱动器拿它核对两侧读的是不是同一份文件。
   * all 单元本来就对应不到单个 json，所以这里单独说明一下。 */
  fprintf(stderr, "[rnn_unit] case file: %s\n",
          case_path != NULL ? case_path
                            : (strcmp(unit, "all") == 0 ? "(all 用三份用例，见下)"
                               : (strcmp(unit, "activation") == 0 ? "(activation 不用用例)"
                                                                  : "(未找到)")));

  if (!strcmp(unit, "conv1d"))   rc = conv1d_unit(case_path);
  else if (!strcmp(unit, "gru")) rc = gru_unit(case_path);
  else if (!strcmp(unit, "all")) rc = conv_gru_unit(case_path);
  else if (!strcmp(unit, "activation")) rc = activation_unit(case_path);
  else {
    usage(argv0, unit);
    return 2;
  }

  /* 只在"启动就错"的那一类（连用例文件都没找到）把完整用法打出来；
   * 数据类错误保持原样 —— 最后一行留真正的错误原因，别被用法盖掉。 */
  if (rc != 0 && case_path == NULL
      && strcmp(unit, "all") != 0 && strcmp(unit, "activation") != 0) {
    fprintf(stderr, "\n[rnn_unit] unit %s 启动失败：没找到用例文件，argv[2] 也没显式给。用法如下:\n",
            unit);
    usage(argv0, NULL);
  }
  return rc;
}

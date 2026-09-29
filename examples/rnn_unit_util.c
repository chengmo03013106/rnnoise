/*
 * rnn_unit 辅助函数实现：用例读取、极简 JSON 读取、输出协议、用例文件定位
 * filename: examples/rnn_unit_util.c
 *
 * 接口与说明见 examples/rnn_unit_util.h；用例结构体见 examples/rnn_unit.h。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rnn_unit.h"
#include "rnn_unit_util.h"

/* ===================== 极简 JSON 读取器 ===================== */

static void uj_skip(UjDoc *d) {
  while (*d->p == ' ' || *d->p == '\t' || *d->p == '\n' || *d->p == '\r') d->p++;
}

/* 给一个"出现过"的键占位（count 从 0 开始，数字会继续挂到同一条上）。
 * 为什么要占位：键存在但值不是我们要的形状时（字符串、空数组、一维数字数组……），
 * 必须能查出"这个键写过"，而不是当成"没写这个键" —— py 侧 json.load 之后是能看见键的，
 * 不占位两边就不一致。返回条目下标，容量不够返回 -1。 */
static int uj_add_entry(UjDoc *d, const char *path) {
  int i;
  for (i = 0; i < d->nb_entries; i++)
    if (!strcmp(d->entries[i].path, path)) return i;
  if (d->nb_entries >= UJ_MAX_ENTRIES) {
    /* 容量不够时 uj_load 只会返回 -1，报错信息里看不出原因，这里补一句 */
    fprintf(stderr, "[rnn_unit] json error: 条目数超过上限 %d（键 %s），"
                    "请调大 rnn_unit_util.h 的 UJ_MAX_ENTRIES\n", UJ_MAX_ENTRIES, path);
    d->ok = 0;
    return -1;
  }
  snprintf(d->entries[i].path, UJ_PATH_MAX, "%s", path);
  d->entries[i].first = d->nb_pool;
  d->entries[i].count = 0;
  d->nb_entries++;
  return i;
}

static void uj_add_number(UjDoc *d, const char *path, double v) {
  int i = uj_add_entry(d, path);
  if (i < 0) return;
  if (d->nb_pool >= UJ_MAX_NUMBERS) {
    fprintf(stderr, "[rnn_unit] json error: 数字个数超过上限 %d，"
                    "请调大 rnn_unit_util.h 的 UJ_MAX_NUMBERS\n", UJ_MAX_NUMBERS);
    d->ok = 0;
    return;
  }
  d->pool[d->nb_pool++] = v;
  d->entries[i].count++;
}

static void uj_parse_value(UjDoc *d, const char *path);

static void uj_parse_object(UjDoc *d, const char *path) {
  d->p++;                                   /* 跳过 '{' */
  for (;;) {
    char key[UJ_KEY_MAX], sub[UJ_PATH_MAX];
    int k = 0;
    uj_skip(d);
    if (*d->p == '}') { d->p++; return; }
    if (*d->p != '"') { d->ok = 0; return; }
    d->p++;
    while (*d->p && *d->p != '"') { if (k < UJ_KEY_MAX-1) key[k++] = *d->p; d->p++; }
    key[k] = 0;
    if (*d->p == '"') d->p++; else { d->ok = 0; return; }
    uj_skip(d);
    if (*d->p != ':') { d->ok = 0; return; }
    d->p++;
    if (path[0] == 0) snprintf(sub, UJ_PATH_MAX, "%s", key);
    else              snprintf(sub, UJ_PATH_MAX, "%s.%s", path, key);
    uj_add_entry(d, sub);        /* 先占位：这个键"出现过" */
    if (!d->ok) return;
    uj_parse_value(d, sub);
    if (!d->ok) return;
    uj_skip(d);
    if (*d->p == ',') { d->p++; continue; }
    if (*d->p == '}') { d->p++; return; }
    d->ok = 0; return;
  }
}

static void uj_parse_array(UjDoc *d, const char *path) {
  int idx = 0, nested;
  d->p++;                                   /* 跳过 '[' */
  uj_skip(d);
  if (*d->p == ']') { d->p++; return; }
  nested = (*d->p == '[' || *d->p == '{');
  for (;;) {
    if (nested) {
      char sub[UJ_PATH_MAX];
      snprintf(sub, UJ_PATH_MAX, "%s[%d]", path, idx);
      uj_parse_value(d, sub);
    } else {
      uj_parse_value(d, path);
    }
    if (!d->ok) return;
    idx++;
    uj_skip(d);
    if (*d->p == ',') { d->p++; continue; }
    if (*d->p == ']') { d->p++; return; }
    d->ok = 0; return;
  }
}

static void uj_parse_value(UjDoc *d, const char *path) {
  uj_skip(d);
  if (*d->p == '{') { uj_parse_object(d, path); return; }
  if (*d->p == '[') { uj_parse_array(d, path); return; }
  if (*d->p == '"') {                       /* 字符串值: 跳过不记 */
    d->p++;
    while (*d->p && *d->p != '"') d->p++;
    if (*d->p == '"') d->p++; else d->ok = 0;
    return;
  }
  {
    char *end;
    double v = strtod(d->p, &end);
    if (end == d->p) { d->ok = 0; return; }
    d->p = end;
    uj_add_number(d, path, v);
  }
}

/* 返回: 1 = 读到且解析成功, 0 = 文件打不开, -1 = 内容解析失败 */
int uj_load(UjDoc *d, const char *filename) {
  FILE *f;
  long len;
  char *buf;
  memset(d, 0, sizeof(*d));
  d->ok = 1;
  f = fopen(filename, "rb");
  if (!f) return 0;
  fseek(f, 0, SEEK_END);
  len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len <= 0) { fclose(f); return -1; }
  buf = (char *)malloc((size_t)len + 1);
  if (!buf) { fclose(f); return 0; }
  if (fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return 0; }
  fclose(f);
  buf[len] = 0;
  d->p = buf;
  uj_parse_value(d, "");
  free(buf);
  return d->ok ? 1 : -1;
}

const double *uj_get(const UjDoc *d, const char *path, int *count) {
  int i;
  for (i = 0; i < d->nb_entries; i++)
    if (!strcmp(d->entries[i].path, path)) {
      *count = d->entries[i].count;
      return &d->pool[d->entries[i].first];
    }
  *count = 0;
  return NULL;
}

/* ===================== 输出协议 ===================== */

void print_item(const char *name, int frame, const float *v, int n) {
  int i;
  printf("[%s#%d]", name, frame);
  for (i = 0; i < n; i++) printf(" %.8e", v[i]);
  printf("\n");
}

/* ===================== 用例文件自动定位 =====================
 *
 * 单元名和用例文件名不再一一对应：一个文件里可能放着同一算子的多层参数，
 * 比如 conv1d 单元用的是 scripts/unit_cases/conv.json（里面有 conv1 / conv2 两层）。
 */

static const char *case_basename(const char *unit) {
  if (strcmp(unit, "conv1d") == 0) return "conv";
  return unit;
}

const char *auto_case_path(const char *argv0, const char *unit) {
  static char found[4096];
  char cand[4096];
  char exe[4096];
  const char *base = case_basename(unit);
  size_t n;
  int i;

  for (i = 0; i < 2; i++) {
    FILE *f;
    if (i == 0) {
      if (argv0 == NULL || realpath(argv0, exe) == NULL) continue;
      n = strlen(exe);
      while (n > 1 && exe[n-1] != '/') n--;      /* 砍掉文件名，留下目录（含结尾 '/'） */
      exe[n] = 0;
      snprintf(cand, sizeof(cand), "%sscripts/unit_cases/%s.json", exe, base);
    } else {
      snprintf(cand, sizeof(cand), "scripts/unit_cases/%s.json", base);
    }
    f = fopen(cand, "rb");
    if (f != NULL) {
      fclose(f);
      snprintf(found, sizeof(found), "%s", cand);
      return found;
    }
  }
  return NULL;
}

/* ===================== 用例读取（严格模式）=====================
 *
 * 一次调用 = 读一层的参数。层名用来拼 JSON 路径（"conv1.bias"、"gru2.inputs[3]"）。
 * 任何必需字段缺失或长度不对，都打错误信息 + 返回 -1，调用方直接退出程序；
 * 不再有"退回内置保底数据"这条路。
 * 字段名和校验规则必须和 examples/rnn_unit.py 的 _load_*_case() 完全一致。
 */

/* 打开用例文件。打不开或解析失败都算错误。 */
static int open_case(UjDoc *d, const char *path) {
  int st;
  if (path == NULL) {
    fprintf(stderr, "[rnn_unit] error: 没有用例文件（scripts/unit_cases/<unit>.json 不存在，"
                    "也没在命令行上指定）\n");
    return 0;
  }
  st = uj_load(d, path);
  if (st != 1) {
    fprintf(stderr, "[rnn_unit] json error: 读取 %s 失败（%s）\n", path,
            st == 0 ? "文件打不开" : "内容解析失败");
    return 0;
  }
  return 1;
}

/* 取一串数字，长度必须严格等于 n；不合格就报错（唯一的长度校验点）。
 * 返回 NULL 表示不合格，调用方直接返回 -1。 */
static const double *need_numbers(const UjDoc *d, const char *key, int n) {
  int got = 0;
  const double *v = uj_get(d, key, &got);
  if (v == NULL || got != n) {
    fprintf(stderr, "[rnn_unit] json error: 字段 %s 缺失或长度不对（需要 %d，实际 %d）\n",
            key, n, got);
    return NULL;
  }
  return v;
}

/* 读一串 float，长度必须严格等于 n。 */
static int need_floats(const UjDoc *d, const char *key, float *dst, int n) {
  const double *v = need_numbers(d, key, n);
  int i;
  if (v == NULL) return 0;
  for (i = 0; i < n; i++) dst[i] = (float)v[i];
  return 1;
}

/* 读 "<layer>.<field>" 这个键下的 float 串。 */
static int need_layer_floats(const UjDoc *d, const char *layer, const char *field,
                             float *dst, int n) {
  char key[UJ_PATH_MAX];
  snprintf(key, sizeof(key), "%s.%s", layer, field);
  return need_floats(d, key, dst, n);
}

/* 读 "<layer>.inputs"：逐帧读 "<layer>.inputs[i]"，每帧必须正好 width 个数。
 * 帧数必须落在 1..max_frames，超上限或某帧元素数不对都算错误（不再"整段不采纳"）。
 * dst 按行存，行距 stride（允许宽数组按窄宽度存）。 */
static int need_frames(const UjDoc *d, const char *layer, float *dst,
                       int stride, int width, int max_frames, int *nb_frames) {
  char key[UJ_PATH_MAX];
  int i, j;

  for (i = 0; i < max_frames; i++) {
    int got = 0;
    const double *v;
    snprintf(key, sizeof(key), "%s.inputs[%d]", layer, i);
    v = uj_get(d, key, &got);
    if (v == NULL) break;                      /* 帧数组到头，正常结束 */
    if (got != width) {
      fprintf(stderr, "[rnn_unit] json error: 字段 %s 每帧应有 %d 个数，实际 %d\n",
              key, width, got);
      return 0;
    }
    for (j = 0; j < width; j++) dst[i*stride + j] = (float)v[j];
  }
  if (i == 0) {
    fprintf(stderr, "[rnn_unit] json error: 字段 %s.inputs 为空\n", layer);
    return 0;
  }
  snprintf(key, sizeof(key), "%s.inputs[%d]", layer, max_frames);
  {
    int got = 0;
    if (uj_get(d, key, &got) != NULL) {
      fprintf(stderr, "[rnn_unit] json error: 字段 %s.inputs 帧数超过上限 %d\n",
              layer, max_frames);
      return 0;
    }
  }
  *nb_frames = i;
  return 1;
}

/* 读可选的 "<layer>.inputs"（多帧输入）。三种情况：
 *   - 有 "<layer>.inputs[i]" → 多帧，交给 need_frames 严格校验
 *   - 只有 "<layer>.inputs" → 说明它是标量 / 字符串 / 空数组 / 一维数字数组，报错
 *   - 两个都没有            → 这一层没有可独立跑的帧（conv2 / gru2 / gru3 就是这样），
 *                             *nb_frames = 0，不算错误
 * "出现就必须合法"这条规则必须和 py 侧 _load_*_case() 的 `"inputs" in sub` 一致。 */
static int need_optional_frames(const UjDoc *d, const char *layer, float *dst,
                                int stride, int width, int *nb_frames) {
  char key[UJ_PATH_MAX];
  int got = 0;

  *nb_frames = 0;
  snprintf(key, sizeof(key), "%s.inputs[0]", layer);
  if (uj_get(d, key, &got) != NULL)
    return need_frames(d, layer, dst, stride, width, CASE_NB_FRAMES, nb_frames);

  snprintf(key, sizeof(key), "%s.inputs", layer);
  if (uj_get(d, key, &got) != NULL) {
    fprintf(stderr, "[rnn_unit] json error: 字段 %s 不是多帧数组"
                    "（应该是 [[...], [...]]，或者干脆不写这一项）\n", key);
    return 0;
  }
  return 1;
}

int load_conv1d_case(Conv1dCase *c, const char *path, const char *layer, int nb_in, int nb_out) {
  static UjDoc doc;
  int n_w = 3 * nb_in * nb_out;

  c->layer = layer;
  c->nb_in = nb_in;
  c->nb_out = nb_out;

  if (!open_case(&doc, path)) return -1;
  if (!need_layer_floats(&doc, layer, "bias", c->bias, nb_out)) return -1;
  if (!need_layer_floats(&doc, layer, "float_weights", c->float_weights, n_w)) return -1;
  /* conv2 在整网里由 conv1 的输出喂，用例里没有 inputs */
  if (!need_optional_frames(&doc, layer, &c->inputs[0][0], CASE_CONV_MAX_IN, nb_in,
                            &c->nb_frames)) return -1;
  return 0;
}

int load_gru_case(GruCase *c, const char *path, const char *layer) {
  static UjDoc doc;

  c->layer = layer;

  if (!open_case(&doc, path)) return -1;
  if (!need_layer_floats(&doc, layer, "input_weights_float", c->input_weights,
                         CASE_GRU_IN*CASE_GRU_OUT)) return -1;
  if (!need_layer_floats(&doc, layer, "input_bias", c->input_bias, CASE_GRU_OUT)) return -1;
  if (!need_layer_floats(&doc, layer, "recurrent_weights_float", c->recurrent_weights,
                         CASE_GRU_N*CASE_GRU_OUT)) return -1;
  if (!need_layer_floats(&doc, layer, "recurrent_bias", c->recurrent_bias, CASE_GRU_OUT)) return -1;
  if (!need_layer_floats(&doc, layer, "state", c->state, CASE_GRU_N)) return -1;
  /* gru2 / gru3 在整网里由上一层的隐状态喂，用例里没有 inputs */
  if (!need_optional_frames(&doc, layer, &c->inputs[0][0], CASE_GRU_IN, CASE_GRU_IN,
                            &c->nb_frames)) return -1;
  /* input_subias / recurrent_subias 不读：用例 JSON 里没有（只在 int8 + USE_SU_BIAS 那条路上用）*/
  return 0;
}

int load_dense_case(DenseCase *c, const char *path, const char *layer, int nb_in, int nb_out) {
  static UjDoc doc;

  c->layer = layer;
  c->nb_in = nb_in;
  c->nb_out = nb_out;

  if (!open_case(&doc, path)) return -1;
  if (!need_layer_floats(&doc, layer, "bias", c->bias, nb_out)) return -1;
  if (!need_layer_floats(&doc, layer, "float_weights", c->float_weights, nb_in*nb_out)) return -1;
  return 0;
}

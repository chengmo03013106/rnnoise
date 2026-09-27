/*
 * rnn_unit 辅助函数实现：极简 JSON 读取、输出协议、用例文件定位
 * filename: examples/rnn_unit_util.c
 *
 * 接口与说明见 examples/rnn_unit_util.h。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rnn_unit_util.h"

/* ===================== 极简 JSON 读取器 ===================== */

static void uj_skip(UjDoc *d) {
  while (*d->p == ' ' || *d->p == '\t' || *d->p == '\n' || *d->p == '\r') d->p++;
}

static void uj_add_number(UjDoc *d, const char *path, double v) {
  int i;
  for (i = 0; i < d->nb_entries; i++)
    if (!strcmp(d->entries[i].path, path)) break;
  if (i == d->nb_entries) {
    if (d->nb_entries >= UJ_MAX_ENTRIES) { d->ok = 0; return; }
    snprintf(d->entries[i].path, UJ_PATH_MAX, "%s", path);
    d->entries[i].first = d->nb_pool;
    d->entries[i].count = 0;
    d->nb_entries++;
  }
  if (d->nb_pool >= UJ_MAX_NUMBERS) { d->ok = 0; return; }
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

int uj_get_floats(const UjDoc *d, const char *path, float *dst, int n) {
  int got = 0, i;
  const double *v = uj_get(d, path, &got);
  if (!v || got != n) return 0;
  for (i = 0; i < n; i++) dst[i] = (float)v[i];
  return 1;
}

void uj_get_int8(const UjDoc *d, const char *path, int8_t *dst, int n) {
  int got = 0, i;
  const double *v = uj_get(d, path, &got);
  if (!v || got != n) return;
  for (i = 0; i < n; i++) dst[i] = (int8_t)v[i];
}

/* ===================== 输出协议 ===================== */

void print_item(const char *name, int frame, const float *v, int n) {
  int i;
  printf("[%s#%d]", name, frame);
  for (i = 0; i < n; i++) printf(" %.8e", v[i]);
  printf("\n");
}

/* ===================== 用例文件自动定位 ===================== */

const char *auto_case_path(const char *argv0, const char *unit) {
  static char found[4096];
  char cand[4096];
  char exe[4096];
  size_t n;
  int i;

  for (i = 0; i < 2; i++) {
    FILE *f;
    if (i == 0) {
      if (argv0 == NULL || realpath(argv0, exe) == NULL) continue;
      n = strlen(exe);
      while (n > 1 && exe[n-1] != '/') n--;      /* 砍掉文件名，留下目录（含结尾 '/'） */
      exe[n] = 0;
      snprintf(cand, sizeof(cand), "%sscripts/unit_cases/%s.json", exe, unit);
    } else {
      snprintf(cand, sizeof(cand), "scripts/unit_cases/%s.json", unit);
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

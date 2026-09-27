/*
 * rnn_unit 辅助函数：用例文件定位、极简 JSON 读取、输出协议
 * filename: examples/rnn_unit_util.h
 *
 * 实现见 examples/rnn_unit_util.c。
 */
#ifndef RNN_UNIT_UTIL_H
#define RNN_UNIT_UTIL_H

#include <stdint.h>

/* ===================== 极简 JSON 读取器 =====================
 *
 * 只服务本项目的用例文件，支持子集:
 *     对象 / 数组 / 数字 / 字符串（字符串仅当 key 和说明用，值直接跳过）
 * 产出是一张扁平表，路径按 JSON 层级拼接:
 *     "bias"        -> 8 个数
 *     "inputs[0]"   -> 4 个数
 * 规则: 一个数组的元素如果全是数字，就不给它加下标；
 *       元素本身还是数组/对象时，才加 [i]
 * 不支持: 转义字符、null/true/false、注释
 */
#define UJ_MAX_ENTRIES 64
#define UJ_MAX_NUMBERS 16384
#define UJ_PATH_MAX    96
#define UJ_KEY_MAX     64

typedef struct {
  char path[UJ_PATH_MAX];
  int  first;
  int  count;
} UjEntry;

typedef struct {
  UjEntry entries[UJ_MAX_ENTRIES];
  int     nb_entries;
  double  pool[UJ_MAX_NUMBERS];
  int     nb_pool;
  const char *p;
  int     ok;
} UjDoc;

/* 返回: 1 = 读到且解析成功, 0 = 文件打不开, -1 = 内容解析失败 */
int uj_load(UjDoc *d, const char *filename);
const double *uj_get(const UjDoc *d, const char *path, int *count);
int uj_get_floats(const UjDoc *d, const char *path, float *dst, int n);
void uj_get_int8(const UjDoc *d, const char *path, int8_t *dst, int n);

/* ===================== 输出协议 ===================== */

void print_item(const char *name, int frame, const float *v, int n);

/* ===================== 用例文件自动定位 =====================
 * 按约定找 <仓库根>/scripts/unit_cases/<unit>.json，省得每次都手打路径。
 * 依次尝试:
 *     ① 由 argv[0] 的真实路径推出可执行文件所在目录（可执行文件就在仓库根）
 *     ② 当前工作目录下的 scripts/unit_cases/
 * 找到返回静态缓冲区里的路径，找不到返回 NULL（调用方退回保底数据）。
 */
const char *auto_case_path(const char *argv0, const char *unit);

#endif /* RNN_UNIT_UTIL_H */

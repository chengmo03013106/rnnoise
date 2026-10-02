#!/bin/sh
# 采集整网 `all` 单元的原始输出：{C, Py} × {高精度, 低精度}
#
# 用法（仓库根目录）：
#     sh scripts/err_stats/collect_all.sh
#
# 落盘到 scripts/err_stats/raw_all/：
#     <impl>.all              完整 stdout（含精度 Notice 与 is_high_accuracy 探针）
#     <impl>.err              stderr
#     <impl>.txt              只有 `[<层名>.<输出名>#<帧>]` 协议行，给统计脚本吃
#     build_high.log / build_low.log    编译日志
#
# impl ∈ {c_high, c_low, py_high, py_low}
#
# C 侧切精度有两个坑，都靠这个脚本兜住：
#   1. automake 不会因为 CFLAGS 变化就重编 .o → 每次切档前先 `touch src/nnet_default.c`
#   2. 光 `touch` 还不够保险，必须**断言运行时那行** `[Notice] Now high|low accuracy used`，
#      否则可能采到"假同精度"的数据（编译没生效，两侧其实同档）
#
# 注意：脚本跑完后留在磁盘上的 C 二进制是**低精度**那份（最后一次编译）。
set -e

cd "$(dirname "$0")/../.."
OUT=scripts/err_stats/raw_all
mkdir -p "$OUT"

echo "=== C 高精度 (-DHIGH_ACCURACY) ==="
touch src/nnet_default.c
make rnn_unit RNN_UNIT_HIGH_ACCURACY=-DHIGH_ACCURACY > "$OUT/build_high.log" 2>&1
./rnn_unit all > "$OUT/c_high.all" 2> "$OUT/c_high.err"
grep -q 'Now high accuracy used' "$OUT/c_high.all" || {
  echo "错误：c_high 那一份不是高精度（src/nnet_default.c 没重编？）" >&2
  exit 1
}
# 单算子微基准（同一份编译产物，高精度档）：给误差归因当刻度尺
./rnn_unit activation > "$OUT/c_act_high.all" 2> "$OUT/c_act_high.err"
# `gru` 单元（三层 gru1/gru2/gru3 共用同一实现，各打 5 项）：
# zrh_recur / sigmoid / recur_tanh / h / state。
# 要它的原因是**整网 `all` 链路里 gru 只打 state**，拿不到 sigmoid 的前置激活，
# 而 `bench_act.py` 需要"前置激活实际落在什么区间"来和全网格对比。
./rnn_unit gru > "$OUT/c_gru_high.all" 2> "$OUT/c_gru_high.err"

echo "=== C 低精度（默认，不定义 HIGH_ACCURACY）==="
touch src/nnet_default.c
make rnn_unit > "$OUT/build_low.log" 2>&1
./rnn_unit all > "$OUT/c_low.all" 2> "$OUT/c_low.err"
grep -q 'Now low accuracy used' "$OUT/c_low.all" || {
  echo "错误：c_low 那一份不是低精度" >&2
  exit 1
}
./rnn_unit activation > "$OUT/c_act_low.all" 2> "$OUT/c_act_low.err"
./rnn_unit gru > "$OUT/c_gru_low.all" 2> "$OUT/c_gru_low.err"

echo "=== Python 高精度 (--acc high) ==="
uv run examples/rnn_unit.py all --acc high > "$OUT/py_high.all" 2> "$OUT/py_high.err"

echo "=== Python 低精度 (--acc low) ==="
uv run examples/rnn_unit.py all --acc low > "$OUT/py_low.all" 2> "$OUT/py_low.err"

# 抽协议行：只保留 `[<层名>.<输出名>#<帧>] ...` 那种，
# 挡掉 `[Notice] ...` 和 rnnoise_demo 里那句 `print('conv1 ...')` 的多行 tensor
for impl in c_high c_low py_high py_low c_act_high c_act_low c_gru_high c_gru_low; do
  grep -E '^\[[^]]*#' "$OUT/$impl.all" > "$OUT/$impl.txt"
done

echo
echo "=== 整网协议行数（应为 16 帧 × 8 项 = 128）==="
wc -l "$OUT"/c_high.txt "$OUT"/c_low.txt "$OUT"/py_high.txt "$OUT"/py_low.txt
echo
echo "=== 单算子网格行数（应为 3 行：act.x / act.sigmoid / act.tanh）==="
wc -l "$OUT"/c_act_high.txt "$OUT"/c_act_low.txt
echo
echo "=== gru 单元协议行数（应为 3 层 × 16 帧 × 5 项 = 240）==="
wc -l "$OUT"/c_gru_high.txt "$OUT"/c_gru_low.txt
echo
echo "=== C 侧精度自检（必须与文件名一致）==="
grep -h 'accuracy' "$OUT"/c_high.all "$OUT"/c_low.all "$OUT"/c_act_high.all \
    "$OUT"/c_act_low.all "$OUT"/c_gru_high.all "$OUT"/c_gru_low.all

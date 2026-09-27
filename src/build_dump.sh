cd /Users/chengmo/Work/rnnoise/src
#CF="-O2 -mtune=native -DRNN_ENABLE_X86_RTCD -I../include -I./"
#CF="-DDISABLE_DEBUG_FLOAT -O3 -march=native -I../include -I./"
#CF=" -O3 -march=native -I../include -I./"
CF="-DHIGH_ACCURACY -DDISABLE_DEBUG_FLOAT -O3 -march=native -I../include -I./"

# 基线组：只到 SSE2
for f in denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c nnet.c nnet_default.c \
         parse_lpcnet_weights.c rnnoise_tables.c rnnoise_data.c dump_rnn_io.c \
         x86/x86_dnn_map.c x86/x86cpu.c; do
  gcc $CF -msse2 -c "$f" -o "$(basename "$f" .c).o" || exit 1
done

# 分派组：严格锁死 ISA，不加 native
gcc $CF -msse4.1           -c x86/nnet_sse4_1.c -o nnet_sse4_1.o || exit 1
gcc $CF -mavx -mfma -mavx2 -c x86/nnet_avx2.c  -o nnet_avx2.o  || exit 1
#
gcc $CF denoise.o kiss_fft.o pitch.o celt_lpc.o rnn.o nnet.o nnet_default.o \
    parse_lpcnet_weights.o rnnoise_tables.o rnnoise_data.o dump_rnn_io.o \
    x86_dnn_map.o x86cpu.o nnet_sse4_1.o nnet_avx2.o -o dump_rnn_io -lm


/*

dump 小工具
输入：features.f32（每帧 65 个 float）
输出：gains.f32（每帧 32 个 float）+ vad.f32（每帧 1 个 float）

把神经网络从 DSP（FFT/基音/OLA）里剥离出来。
所有精度/性能对比都只比 compute_rnn()，口径干净。

验收（CP-0）
- [ ] 报告 4 个变体两两之间的 `gains` 最大/平均绝对误差，并**归因**：
  - `debugfloat-hiacc` vs `debugfloat` → 误差源 ①（激活近似），应 ≈ 6e-5 量级
  - `debugfloat` vs 默认 → 误差源 ②（权重量化），这是你的**生产基线误差**
- [ ] 回答：为什么 `float_weights` 字段是 NULL 时才走 int8？
（线索：`src/parse_lpcnet_weights.c:91-96` 的 `opt_array_check` 对缺失返回 NULL 且不报错）

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "rnnoise.h"
#include "common.h"
#include "denoise.h"
#include "arch.h"
#include "kiss_fft.h"
#include "_kiss_fft_guts.h"
#include "rnn.h"

/* 
This tool drives the network directly, so it needs the *full* DenoiseState
   (model / rnn / arch fields) which is only defined in denoise.c. We reproduce
   the exact layout here so pointers returned by the C implementation (linked
   from denoise.c) line up. Build WITHOUT -DTRAINING (it runs the net, so the
   arch field must exist): 
   build_dump.sh
*/

struct DenoiseState {
  RNNoise model;
#ifndef TRAINING
  int arch;
#endif
  float analysis_mem[FRAME_SIZE];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  RNNState rnn;
  kiss_fft_cpx delayed_X[FREQ_SIZE];
  kiss_fft_cpx delayed_P[FREQ_SIZE];
  float delayed_Ex[NB_BANDS], delayed_Ep[NB_BANDS];
  float delayed_Exp[NB_BANDS];
};

int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

#define SEQUENCE_LENGTH 2000
#define SEQUENCE_SAMPLES (SEQUENCE_LENGTH*FRAME_SIZE)

#define RIR_FFT_SIZE 65536
#define RIR_MAX_DURATION (RIR_FFT_SIZE/2)
#define FILENAME_MAX_SIZE 1000

static unsigned rand_lcg(unsigned *seed) {
  *seed = 1664525**seed + 1013904223;
  return *seed;
}

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static float randf(float f) {
  return f*rand()/(double)RAND_MAX;
}

static void rand_filt(float *a) {
  if (rand()%3!=0) {
    a[0] = a[1] = 0;
  }
  else if (uni_rand()>0) {
    float r, theta;
    r = rand()/(double)RAND_MAX;
    r = .7*r*r;
    theta = rand()/(double)RAND_MAX;
    theta = M_PI*theta*theta;
    a[0] = -2*r*cos(theta);
    a[1] = r*r;
  } else {
    float r0,r1;
    r0 = 1.4*uni_rand();
    r1 = 1.4*uni_rand();
    a[0] = -r0-r1;
    a[1] = r0*r1;
  }
}

static void rand_resp(float *a, float *b) {
  rand_filt(a);
  rand_filt(b);
}


short speech16[SEQUENCE_LENGTH*FRAME_SIZE];
short noise16[SEQUENCE_LENGTH*FRAME_SIZE];
short fgnoise16[SEQUENCE_LENGTH*FRAME_SIZE];
float x[SEQUENCE_LENGTH*FRAME_SIZE];
float n[SEQUENCE_LENGTH*FRAME_SIZE];
float fn[SEQUENCE_LENGTH*FRAME_SIZE];
float xn[SEQUENCE_LENGTH*FRAME_SIZE];

struct rir_list {
  int nb_rirs;
  kiss_fft_state *fft;
  kiss_fft_cpx **rir;
  kiss_fft_cpx **early;
};

#define P00 0.99f
#define P01 0.01f
#define P10 0.01f
#define P11 0.99f
#define LOGIT_SCALE 0.5f

static void viterbi_vad(const float *E, int *vad) {
  int i;
  float Enoise, Esig;
  int back[SEQUENCE_LENGTH][2];
  float curr;
  Enoise = Esig = 1e-30;
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    Esig += E[i]*E[i];
  }
  Esig = sqrt(Esig/SEQUENCE_LENGTH);
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    Enoise += 1.f/(1e-8*Esig*Esig + E[i]*E[i]);
  }
  Enoise = 1.f/sqrt(Enoise/SEQUENCE_LENGTH);
  curr = 0.5;
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    float p0, pspeech, pnoise;
    float prior;
    p0 = (log(1e-15+E[i]) - log(Enoise))/(.01 + log(Esig) - log(Enoise));
    p0 = MIN16(.9f, MAX16(.1f, p0));
    p0 = 1.f/(1.f + pow((1.f-p0)/p0, LOGIT_SCALE));
    if (curr*P11 > (1-curr)*P01) {
      back[i][1] = 1;
      prior = curr*P11;
    } else {
      back[i][1] = 0;
      prior = (1-curr)*P01;
    }
    pspeech = prior*p0;

    if ((1-curr)*P00 > curr*P10) {
      back[i][0] = 0;
      prior = (1-curr)*P00;
    } else {
      back[i][0] = 1;
      prior = curr*P10;
    }
    pnoise = prior*(1-p0);
    curr = pspeech / (pspeech + pnoise);
    /*printf("%f ", curr);*/
  }
  vad[SEQUENCE_LENGTH-1] = curr > .5;
  for (i=SEQUENCE_LENGTH-2;i>=0;i--) {
    if (vad[i+1]) {
      vad[i] = back[i+1][1];
    } else {
      vad[i] = back[i+1][0];
    }
  }
  for (i=0;i<SEQUENCE_LENGTH-1;i++) {
    if (vad[i+1]) vad[i] = 1;
  }
  for (i=SEQUENCE_LENGTH-1;i>=1;i--) {
    if (vad[i-1]) vad[i] = 1;
  }
}

static void clear_vad(float *x, int *vad) {
  int i;
  int active = vad[0];
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    if (!active) {
      if (i<SEQUENCE_LENGTH-1 && vad[i+1]) {
        int j;
        for (j=0;j<FRAME_SIZE;j++) x[i*FRAME_SIZE+j] *= j/(float)FRAME_SIZE;
        active = 1;
      } else {
        RNN_CLEAR(&x[i*FRAME_SIZE], FRAME_SIZE);
      }
    } else {
      if (i>=1 && vad[i]==0 && vad[i-1]==0) {
        int j;
        for (j=0;j<FRAME_SIZE;j++) x[i*FRAME_SIZE+j] *= 1.f - j/(float)FRAME_SIZE;
        active = 0;
      }
    }
  }
  /*printf("\n");
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    printf("%d ", vad[i]);
  }
  printf("\n");*/
}

static float weighted_rms(float *x) {
  int i;
  float tmp[SEQUENCE_SAMPLES];
  float weighting_b[2] = {-2.f, 1.f};
  float weighting_a[2] = {-1.89f, .895f};
  float mem[2] = {0};
  float mse = 1e-15f;
  rnn_biquad(tmp, mem, x, weighting_b, weighting_a, SEQUENCE_SAMPLES);
  for (i=0;i<SEQUENCE_SAMPLES;i++) mse += tmp[i]*tmp[i];
  return 0.9506*sqrt(mse/SEQUENCE_SAMPLES);
}

int main( int ac, char **argv) {
  if( ac < 3) {
    printf("Usage: ./exe <features.f32> <gains.f32> [count=500] \n");
    return 0;
  }
  int maxCount = 500;
  if( ac == 4) {
    maxCount = atoi(argv[3])>0?atoi(argv[3]):500;
  }

  float vad_prob = 0;
  DenoiseState *st;
  static const float a_hp[2] = {-1.99599f, 0.99600f};
  static const float b_hp[2] = {-2.f, 1.f};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_fgnoise[2] = {0};
  float b_fgnoise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float speech_gain = 1, noise_gain = 1, fgnoise_gain = 1;
  FILE *f1, *fout;
  long speech_length, noise_length, fgnoise_length;
  unsigned seed = 12345;
  st = rnnoise_create(NULL);

  f1 = fopen(argv[1], "rb"); // features.f32
  if (f1 == NULL) {
    fprintf(stderr, "Open %s failed\n", argv[1]);
    return -1;
  }
  fout = fopen(argv[2], "wb"); // gains.f32
  if (fout == NULL) {
    fprintf(stderr, "Open %s failed\n", argv[2]);
    return -1;
  }
  fseek(f1, 0, SEEK_END);
  speech_length = ftell(f1);
  fseek(f1, 0, SEEK_SET);
  
  for (int count = 0; count < maxCount; ++count) {
    float features[NB_FEATURES];
    float g[NB_BANDS], g_real[NB_BANDS];
    float vad, vad_real;
    size_t features_count = fread(features, sizeof(float), NB_FEATURES, f1);
    size_t g_count = fread(g, sizeof(float), NB_BANDS, f1); 
    size_t vad_target_count = fread(&vad, sizeof(float), 1, f1); 
    if (features_count != NB_FEATURES || g_count != NB_BANDS) {
      fprintf(stderr, "Warning: read features.f32 failed, features=%luor g=%lu\n",features_count, g_count );
      break;
    }

    compute_rnn(&st->model, &st->rnn, g_real, &vad_real, features, st->arch);
    fwrite(&g_real, sizeof(float), NB_BANDS, fout);
    fwrite(&vad_real, sizeof(float), 1, fout);
    fprintf(stdout, "%d vad:%f\ngains: ", count + 1, vad_real);
    for( int i = 0; i < NB_BANDS; ++i) {
      fprintf(stdout, "%f ", g_real[i]);
    }
    fprintf(stdout, "\n");

#if 0
    int rir_id;
    int vad[SEQUENCE_LENGTH];
    long speech_pos;
    int start_pos=0;
    float E[SEQUENCE_LENGTH] = {0};
    float mem[2]={0};
    int frame;
    int silence;
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float speech_rms, noise_rms, fgnoise_rms;
    if ((count%1000)==0) 
      fprintf(stderr, "%d\r", count);

    speech_pos = (rand_lcg(&seed)*2.3283e-10)*speech_length;
    if (speech_pos > speech_length-(long)sizeof(speech16)) 
      speech_pos = speech_length-sizeof(speech16);

    speech_pos -= speech_pos&1;
    fseek(f1, speech_pos, SEEK_SET);
    fread((void*)(&speech16), sizeof(speech16), 1, f1);
    if (rand()%4) 
      start_pos = 0;
    else 
      start_pos = -(int)(1000*log(rand()/(float)RAND_MAX));
    
    start_pos = IMIN(start_pos, SEQUENCE_LENGTH*FRAME_SIZE);

    speech_gain = pow(10., (-45+randf(45.f)+randf(10.f))/20.);
    noise_gain = pow(10., (-30+randf(40.f)+randf(15.f))/20.);
    fgnoise_gain = pow(10., (-30+randf(40.f)+randf(15.f))/20.);

    if (rand()%8==0) 
      noise_gain = 0;
    if (rand()%8!=0) 
      fgnoise_gain = 0;
    if (rand()%12==0) {
      noise_gain *= 0.03;
      fgnoise_gain *= 0.03;
    }
    noise_gain *= speech_gain;
    fgnoise_gain *= speech_gain;

    rand_resp(a_noise, b_noise);
    rand_resp(a_fgnoise, b_fgnoise);
    rand_resp(a_sig, b_sig);
    lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
    for (i=0;i<NB_BANDS;i++) {
      if (eband20ms[i] > lowpass) {
        band_lp = i;
        break;
      }
    }

    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      E[frame] = 0;
      for(j=0;j<FRAME_SIZE;j++) {
        float s = speech16[frame*FRAME_SIZE+j];
        E[frame] += s*s;
        x[frame*FRAME_SIZE+j] = speech16[frame*FRAME_SIZE+j];
        n[frame*FRAME_SIZE+j] = noise16[frame*FRAME_SIZE+j];
        fn[frame*FRAME_SIZE+j] = fgnoise16[frame*FRAME_SIZE+j];
      }
    }
    viterbi_vad(E, vad);

    RNN_CLEAR(mem, 2);
    rnn_biquad(x, mem, x, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(x, mem, x, b_sig, a_sig, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(n, mem, n, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(n, mem, n, b_noise, a_noise, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(fn, mem, fn, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(fn, mem, fn, b_fgnoise, a_fgnoise, SEQUENCE_LENGTH*FRAME_SIZE);

    speech_rms = weighted_rms(x);
    noise_rms = weighted_rms(n);
    fgnoise_rms = weighted_rms(fn);

    RNN_CLEAR(vad, start_pos/FRAME_SIZE);
    clear_vad(x, vad);

    speech_gain *= 3000.f/(1+speech_rms);
    noise_gain *= 3000.f/(1+noise_rms);
    fgnoise_gain *= 3000.f/(1+fgnoise_rms);
    for (j=0;j<SEQUENCE_SAMPLES;j++) {
      x[j] *= speech_gain;
      n[j] *= noise_gain;
      fn[j] *= fgnoise_gain;
      xn[j] = x[j] + n[j] + fn[j];
    }
    
    if (rand()%4==0) {
      /* Apply input clipping to 0 dBFS (don't clip target). */
      for (j=0;j<SEQUENCE_SAMPLES;j++) {
        xn[j] = MIN16(32767.f, MAX16(-32767.f, xn[j]));
      }
    }
    if (rand()%2==0) {
      /* Apply 16-bit quantization. */
      for (j=0;j<SEQUENCE_SAMPLES;j++) {
        xn[j] = floor(.5f + xn[j]);
      }
    }
#endif
  /*
  rnn_compute_frame_features(st, X, P, Ex, Ep, Exp, features, in)
    │
    ├─ 输入：480 采样 + 跨帧状态 st
    │
    └─ 输出：features[65]  ←── 这就是网络 conv1 的输入
            │
            ├─ features[0..31]  = DCT( log10(Ex) )        频谱包络
            ├─ features[32..63] = DCT( 归一化 X↔P 相关性 )   基音/浊音
            └─ features[64]     = .01*(pitch_index-300)    基音周期
            然后 features[0]-=12, features[1]-=4  → 值域约 [−12,+5]

    附带：Ex/Ep/Exp（各 32）、X/P（频谱）、返回 silence 标志
*/
#if 0
    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      float vad_target;
      // rnn_frame_analysis(st, Y, Ey, &x[frame*FRAME_SIZE]);
      // silence = rnn_compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, &xn[frame*FRAME_SIZE]);

      compute_rnn(&st->model, &st->rnn, g, &vad_prob, features, st->arch);
      /*rnn_pitch_filter(X, P, Ex, Ep, Exp, g);*/
      vad_target = vad[frame];
      for (int i=0;i<NB_BANDS;i++) {
        g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
        if (g[i] > 1) g[i] = 1;
        if (silence || i > band_lp) g[i] = -1;
        if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
        if (vad_target==0 && noise_gain==0 && fgnoise_gain==0) g[i] = -1;
      }

    }
  #endif 
  }
  if (f1!=NULL) {
    fflush(f1);
    fclose(f1);
  }
  if (fout!= NULL) {
    fflush(fout);
    fclose(fout);
  }
  return 0;
}
#!/usr/bin/env python3
# coding=utf-8
"""RNN / GRU 手撕练习：不用 nn.RNN / nn.GRU，逐时间步自己写递推。

对应 `doc/phase1学习输出.md` 的「RNN 学习」和「GRU」两节，
目的是把公式落成能跑的代码，并逐步观察每个中间张量的形状和数值范围。

---- 统一约定（row-vector 写法，和 D2L 一致）----

    X_t              [B, F]     单个时间步的输入
    H_{t-1}, H_t     [B, H]     隐状态
    W_x*             [F, H]     输入   -> 隐状态
    W_h*             [H, H]     隐状态 -> 隐状态

---- RNN 递推 ----

    H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)
    O_t = H_t @ W_hq + b_q

---- GRU 递推 ----

    R_t = sigmoid(X_t @ W_xr + H_{t-1} @ W_hr + b_r)         重置门：历史参与候选多少
    Z_t = sigmoid(X_t @ W_xz + H_{t-1} @ W_hz + b_z)         更新门：最终保留多少旧状态
    H~  = tanh(X_t @ W_xh + (R_t ⊙ H_{t-1}) @ W_hh + b_h)    候选状态（新的状态内容）
    H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H~                     旧状态与新候选的逐元素插值

    极端取值下：Z=1 -> H_t = H_{t-1}（恒等复制，此时 R 失效）；
                Z=0, R=0 -> H_t = tanh(X_t @ W_xh + b_h)，退化成 MLP；
                Z=0, R=1 -> H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)，退化成普通 RNN。

---- 用法（本仓库用 uv 管理 python 环境）----

    uv run examples/gru.py rnn    # 跑 RNN 手撕实现
    uv run examples/gru.py gru    # 跑 GRU 手撕实现（单层、随机权重的教学 demo）

激活函数的近似实现已经独立到 `examples/rnnoise_activation.py`（tanh_approx /
sigmoid_approx / sigmoid_from_tanh / check_sigmoid_diff / pick）。

---- 已知待修正：与 nn.GRUCell 对齐时误差很大（约 1e+00 量级）----

  ① `get_gru_params_from_net()` 的 gate 分块顺序错了。
     PyTorch 的 weight_ih / bias_ih 按 **[r | z | n]** 顺序分块，
     当前取的是 h←r、r←z、z←n，整体串位了一位。

  ② 只读了 `bias_ih`，漏了 `bias_hh`。
     `nn.GRUCell(bias=True)` 两个都用，而且 **candidate 的 b_hh 要乘在 reset gate 里面**：
         n = tanh(i_n + b_in + r ⊙ (h @ W_hn.T + b_hn))
                                      ^^^^^^^^^^^^^^^^^^ b_hn 在括号里

  ③ candidate 写成了 `(R ⊙ H) @ W_hh`，属于教材常见变体。
     而 `nn.GRUCell` 和 RNNoise 的 `compute_generic_gru()`（src/nnet.c:65-94）
     都是 **R ⊙ (W_hh·H + b_hh)** —— reset 乘在投影之后。两者数学上不等价。

  另外注意：float32 下 sigmoid 会被舍入到**精确的 1.0**（pre-activation 一大就饱和），
  所以上面"严格小于 1"只在中等幅度下成立。
"""

import sys

import torch
from torch import nn          # get_gru_params_from_net() 的类型注解要用；缺了连模块都 import 不了

from rnnoise_activation import pick


# =====================================================================
# RNN —— TODO: 这段是 RNN 的教学脚手架，**它不是 GRU**，暂时留在本文件里
# =====================================================================

def get_params(nb_features, nb_hiddens, nb_outputs):
    """随机初始化。

    权重必须随机（打破神经元对称性，否则同一层的神经元会得到相同的梯度），
    并且不能太大——这里的 0.01 是为了让初始输出落在 tanh 的线性区。
    bias 可以是 0，因为 W 已经打破了对称性。
    """
    W_xh = torch.randn((nb_features, nb_hiddens), dtype=torch.float32) * 0.01
    W_hh = torch.randn((nb_hiddens, nb_hiddens), dtype=torch.float32) * 0.01
    b_h = torch.zeros(nb_hiddens, dtype=torch.float32)

    # 输出层参数，不属于 RNN 本体
    W_hq = torch.randn((nb_hiddens, nb_outputs), dtype=torch.float32) * 0.01
    b_q = torch.zeros(nb_outputs, dtype=torch.float32)
    return W_xh, W_hh, b_h, W_hq, b_q


def rnn(inputs, params, state, low_accuracy=True):
    """逐时间步手写 RNN 前向。不用 nn.RNN。

    inputs: [T, B, F]  整个序列
    state:  [B, H]     初始隐状态 H_0
    low_accuracy: True -> rnnoise_activation 的 tanh 多项式；False -> torch.tanh
    返回:   (outputs, states)
            outputs 是把每步的 [B, O] 沿 batch 维拼起来的 [T*B, O]
            states  是长度为 T 的 list，每项 [B, H]
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    _, tanh = pick(low_accuracy)            # 高/低精度只换激活函数
    outputs = []
    states = []
    for X in inputs:                       # X: [B, F]，循环变量就是时间步 t
        s = tanh(X @ W_xh + state @ W_hh + b_h)   # H_t   [B, H]
        output = s @ W_hq + b_q                         # O_t   [B, O]
        outputs.append(output)
        states.append(s)
    return torch.cat(outputs, dim=0), states


class RNNScratch:
    def __init__(self, batch_size, nb_features, nb_hiddens, nb_outputs):
        self.params = get_params(nb_features, nb_hiddens, nb_outputs)

    def __call__(self, inputs, state, low_accuracy=True):
        return rnn(inputs, self.params, state, low_accuracy=low_accuracy)


def rnn_demo(out=sys.stdout):
    input_size = 3          # F：每个时间步的特征数
    batch_size = 2          # B：一次处理几个样本
    hidden_size = 4         # H：隐状态维度（超参数）
    sequence_length = 5     # T：一共多少个时间步
    output_size = 4         # O：输出特征数

    # 输入是三维：[T, B, F]
    inputs = torch.randn((sequence_length, batch_size, input_size)) * .1
    h0 = torch.randn((batch_size, hidden_size), dtype=torch.float32)

    net = RNNScratch(batch_size, input_size, hidden_size, output_size)
    outputs, h_list = net(inputs, h0)

    # rnn() 把每步的输出沿 batch 维 cat 成了 [T*B, O]，
    # 所以第 t 步的输出要切片取：outputs[t*B : (t+1)*B]
    for t, h in enumerate(h_list):
        o = outputs[t * batch_size:(t + 1) * batch_size]
        print(f'step {t}  out:{o}, h:{h}')

import pdb
# =====================================================================
# GRU
# =====================================================================
def get_gru_params_from_case(case,input_features_size, hidden_size, output_features_size):
    # h@z/r/h_t-1 in using *_recurrent_weights_float in .c
    # x@z/r/h_t-1 in using *_input_weights_float in .c
    # case 是「某一层」的子对象（examples/rnn_unit.py 的 _load_gru_case() 里按层取出来）
    W_xzrh = torch.tensor(
        case['input_weights_float'], 
        dtype=torch.float32).reshape(input_features_size,-1)
    
    b_xzrh = torch.tensor(case['input_bias'], dtype=torch.float32)

    W_hzrh = torch.tensor(
        case['recurrent_weights_float'], 
        dtype=torch.float32).reshape(hidden_size,-1)

    b_hzrh = torch.tensor(case['recurrent_bias'], dtype=torch.float32)
    
    return W_xzrh, W_hzrh, b_xzrh, b_hzrh


def get_gru_params(input_features_size, hidden_size, output_features_size):
    """随机初始化 GRU 本体的 9 组参数（h / z / r 各 3 组）。
    output_features_size 这里用不到 —— GRU 本体没有输出层，
    W_hq / b_q 属于外挂的 output projection。
    """
    # 候选状态 h
    W_xh = torch.randn((input_features_size, hidden_size), dtype=torch.float32)
    W_hh = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    b_h = torch.zeros(hidden_size, dtype=torch.float32)

    # 更新门 z
    W_xz = torch.randn((input_features_size, hidden_size), dtype=torch.float32)
    W_hz = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    b_z = torch.zeros(hidden_size, dtype=torch.float32)

    # 重置门 r
    W_xr = torch.randn((input_features_size, hidden_size), dtype=torch.float32)
    W_hr = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    b_r = torch.zeros(hidden_size, dtype=torch.float32)

    return W_xh, W_hh, b_h, W_xz, W_hz, b_z, W_xr, W_hr, b_r

def get_gru_params_from_net(net: nn.GRUCell):
    """从 nn.GRUCell 里取出参数，转换成上面那套 (X @ W) 写法。
    """
    hidden_size = net.hidden_size
    W_xh = net.state_dict()['weight_ih'][:hidden_size].T
    W_xr = net.state_dict()['weight_ih'][hidden_size:2*hidden_size].T
    W_xz = net.state_dict()['weight_ih'][hidden_size*2:].T
    W_hh = net.state_dict()['weight_hh'][:hidden_size].T
    W_hr = net.state_dict()['weight_hh'][hidden_size:2*hidden_size].T
    W_hz = net.state_dict()['weight_hh'][2*hidden_size:].T

    b_h = net.state_dict()['bias_ih'][:hidden_size]
    b_r = net.state_dict()['bias_ih'][hidden_size:2*hidden_size]
    b_z = net.state_dict()['bias_ih'][2*hidden_size:]

    return W_xh, W_hh, b_h, W_xz, W_hz, b_z, W_xr, W_hr, b_r

def emit(out, name, frame, values):
    """按与 C 侧完全一致的协议打印一项：`[name#frame] v0 v1 ...`，每值 %.8e。

    对应 C 的 examples/rnn_unit_util.c: print_item()。out 为 None 时静默，
    这样调用方想要纯计算、不打印也不会被干扰。
    """
    if out is None:
        return
    out.write("[%s#%d] %s\n" % (
        name, frame, " ".join("%.8e" % float(v) for v in values)))


def do_rnnoise_gru_step(x, params, hidden, low_accuracy=True, out=sys.stdout,
                        layer="gru", frame=0, detail=True):
    """按 RNNoise C 实现方式递推**一帧**（对应 src/nnet.c: compute_generic_gru()）。
    weights 布局 [z|r|h]，hidden / x 各自有 bias，不单独计算 R、Z、H_candidate。

    递推和 examples/rnn_unit.c 的 local_compute_generic_gru() 一一对应，
    打印项名字也逐字节一致（<layer>.zrh_recur / <layer>.sigmoid /
    <layer>.recur_tanh / <layer>.h / <layer>.state），脚本才能按 [name#frame] 配对。

    ★ 一次只吃**一帧**：整段序列的递推由调用方按帧循环，状态也由调用方自己存 ——
      和 C 一样（C 每次也只处理一帧，状态是调用方传进来的数组）。
      整网里就是 gru1 的输出喂 gru2、gru2 的输出喂 gru3，三个 gru 各存各的隐状态。

    x:            [F]   这一帧的输入（不是整段序列）
    hidden:       [H]   上一帧的隐状态
    frame:        帧号，只用于协议行（C 侧同样用调用方的循环变量）
    low_accuracy: True  -> 用本文件手写的多项式近似（对应 C 的 vec_sigmoid / vec_tanh）
                  False -> 用 torch.sigmoid / torch.tanh（对应 C 的 -DHIGH_ACCURACY）
    out:          协议输出目标；None 表示不打印
    layer:        层名，做协议行前缀（gru1 / gru2 / gru3 共用这一份实现）
    detail:       True  -> 打全部 5 项（zrh_recur / sigmoid / recur_tanh / h / state），
                          对应 C 的 gru 单元（走 `local_compute_generic_gru()`，中间量齐全）
                  False -> 只打 `<layer>.state`，对应 C 的整网链路 `conv_gru_unit()`
                          （那边走的是正版 `compute_generic_gru()`，拿不到中间量）

    返回这一帧算出的新隐状态 [H]，调用方把它喂给下一帧。
    """
    assert params
    hidden_size = hidden.shape[0]
    W_xzrh, W_hzrh, b_xzrh, b_hzrh = params
    # 高/低精度只换激活函数，递推本身完全一样
    # （两套近似实现搬到了 examples/rnnoise_activation.py）
    sigmoid, tanh = pick(low_accuracy)
    # 一次算出 zrh：三个门（z / r / h）的输入投影一起算，公式里都和 x 相乘
    zrh = x @ W_xzrh + b_xzrh          # compute_linear(input_weights, zrh, in, arch);
    # 一次算出 recur：三个门的递归投影也一起算，都和 hidden_t-1 相乘
    recur = hidden @ W_hzrh + b_hzrh   # compute_linear(recurrent_weights, recur, state, arch);

    # C: for (i=0;i<2*N;i++) zrh[i] += recur[i];
    pre = zrh[:hidden_size*2] + recur[:hidden_size*2]
    if detail:
        emit(out, "%s.zrh_recur" % layer, frame, pre)

    # C: compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID, arch);
    # z 和 r 一起过 sigmoid；只覆盖前 2N 个，zrh[2*N:] 仍是候选状态的输入投影
    zrh[:hidden_size*2] = sigmoid(pre)  # [H*2]，收敛到 (0, 1)
    if detail:
        emit(out, "%s.sigmoid" % layer, frame, zrh[:hidden_size*2])

    # 对应c代码
    #   for (i=0;i<N;i++)
    #       h[i] += recur[2*N+i]*r[i];
    r = zrh[hidden_size:hidden_size*2]
    # 注意上面那行 C 代码里 h 指向 zrh[2*N]，是复用了候选状态的输入投影，
    # 所以这里必须补上 zrh[2*N:] 这一项（之前漏掉，导致和 C 不等价）
    h_candidate = tanh(zrh[hidden_size*2:] + recur[hidden_size*2:]*r)
    if detail:
        emit(out, "%s.recur_tanh" % layer, frame, h_candidate)

    # 对应c代码
    # for (i=0;i<N;i++)
    #     h[i] = z[i]*state[i] + (1-z[i])*h[i];
    # for (i=0;i<N;i++)
    #     state[i] = h[i];
    z = zrh[:hidden_size]
    h = z * hidden + (1 - z) * h_candidate
    if detail:
        emit(out, "%s.h" % layer, frame, h)
    emit(out, "%s.state" % layer, frame, h)
    return h


def do_gru(inputs_data, params, hidden, low_accuracy=True):
    """逐时间步手写 GRU 前向，可选地与 nn.GRUCell 逐步对比误差。

    注意这是**教材版**布局（9 组参数、`X @ W_xh` 那种写法），
    不是 RNNoise 的 [z|r|h] 布局 —— 后者是 `do_rnnoise_gru_step()`。

    inputs_data: [T, B, F]
    hidden:      [B, H]    初始隐状态
    low_accuracy: True -> rnnoise_activation 的多项式；False -> torch 自带

    返回 (states, candidates)，都是长度为 T 的 list，每项 [B, H]。
    """
    assert params

    resets = []
    updates = []
    candidates = []
    states = []
    h_comp = hidden.clone()
    W_xh, W_hh, b_h, W_xz, W_hz, b_z, W_xr, W_hr, b_r = params
    sigmoid, tanh = pick(low_accuracy)     # 高/低精度只换激活函数

    for step, X in enumerate(inputs_data):  # X: [B, F]
        # 重置门：控制上一轮隐状态有多少参与"生成候选状态"
        rr = X @ W_xr + hidden @ W_hr + b_r
        r = sigmoid(rr)                    # [B, H]，收敛到 (0, 1)
        # print(f'reset\n{rr}\n{r}')

        # 更新门：控制最终保留多少旧状态、采用多少候选状态
        zz = X @ W_xz + hidden @ W_hz + b_z
        z = sigmoid(zz)                    # [B, H]，收敛到 (0, 1)
        # print(f'update\n{zz}\n{z}')

        # 候选状态：(r * hidden) 先按元素缩放，再和 W_hh 做矩阵乘法
        hh = X @ W_xh + (r * hidden) @ W_hh + b_h
        h_candidate = tanh(hh)             # [B, H]，收敛到 (-1, 1)
        # print(f'h_candidate\n{hh}\n{h_candidate}')

        # r、z 已经各自做了 sigmoid，这里直接用
        h = z * hidden + (1 - z) * h_candidate
        print(f'state {step}: {h.data}')
        hidden = h                         # 把新状态传给下一个时间步

        resets.append(r)
        updates.append(z)
        candidates.append(h_candidate)
        states.append(h)

    return states, candidates


class GRUScratch():
    """一层 RNNoise GRU：参数和隐状态都在实例里，逐帧调用。

    一次只喂**一帧**（x: [F]），返回这一帧算出的新隐状态；隐状态存在 self.hidden 里，
    所以每个实例各维护自己那份 —— 整网里三个 gru 就是三个实例：

        h = gru1(x)     # gru1 的输出
        h = gru2(h)     # 作为 gru2 的输入
        h = gru3(h)     # gru2 的输出作为 gru3 的输入

    整段序列要自己按帧循环：

        for frame in range(nb_frames):
            h = net(x[frame], low_accuracy=..., out=out, frame=frame)

    递推和 C 的 compute_generic_gru() 一致（C 每次也只处理一帧）。
    初始隐状态是零 —— C 侧同样是 memset 成 0，用例 json 里的 state 只做完整性校验。
    """

    def __init__(self, input_features_size, hidden_size, output_features_size,
                 case=None, layer="gru", detail=True):
        self.layer = layer
        # detail=True  -> 打全部 5 项（和 C 的 gru 单元一致）
        # detail=False -> 只打 <layer>.state（和 C 的整网链路 conv_gru_unit 一致）
        self.detail = detail
        if case:
            self.params = get_gru_params_from_case(case,input_features_size, hidden_size, output_features_size)
        else:
            self.params = get_gru_params(input_features_size, hidden_size, output_features_size)
        self.hidden = torch.zeros(hidden_size, dtype=torch.float32)

    def __call__(self, x, low_accuracy=True, out=sys.stdout, frame=0):
        self.hidden = do_rnnoise_gru_step(x, self.params, self.hidden,
                                          low_accuracy=low_accuracy, out=out,
                                          layer=self.layer, frame=frame,
                                          detail=self.detail)
        return self.hidden


def gru_demo(out=sys.stdout):
    """教学 demo：随机权重 + 教材版递推（`do_gru()`），逐帧打印状态与输出。

    注意这里用的是**教材版**布局（9 组参数、`torch` 激活），**不是** RNNoise 的
    [z|r|h] 布局 —— 后者是 `do_rnnoise_gru_step()` / `GRUScratch`，由
    `examples/rnn_unit.py` 调。
    """
    time_step = 2               # T
    batch_size = 3              # B
    input_features_size = 4     # F
    hidden_size = 6             # H
    output_features_size = 5    # O

    inputs_data = torch.randn((time_step, batch_size, input_features_size))
    hidden_0 = torch.randn((batch_size, hidden_size))

    # 💡 输出层参数不属于 GRU 本体，单独准备
    W_hq = torch.randn((hidden_size, output_features_size), dtype=torch.float32)
    b_q = torch.zeros(output_features_size, dtype=torch.float32)

    # 官方实现：n = tanh(X @ W_in + b_in + r ⊙ (W_hn @ H + b_hn))
    # gru = nn.GRUCell(input_features_size, hidden_size, bias=True,
    #                  device='cpu', dtype=torch.float32)

    params = get_gru_params(input_features_size, hidden_size, output_features_size)
    states, candidates = do_gru(inputs_data, params, hidden_0)

    outputs = []
    for i in range(time_step):
        print(f'-------- {i + 1} --------')
        # 想观察哪个中间量，就把对应的行取消注释：
        # print(f'reset gate {resets[i]}')
        # print(f'update gate {updates[i]}')
        # print(f'candidate {candidates[i]}')
        # print(f'new hidden {states[i]}')
        #
        # 注意 gate 的数值范围（激活函数起到收敛作用）：
        # print(f'reset gates   :{resets[0].min()}, {resets[0].max()}')     # 0 ~ 1
        # print(f'updates gates :{updates[0].min()}, {updates[0].max()}')   # 0 ~ 1
        # print(f'candidates    :{candidates[0].min()}, {candidates[0].max()}')  # -1 ~ 1

        o = states[i] @ W_hq + b_q
        # print(f'output {o.min()}, {o.max()}')
        # print(f'out {o}')
        outputs.append(o)


# =====================================================================
# 入口
# =====================================================================

UNITS = {
    "rnn": rnn_demo,
    "gru": gru_demo,
}

def main(argv):
    torch.set_printoptions(sci_mode=True, precision=8)
    gru_demo()
    # rnn_demo()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

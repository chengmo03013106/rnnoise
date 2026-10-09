#!/usr/bin/env python3
# coding=utf-8
"""GRU 单元测试。

被测对象：`examples/gru.py`
    get_gru_params / do_gru / GRUMo

（2026-10-06 起 GRU 的实现全都在 examples/gru.py 了。这些用例以前 import 的是
 `examples/rnn_unit.py` 和里面那个早已删除的 `GRU` 类 / `compare_gru` 参数，
 整个文件 15/17 失败 —— 现在改成对着真实存在的接口测。文件名暂时没跟着改。）

验收依据：`doc/phase1学习输出.md`
    - §12         gate / candidate 的数值范围 sanity check
    - §15         shape 表
    - 214-225 行  Z / R 取极端值时的退化行为

运行（用 --with 临时叠加 pytest，不往 venv 里装东西）：
    uv run --with pytest pytest tests/ -v

---- 与被测实现对齐的三件事（都读过源码确认，不是猜的）----

1. `do_gru(inputs, params, hidden, low_accuracy=True)` **没有** `compare_gru` 参数，
   也不再内部调用 nn.GRUCell —— 它只是教材版递推本身。
2. 它返回 **4 项** `(states, resets, updates, candidates)`，
   所以本文件统一解包 4 项。
3. gate 在 float32 下可以饱和到精确的 0 / 1：
       sigmoid(+60) == 1.0        （exp(-60) ≈ 8.8e-27，相对 1 已下溢）
       sigmoid(-60) ≈ 8.8e-27
   所以极端取值下的断言可以卡得很紧。
"""

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import gru  # noqa: E402  (必须在 sys.path 调整之后)


# ===================== 公共设置 =====================

T, B, F, H = 3, 2, 4, 5        # 时间步 / batch / 输入特征 / 隐状态

BIG = 60.0                     # sigmoid(±60) 在 float32 下饱和到 1 / 0
SAT_ATOL = 1e-6                # 饱和断言容差
F32_ATOL = 1e-5                # 一般数值对齐容差

# get_gru_params() 的返回顺序；do_gru() 是按位置解包的，顺序错了整体就错
PARAM_ORDER = ("W_xh", "W_hh", "b_h", "W_xz", "W_hz", "b_z", "W_xr", "W_hr", "b_r")


def make_inputs(t=T, b=B, f=F, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t, b, f, generator=g) * scale


def make_hidden(seed=7, b=B, h=H):
    return torch.randn(b, h, generator=torch.Generator().manual_seed(seed))


def base_params(f=F, h=H, seed=0, scale=0.5):
    """按名字构造参数字典，方便单独改某一项。"""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g) * scale

    return {
        "W_xh": rnd(f, h), "W_hh": rnd(h, h), "b_h": torch.zeros(h),
        "W_xz": rnd(f, h), "W_hz": rnd(h, h), "b_z": torch.zeros(h),
        "W_xr": rnd(f, h), "W_hr": rnd(h, h), "b_r": torch.zeros(h),
    }


def as_tuple(p):
    return tuple(p[k] for k in PARAM_ORDER)


def run_do_gru(inputs, p, h0):
    """用受控权重驱动 do_gru 的递推逻辑（按 PARAM_ORDER 组成元组传进去）。

    ★ 必须显式传 `low_accuracy=False`：do_gru 的默认是 True（多项式近似），
      而本文件里那些"手算期望值"的用例是拿 torch.sigmoid / torch.tanh 当参考的，
      不切到高精度会比出 ~5e-05 的假差异（正好是多项式逼近误差的量级）。
    """
    return gru.do_gru(inputs, as_tuple(p), h0, low_accuracy=False)


# ===================== 1. shape（doc §15） =====================

def test_do_gru_returns_one_entry_per_time_step_with_batch_by_hidden_shape():
    inputs, h0 = make_inputs(), make_hidden()
    states, resets, updates, candidates = run_do_gru(inputs, base_params(), h0)

    assert len(states) == len(resets) == len(updates) == len(candidates) == T
    for t in range(T):
        for seq in (states, resets, updates, candidates):
            assert seq[t].shape == (B, H), "第 %d 步 shape 应为 (B,H)=%s" % (t, (B, H))
        assert states[t].dtype == torch.float32


def test_get_gru_params_shapes_match_doc_table():
    p = gru.get_gru_params(F, H, 3)
    assert len(p) == 9, "应返回 9 个参数（h/z/r 各 3 个）"

    W_xh, W_hh, b_h, W_xz, W_hz, b_z, W_xr, W_hr, b_r = p
    assert W_xh.shape == (F, H) and W_hh.shape == (H, H) and b_h.shape == (H,)
    assert W_xz.shape == (F, H) and W_hz.shape == (H, H) and b_z.shape == (H,)
    assert W_xr.shape == (F, H) and W_hr.shape == (H, H) and b_r.shape == (H,)


def test_get_gru_params_returns_documented_order():
    """位置不能串：W_x* 是 [F,H]，W_h* 是 [H,H]，b_* 是 [H]。"""
    p = gru.get_gru_params(F, H, 1)
    for i, name in enumerate(PARAM_ORDER):
        if name.startswith("W_x"):
            assert p[i].shape == (F, H), "第 %d 个位置应是 %s([F,H])" % (i, name)
        elif name.startswith("W_h"):
            assert p[i].shape == (H, H), "第 %d 个位置应是 %s([H,H])" % (i, name)
        else:
            assert p[i].shape == (H,), "第 %d 个位置应是 %s([H])" % (i, name)


# ===================== 2. 数值范围（doc §12） =====================

def test_gates_and_candidate_stay_within_their_ranges():
    """doc §12：sigmoid ∈ (0,1)、tanh ∈ (-1,1)。

    注意一个 float32 的细节：pre-activation 一大，sigmoid 会被舍入到**精确的 1.0**
    （1 - 2e-22 在 float32 下就是 1.0），tanh 同理。所以"严格开区间"只在中等幅度下检查，
    大输入只用闭区间兜底。
    """
    # 中等幅度：不饱和，应严格落在开区间
    inputs, h0 = make_inputs(scale=0.5), make_hidden()
    _, resets, updates, candidates = run_do_gru(inputs, base_params(scale=0.5), h0)

    for t in range(T):
        assert resets[t].min() > 0 and resets[t].max() < 1, \
            "reset gate 应严格落在 (0,1)，第 %d 步越界" % t
        assert updates[t].min() > 0 and updates[t].max() < 1, \
            "update gate 应严格落在 (0,1)，第 %d 步越界" % t
        assert candidates[t].min() > -1 and candidates[t].max() < 1, \
            "candidate 应严格落在 (-1,1)，第 %d 步越界" % t

    # 大输入：允许饱和到端点，但绝不能越界
    big_inputs, big_h0 = make_inputs(scale=5.0), make_hidden() * 5
    _, resets, updates, candidates = run_do_gru(
        big_inputs, base_params(scale=2.0), big_h0)

    for t in range(T):
        assert torch.all(resets[t] >= 0) and torch.all(resets[t] <= 1), \
            "reset gate 越界，第 %d 步" % t
        assert torch.all(updates[t] >= 0) and torch.all(updates[t] <= 1), \
            "update gate 越界，第 %d 步" % t
        assert torch.all(candidates[t] >= -1) and torch.all(candidates[t] <= 1), \
            "candidate 越界，第 %d 步" % t


# ===================== 3. update gate 的两个极端 =====================

def test_update_gate_saturated_to_one_copies_previous_hidden():
    """Z=1  =>  H_t = H_{t-1}（candidate 完全不参与）。"""
    p = base_params()
    p["b_z"] = torch.full((H,), BIG)          # z == 1.0

    inputs, h0 = make_inputs(), make_hidden()
    states, _, _, _ = run_do_gru(inputs, p, h0)

    h_prev = h0
    for t in range(T):
        diff = (states[t] - h_prev).abs().max().item()
        assert diff < SAT_ATOL, \
            "第 %d 步：Z=1 时 H 应等于 H_{t-1}，实测 max|Δ|=%.3e" % (t, diff)
        h_prev = states[t]


def test_update_gate_saturated_to_one_ignores_input_and_candidate():
    """Z=1 时换掉输入和 candidate 相关权重，输出应纹丝不动。"""
    inputs, h0 = make_inputs(), make_hidden()

    p1 = base_params()
    p1["b_z"] = torch.full((H,), BIG)
    s1, _, _, _ = run_do_gru(inputs, p1, h0)

    p2 = base_params(seed=99)                 # W_xh / W_hh / b_h 全不同
    p2["b_z"] = torch.full((H,), BIG)
    s2, _, _, _ = run_do_gru(inputs * 10.0, p2, h0)   # 输入也放大 10 倍

    for t in range(T):
        assert torch.allclose(s1[t], s2[t], atol=SAT_ATOL), \
            "Z=1 时输出不应受输入 / candidate 影响，第 %d 步不一致" % t


def test_update_gate_saturated_to_zero_returns_candidate():
    """Z=0  =>  H_t = candidate_t。"""
    p = base_params()
    p["b_z"] = torch.full((H,), -BIG)         # z ≈ 0

    inputs, h0 = make_inputs(), make_hidden()
    states, _, _, candidates = run_do_gru(inputs, p, h0)

    for t in range(T):
        diff = (states[t] - candidates[t]).abs().max().item()
        assert diff < SAT_ATOL, \
            "第 %d 步：Z=0 时 H 应等于 candidate，实测 max|Δ|=%.3e" % (t, diff)


# ===================== 4. reset gate 的两个极端 =====================

def test_reset_gate_saturated_to_zero_makes_candidate_independent_of_history():
    """R=0  =>  candidate 不再依赖 H_{t-1}。

    把 W_hr / W_hz 也置零，切断 r / z 本身对 hidden 的依赖，
    这样唯一可能引入历史信息的通路就只剩 candidate 里的 reset 项。
    """
    p = base_params()
    p["b_r"] = torch.full((H,), -BIG)
    p["W_hr"] = torch.zeros(H, H)
    p["W_hz"] = torch.zeros(H, H)

    inputs = make_inputs()
    _, _, _, cand_a = run_do_gru(inputs, p, make_hidden(seed=1))
    _, _, _, cand_b = run_do_gru(inputs, p, make_hidden(seed=2))

    for t in range(T):
        assert torch.allclose(cand_a[t], cand_b[t], atol=SAT_ATOL), \
            "R=0 时 candidate 不应依赖 H_{t-1}，第 %d 步不一致" % t


def test_reset_gate_saturated_to_one_leaves_history_fully_in_candidate():
    """R=1 且 Z=0  =>  candidate 里的历史分量完整保留（等价于普通 RNN 的递推项）。"""
    p = base_params()
    p["b_r"] = torch.full((H,), BIG)          # r == 1.0
    p["b_z"] = torch.full((H,), -BIG)         # z ≈ 0，H 直接等于 candidate

    inputs, h0 = make_inputs(), make_hidden()
    states, _, _, _ = run_do_gru(inputs, p, h0)

    expected = torch.tanh(inputs[0] @ p["W_xh"] + h0 @ p["W_hh"] + p["b_h"])
    assert torch.allclose(states[0], expected, atol=F32_ATOL), \
        "R=1,Z=0 时第 0 步应等于 tanh(X@W_xh + H_0@W_hh + b_h)，max|Δ|=%.3e" \
        % (states[0] - expected).abs().max()


# ===================== 5. 退化情形（doc 214-225） =====================

def test_z0_r0_degenerates_to_mlp():
    """Z=0 且 R=0  =>  H_t = tanh(X_t@W_xh + b_h)，无任何递归项（MLP）。"""
    p = base_params()
    p["b_r"] = torch.full((H,), -BIG)
    p["b_z"] = torch.full((H,), -BIG)

    inputs, h0 = make_inputs(), make_hidden()
    states, _, _, _ = run_do_gru(inputs, p, h0)

    for t in range(T):
        expected = torch.tanh(inputs[t] @ p["W_xh"] + p["b_h"])
        assert torch.allclose(states[t], expected, atol=F32_ATOL), \
            "Z=0,R=0 时第 %d 步应退化为 MLP，max|Δ|=%.3e" \
            % (t, (states[t] - expected).abs().max())

    # 换一个完全不同的 H_{t-1}，结果必须一样 —— 这才叫"忽略历史"
    states_b, _, _, _ = run_do_gru(inputs, p, make_hidden(seed=2))
    for t in range(T):
        assert torch.allclose(states[t], states_b[t], atol=SAT_ATOL)


def test_z0_r1_degenerates_to_vanilla_rnn():
    """Z=0 且 R=1  =>  H_t = tanh(X_t@W_xh + H_{t-1}@W_hh + b_h)，即普通 RNN 递推式。"""
    p = base_params()
    p["b_r"] = torch.full((H,), BIG)
    p["b_z"] = torch.full((H,), -BIG)

    inputs, h0 = make_inputs(), make_hidden()
    states, _, _, _ = run_do_gru(inputs, p, h0)

    h_prev = h0
    for t in range(T):
        expected = torch.tanh(inputs[t] @ p["W_xh"] + h_prev @ p["W_hh"] + p["b_h"])
        assert torch.allclose(states[t], expected, atol=F32_ATOL), \
            "Z=0,R=1 时第 %d 步应退化为 RNN，max|Δ|=%.3e" \
            % (t, (states[t] - expected).abs().max())
        h_prev = states[t]


def test_z1_result_is_identity_regardless_of_reset_gate():
    """Z=1 时 H 恒等于 H_0，与 R 无关。

    所以 doc 里 "Z=1，R=1 就是 RNN" 这句不成立：
    Z=1 是恒等复制（R 失效），真正退化成普通 RNN 的是 **Z=0 且 R=1**。
    """
    inputs, h0 = make_inputs(), make_hidden()

    p_r0 = base_params()
    p_r0["b_z"] = torch.full((H,), BIG)
    p_r0["b_r"] = torch.full((H,), -BIG)

    p_r1 = base_params()
    p_r1["b_z"] = torch.full((H,), BIG)
    p_r1["b_r"] = torch.full((H,), BIG)

    s_r0, _, _, _ = run_do_gru(inputs, p_r0, h0)
    s_r1, _, _, _ = run_do_gru(inputs, p_r1, h0)

    for t in range(T):
        assert torch.allclose(s_r0[t], s_r1[t], atol=SAT_ATOL), \
            "Z=1 时 R 不应有任何影响，第 %d 步不一致" % t
        assert torch.allclose(s_r0[t], h0, atol=SAT_ATOL), \
            "Z=1 时每一步都应等于 H_0，第 %d 步不一致" % t


# ===================== 6. 时间步之间 state 要传下去 =====================

def test_hidden_state_is_carried_between_time_steps():
    """第 t 步的结果必须真的喂给第 t+1 步，而不是每步都从 H_0 重新开始。"""
    p = base_params()
    inputs, h0 = make_inputs(), make_hidden()
    states, _, _, _ = run_do_gru(inputs, p, h0)

    hidden = h0
    for t in range(T):
        r = torch.sigmoid(inputs[t] @ p["W_xr"] + hidden @ p["W_hr"] + p["b_r"])
        z = torch.sigmoid(inputs[t] @ p["W_xz"] + hidden @ p["W_hz"] + p["b_z"])
        cand = torch.tanh(inputs[t] @ p["W_xh"] + (r * hidden) @ p["W_hh"] + p["b_h"])
        expected = z * hidden + (1 - z) * cand
        assert torch.allclose(states[t], expected, atol=F32_ATOL), \
            "第 %d 步递推不符，max|Δ|=%.3e" % (t, (states[t] - expected).abs().max())
        hidden = expected


# ===================== 7. 候选状态里 reset 门的位置（对齐 src/nnet.c） =====================

@pytest.mark.xfail(strict=False, reason="do_gru 是教材版：candidate 写成 (r ⊙ h) @ W_hh，"
                                        "与 RNNoise 的 r ⊙ (W_hh·h + b_hh) 结构性不等价（文件头③）")
def test_candidate_applies_reset_to_the_recurrent_projection():
    """RNNoise 的 compute_generic_gru（src/nnet.c:65-94）把重置门乘在 **recurrent 投影之后**：

        recur = W_rec @ state + b_rec        # 先投影
        h_cand = tanh( (W_in x + b_in) + r ⊙ recur )

    PyTorch 的 nn.GRUCell 是同一套结构（已实测：这样能复现到 1.8e-07）。
    而教材常见变体是  (r ⊙ h) W_rec + b —— 两者数学上不等价。

    ★ 本文件测的 `do_gru()` 恰好是**教材变体**，所以这条必然不满足（标 xfail）。
      真正实现 RNNoise 结构的是 `GRUMo.forward()`（examples/gru.py），它由
      `uv run scripts/unit_check.py gru` 对着 C 逐项验证（272 项之一）。
    """
    p = base_params()
    inputs, h0 = make_inputs(), make_hidden()
    _, _, _, candidates = run_do_gru(inputs, p, h0)

    # 用同一组参数，按 RNNoise 的结构算第 0 步的期望值（单步即可判定）
    r0 = torch.sigmoid(inputs[0] @ p["W_xr"] + h0 @ p["W_hr"] + p["b_r"])
    expected = torch.tanh(inputs[0] @ p["W_xh"] + r0 * (h0 @ p["W_hh"]) + p["b_h"])
    d = (candidates[0] - expected).abs().max().item()
    assert d < F32_ATOL, (
        "第 0 步 candidate 与 RNNoise 的结构不符（max|Δ|=%.3e）。\n"
        "RNNoise:   tanh( W_in x + b_in + r ⊙ (W_rec h + b_rec) )\n"
        "当前实现:  tanh( W_xh x + (r ⊙ h) W_hh + b_h )  <- reset 乘在了投影之前" % d)


# ===================== 9. GRUMo：真正接进整网的那个包装 =====================
#
# 旧版这里测的是 `rnn_unit.GRU`，那个类已经不存在了。现在整网用的是
# `GRUMo`（RNNoise 的 [z|r|h] 布局、一次吃一帧，隐状态由**调用方持有**：
# `h_out = net(x, h_in)`），它和 `do_gru` 不是一回事：公式正确性由
# `uv run scripts/unit_check.py gru` 对着 C 的 compute_generic_gru() 验证，
# 这里只测"状态有没有按要求进出、且不偷偷留在 self 里"。

def _synthetic_gru_case(f=F, h=H, seed=0):
    """造一份最小 case（[z|r|h] 布局：input/recurrent 各有 weights + bias）。

    GRUMo 现在**无条件**读 case（`case_tensor(case, ...)`），`case=None` 会崩；
    这里给一份形状正确、数值非退化的假用例 —— 测试就不必依赖
    `scripts/unit_cases/gru.json`，也能继续用 F=4 / H=5 这种小规模。
    """
    n = h * 3
    gen = torch.Generator().manual_seed(seed)
    def lst(count):
        return (torch.randn(count, generator=gen) * 0.1).tolist()
    return {
        "input_weights_float": lst(f * n),
        "input_bias": lst(n),
        "recurrent_weights_float": lst(h * n),
        "recurrent_bias": lst(n),
    }


def test_gruscratch_carries_hidden_state_between_calls():
    torch.manual_seed(0)
    case = _synthetic_gru_case()
    # low_accuracy 是**构造函数**参数（forward 只剩 (x, hidden)）。
    # 注意第三项是 output_features_size = 3*H（[z|r|h] 三个门拼起来），**不是** F*3：
    # 旧实现自己按 hidden_size*3 造随机权重，所以写成 F*3 也能跑；现在权重来自 case，
    # 形状必须自洽，否则 recur 的切片和 r 对不上（size 2 vs 5）。
    net = gru.GRUMo(F, H, H * 3, case, low_accuracy=False)

    h0 = torch.zeros(H, dtype=torch.float32)           # 初始隐状态 = 0（C 侧同样 memset）
    x = torch.randn(F, dtype=torch.float32)

    h1 = net(x, h0)                                    # 状态从参数进、从返回值出
    assert h1.shape == (H,)
    assert h1.dtype == torch.float32
    assert not torch.allclose(h1, h0), "喂了输入就不该原地不动"

    h2 = net(x, h1)
    assert not torch.allclose(h2, h1), "第二帧必须拿第一帧的结果当历史"

    # ★ 无隐藏状态：同样的 (x, h_in) 必须得到同样的 h_out。
    #   这条是为了防止有人把状态又塞回 self.hidden（那样 ONNX 会每次从 0 开始）。
    assert torch.allclose(net(x, h0), h1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

# C代码函数要回答 7个问题
1. 这个 C 函数做了什么？
2. 它对应什么数学运算？
3. 它属于哪个抽象层？
4. 它在 RNNoise 的哪一层/哪条计算路径中被调用？
5. 为什么这条计算路径需要它？
6. 我能否用最简单的 Python/PyTorch 写出等价计算？
7. C / Python 输出不一样时，差异来自哪里？

## `compute_linear()`
**线性代数计算 primitive/ kernel**  这对你以后读 vLLM、TensorRT、CUDA kernel 非常重要。

1. c语言：判断层的权重weights、偏移bias 使用 float类型还是int类型，float 优先。根据 idx 值判断是否在给定的weights、bias 数据中做数据偏移。输入数据 和 权重值 做矩阵乘法。尽可能的使用并行计算，按照16-8-4-2-1的分段，如果输入数据超过满足16个元素，一次计算16个，不满足16个，一次处理8个，以此类推。 输入数据与权重做矩阵乘法后，结果再和 bias 相加。如果有diag
2. 权重矩阵是 nb_input x nb_output, 数学上是权重矩阵和输入向量做矩阵乘法， 输入向量的长度与权重 行数必须相同，bias向量的长度与权重矩阵列数相同，最中输出 shape 是长度为 nb_output 的向量
3. 计算1D卷积中使用，input 首先经过这一层，output 结果进入激活函数，这是一个全连接层，把输入的全部数据与权重建立映射关系，也就是把输入的数据和模型参数产生了联系

## `compute_activation`
**它是一个通用 activation primitive**

net_arch.h:79 
1. 根据 activation 使用不同的函数， 根据根据宏定义开启普通精度和高精度
2. 低精度 vec_tanh(out,in, nb_output)
3. 高精度 math 库中的实现
4. tanh 函数三个核心特性
 - 关于远点对称
 - 值域在(-1,1)之间，随着x的减小无限接近-1，随着x的增大无限接近1
 - 均值是0 ❌， 这里不应该这么表达。 `tanh(0) = 0, -tanh(x) = tanh(-x)`

在convd1中使用，把输出结果收敛-1，1之间

所以 
```
compute_linear  -> 得到 pre-activation
compute_activation -> 把 pre-activation 变成 activation
```

## `compute_generic_conv1d` 
- C代码：如果输入数据size不满足层的权重nb_input（行）无法做矩阵乘法，所以把层的历史状态 state 数据先放到最前面，后面拼接输入的数据 `[state][input]`，但这里state的数据可能是0

conv1d_1 权重 195x128， 输入长度 65
对于rnn.c:48 来说，195！=65，就把mem中的 130 个元素，也就是之前的两次数据先拷贝到 tmp中，再把真正的input 补齐, mem中永远保持最近的两个input数据: mem[1][2][3]-> tmp[mem[2]|mem[3]|input]

💡 因为带有之前的2次记忆，所以是Conv1d 卷积 卷的关键！！！！
相当于 kernel = 3 的 1D convolution

对于rnn.c:48 来说，384！=128，同上，保存之前的两次数据

- ❌ 对输入数据做线性回归❌ 对输入数据做**线性变换 / affine transformation**，通过 tanh激活函数， 把输出结果收敛到 -1，1 之间

线性变换≠线性回归！！！！ y=Wx+b 即 **线性变换+bias**，线性回归是一个神经网络任务/模型的名称。不是一种`处理`

- 把input的数据拷贝到 mem中，也就是当前 conv1d 从第二个输入数据开始，mem就保存了上一次的输入数据


conv1d_2 权重 384x384， 输入长度 128



## RNN 学习

8.4 RNN 章节已经学习完， 记住两个关键的公式

输入：X， shape(batch_size, nb_features) batch_size 处理批次， nb_features， 输出数据 X的features数
隐状态 H = X@W_xh + b_h + W_hh@H_last, 隐状态形状 n x h, 因为 X形状 nxd, W_xh形状 dxh， X@W_xh 结果 H形状 nxh
W_xh.shape: (nb_features, num_hiddens), num_hiddens 是超参数，可调节
b_h： bias
W_hh.shape: (num_hiddens, num_hiddens)
H_last ： 时间序列上一次的H值
输出 O = H@W_hq + b_q
W_hq.shape: (num_hiddens, nb_outputs) , nb_output 输出 features. 
b_q: 输出层的 bias

hidden_state.shape: 输入X的属性数作为行，列是一个超参数，可调节。

同时也认识了 one hot 编码， 了解了剪裁梯度。
RNN和MLP 不一样的点在于 RNN中的 隐状态，记录了之前输入数据的历史状态，当前权重计算需要依赖于历史数据。MLP每一层的输入相对独立。

RNN Scratch 手撕实现，我一边学习 8.5 节，一边完成了手写代码，没有直接使用 torch.RNN，完全自己实现

RNN 完全按照上面的公式 自己实现
在自己实现的过程中， time_step 没有体现。 在读取训练数据时使用了 batch_size 和 time_step，这两个值是否需要相关？ batch_size 是 输入数据 X 的 "行"。

需要有哪些注意的关键点？哪些点是上面我回答的问题中没有提出，但是值得注意的部分？

time step, batch_size, 这两个参数的意义是什么？有些案例中会提到 input_size, sequence，又是怎么回事？

---

**遗留问题** ✅

1. inputs 三维矩阵，每一个维度分别代表了实际应用中的什么？举出几个使用RNN的真实场景，把inputs 的三维矩阵分别映射到真实场景中，把 time_step, sequence_length, batch_size, nb_features 这些参数一一对应。

### `time_step` 和 `sequence_length` 不是同一个东西
```text
sequence length = T = 总共有多少步
time step = t = 当前正在处理第几步

sequence_length = 100
t = 1
t = 2
...
t = 100
```
而 `time step` 通常指其中**某一个具体时刻**：X_37， 就是第 37 个 time step。

真实场景
rnnoise
股票预测
服务器参数预测
情感分析

2. 权重矩阵初始化使用随机数，而不是0填充，是必须的嘛？原因是什么？bias 使用0初始化
- 打破神经元对称性，防止参数同质化
- 如果存在多个权重参数，W_xh, W_hh, W_hq，如果都相同，只能得到相同的梯度
**让不同神经元从不同起点出发，打破对称性。**
bias 初始化可以是0，但也可以不是0，看情况而定。 因为W已经打破了对称性，所以bias 不强制。


## BPTT 梯度消失，梯度爆炸的原因

因为初始条件的微小变化就可能会对结果产生巨大的影响
W_hh.T 随着时间推移，W_hh大于1的，会越来越大-爆炸； 反之小于1的，会越来越小-消失；

不用在这个阶段深入推导，因为 GRU 为什么设计出来，和这个背景直接相关。D2L 在 GRU 前面就是从长期依赖、状态保留、梯度问题引出门控机制的。([D2L AI][2])

## GRU 手撕中的问题
1. 忘记激活函数
2. (r*hidden) 然后再和 W_hr做矩阵乘法
3. 输出层的权重，偏置 不应该属于 GRU 的参数
4. **缺少 hidden 的时间传播！！！**
5. 缺少stack 输出

TODO未完成
- 把激活函数使用前后做对比，更好的学习，记忆激活函数的作用 ✅
- 不用看资料画出 GRU 结构。✅
- 使用 nn.GRUCell 与自己的实现做对比 ✅

初始化的参数全部不同，这样的实验是否可靠，是否需要把参数保持全部一致？

这个输出是否合理？
max_abs_error: 0.813206672668457
mean_abs_error: 0.32638347148895264
max_abs_error: 1.023895025253296
mean_abs_error: 0.32801032066345215


## GRU 学习验收问题 我的回答，未修正

GRU我已经学完了，你要的验收 

candidate hidden 为什么不能直接成为新的 hidden state？
candidate 中只包含了历史数据的重要性与否（0-1）但并没有给出当前最近一次历史信息（H_t-1）和之前的历史信息（H_candidate）对于本次结果的重要关系，这个关系也需要通过0-1安排比例 

回答为什么需要 GRU ？ 
RNN 会随着时间序列的推移，导致梯度消失或者梯度爆炸的情况。 梯度计算会变得越来越缓慢/困难

流程图：结构图已经确认可以画出，但是我你是否有途径检查我的手绘或者其他方式的流程图？

公式：
重置门 R_t = sigmoid(X_t@W_xr + b_r) 忘记了 H_t-1
更新门 Z_t = sigmoid(X_t@W_xz + b_z) 忘记了 H_t-1
候选 H_cadicate = tanh(X_t@W_xh + R_t*(H_t-1) + b_h)
最终 H = Z_t*H_t-1 + (1-Z_t)*H_candidate

含义：
reset gate 重置门，范围(0-1)， 控制上一个隐状态对当前的影响大小，0 表示完全忽略上一次状态，从而达到“重置”效果。1 表示完全使用上一次的状态。
本轮输入与上一轮隐状态通过它得到一个候选隐状态

update gate 更新门，范围0-1，控制当前输入与上一次隐状态对本轮影响大小，1 表示忽略候选隐状态，完全采用上一轮的隐状态，0表示使用候选隐状态，忽略上一轮的隐状态。最终得到一个新的隐状态。

Tensor shape

X_t: time_step, batch_size, in_features

W_xr: in_features,hidden_size
R_t: batch_size, hidden_size
b_r: 1,hidden_size

W_xz: in_features,hidden_size
Z_t: batch_size, hidden_size
b_z: 1, hidden_size

W_xh: in_features, hidden_size
b_h: i, hidden_size

H_t-1, H_t, H_candidate: batch_size, hidden_size

W_hq = hidden_size, out_features
out: batch_size, out_features

features 输入数据的特征长度
time_step 时间序列
batch_size 一次处理的采样数量
hidden_size 隐状态大小

未解之谜

> GRU 通过 update gate 建立了一条可以较大程度保留旧 hidden state 的路径，使信息能够跨越更多时间步传播，从而缓解普通 RNN 的长期依赖和梯度问题。

💡 这里提到的**路径**就是 
Z = 1: **H_t-1 -> H_t**
Z = 0: **H_t-1->R->H_candidate->H_r**

我先列出公式：
重置门 R = sigmoid(X@W_xr + H_t-1@W_hr + b_r)
H 候选 H_candidate = tanh(X@W_xh + **R*(H_t-1)@W_hh** + b_h)
更新门 Z = sigmoid(X@W_xz + H_t-1@W_hz + b_z)
H = Z * H_t-1 + (1-Z) * H_candidate

RNN 
H = X@W_xh + H_t-1@W_hh + b_h


我的问题 ：

---

在RNN隐状态公式中
H = X@W_xh + H_t-1@W_hh + b_h
当前 H 保留了全部 H_t-1的信息，也就是RNN的问题，没有选择的把历史信息一直携带，从而导致梯度爆炸/梯度消失

GRU中，按照Z的两个极端路径举例
Z=0：H=H_candidate, 而 candidate 根据R的值，两个极端方向
    R=0：完全忽略历史信息
    R=1：本次信息+历史信息
Z=1: H=H_t-1, 也就是完全遗忘历史信息，直接截断
如果按照一种偏特化例子：Z=1，R=1 就是 RNN；Z=0，R=0 就是MLP
这样看起来，GRU与RNN相比，更多的是截断历史信息，过滤历史信息的功能。从而解决 RNN BPTT 存在的梯度消失/爆炸的现象。

---

使得**信息跨越更多的时间步** 
update gates： H = Z * H_t-1 + (1-Z) * H_candidate
比如 Z_1 = 0.99, 那么 H_1 ~ H_0, Z_2=0.99 H_2~H_1~H_0，通过链式把信息传递下去，但这里不是完全拷贝复制。这里说明 H_2 中保留了很多 H_0的内容，既**H3 对早期输入的信息仍然具有依赖关系。**

💡 R 控制“旧信息参与 candidate 的程度”
💡 Z 控制“旧 hidden 直接保留多少”

带入更新门公式，H_candidate 最多相差一个 (H_t-1@W_hr) 也就是上次的隐状态保存到本次的“程度”，如果本次带了比较多的(H_t-1@W_hr)， 那么下一轮 比如时间 T3， 这时候的H 就包括了T2，T1 的隐状态。

但这怎么缓解普通RNN的长期依赖和梯度问题？通过更新门，重置门 把之前的时间中的隐状态一直带入到下一次中，就能缓解吗？

第二个问题，我使用 nn.GRUCell 与自己的实现做对比
但是 nn.GRUCell 中没有W_xr, W_hr, W_xz, W_hz, bias_x, bias_h ，这与我自己实现的具有参数不一致
权重参数不一致，必然导致结果不同
GRUCell 对 W做了cat连接，我在学习阶段是否需要与其保持一致？
如果不一致，测试我实现的GRU和GRUCell 势必会有误差。
这是否阻碍我进入下一环节，读c代码

## 使用 nn.GRUCell 与自己的实现做对比， 解释误差出现的原因

## `compute_generic_gru()` 学习，自己手写实现 GRU，而不是使用 官方 GRUCell，这样可以保证权重参数，计算方式全部一致并且可控。

# GRU 我的总结与 chatgpt 评价

---

1. C代码干什么了
compute_generic_gru 实现 nnet.c:65-94 

- zrh: 看名称，是把 GRU 中的 重置门，更新门 和 隐状态放到了一个数组, 指针 z、r、h 地址间隔一个 
in 数据 一个 384 float 向量， shape(1,384)。 zrh 内存布局 [z 384|r 384|h 384], 384是 float，不是字节。

zrh (3,384) or (1,1152)

不能说 ❌` z.shape(384,384), r.shape(384,384), h.shape(384,384)` ❌ ，z，r，h 不是矩阵，是向量。

- nb_outputs = 1152 = 3*nb_inputs, 通过 `compute_linear(input_weights, zrh, in, arch);` 输入 384 得到 1152 的 out 数据 zrh，也就是 zrh[X@W_xz+bias|X@W_xr+bias|X@W_xh+bias] , 三个权重数据都在 input_weights 中。 
input_weights 中保存的是 隐状态计算 zrh 需要的权重 


- `compute_linear(recurrent_weights, recur, state, arch);` 实现 隐状态@W，分成了3份，可以这样布局 recur[H_t-1@W_hz+bias|H_t-1@W_hr+bias|H_t-1@W_hh+bias]，只不过这里的 W_hz/W_hr/W_hh 都放到了一个recurrents_weights 中，与上面的input_weights 一致
recurrent_weights 中保存的是 隐状态计算 zrh 需要的权重 

- line 84-86: zrh[ X@W_xz+H_t-1@W_hz | X@W_xr+H_t-1@W_hr | X@W_xh ], 第三部分 h 保持不变，z，r 对位相加并保存到zrh对应位置，使用激活函数sigmoid 处理 z，r 两部分

- line 87-89: recur的第三部分与r对位相乘，结果再与 h 对位相加, h部分使用tanh作为激活函数处理。此时 zrh [X@W_xz+H_t-1@W_hz|X@W_xr+H_t-1@W_hr| (H_t-1@W_hh)*(H_t-1@W_hr)+X@W_xh ]

- line 90，91：最终隐状态确定。z与历史state对位相乘，(1-z)与当前的h对位相乘。
- line 92，93：保存 h 到 state中，作为历史状态下一轮使用

这里是根据C代码得到读到的内容
```
Z_t = sigmoid(X@W_xz+H_t-1@W_hz)
R_t = sigmoid(X@W_xr+H_t-1@W_hr)
H_candidate = tanh((H_t-1@W_hh)*(H_t-1@W_hr)+X@W_xh)
H = z*(H_t-1) + (1-z)* H_candicate
```
这里没有提到 bias，因为 compute_linear() 中处理了 bias，为了减少冗余没有表达。
除此之外，与GRU的公式一致


以下是看代码后的推测，没有实践确认

conv1 
nb_input 195 , output 128, weight shape 195x128
input frame_t, shape 1x65
state 65x2=130 [frame_t-2|frame_t-1]

output 128 [frame_t-2@W + b |frame_t-1@W + b|frame_t@W+b]  ❌ 这里不等价
应该是 output 128 [(frame_t-2+frame_t-1+frame_t)@W + b] 先拼接，在矩阵乘法

conv2 
nb_inpupt 384, output 384, weights shape 384x384
input conv1_output_t
state[conv1_output_t-2|conv1_output_t-1]
output 384 [(conv1_output_t-2 + conv1_output_t-1 + conv1_output_t)@W+b] 

gru1
nb_inpupt 384, output 1152, weights shape 
输入: conv2_output_t
隐藏层: zrh[2xnb_input:]
输出: gru1_state

gru2 
nb_inpupt 384, output 1152, weights shape 
输入: gru1_state
输出: gru2_state

gru3 
nb_inpupt 384, output 1152, weights shape 
输入: gru2_state
输出: gru3_state


每一个GRU 
- 输入是什么？
- hidden state 是什么？
- 输出是什么？
- c变量 <-> 数学符号 <-> pytorch 变量，建立一张图表
x
h
z
r
h_candidate
- 分析 weight，bias， state 的 shape

必须写出
input_size
hidden_size
weight shape
bias shape
state shape
- 任何一个矩阵乘法，都能解释，矩阵乘法需要解释什么？不就是两个矩阵相乘吗？


dense_out: MLP+激活函数, 得到 gains
nb_input 1536, output 32, weight = 1536 x 32
input [conv2_out|gru1_state|gru2_state|gru3_state] 💡 注意，这里没有 conv1_out
output: float[32]

vad_dense: MLP+激活函数， 得到 vad
nb_input 1536, output 1
input [conv2_out|gru1_state|gru2_state|gru3_state] 💡 注意，这里没有 conv1_out
output float

---

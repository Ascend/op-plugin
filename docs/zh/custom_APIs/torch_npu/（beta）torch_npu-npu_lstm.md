# （beta）torch_npu.npu_lstm

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

计算LSTM（长短期记忆网络）前向传播。

## 函数原型

```python
torch_npu.npu_lstm(x, weight, bias, seqMask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction)
```

## 参数说明

- **x** (`Tensor`)：输入序列，逻辑形状为 (seq_length, batch_size, input_size)。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **weight** (`Tensor`)：拼接后的输入权重和循环权重，逻辑形状为 (input_size + hidden_size, 4 × hidden_size)。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **bias** (`Tensor`)：1D张量。数据类型支持`torch.float16`、`torch.float32`；格式支持ND。
- **seqMask** (`Tensor`)：张量。仅支持FRACTAL_NZ格式的`torch.float16`和ND格式的`torch.int32`类型。
- **h** (`Tensor`)：初始隐藏状态，单层单向场景的逻辑形状为 (1, batch_size, hidden_size)。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **c** (`Tensor`)：初始细胞状态，单层单向场景的逻辑形状为 (1, batch_size, hidden_size)。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **has_biases** (`bool`)：如果值为True，则存在偏置。
- **num_layers** (`int`)：循环层数，目前只支持单层。
- **dropout** (`float`)：丢弃概率，当前不支持，传入`0.0`。
- **train** (`bool`)：是否进入训练模式，默认值为`True`。
- **bidirectional** (`bool`)：如果值为True，LSTM为双向。当前不支持。
- **batch_first** (`bool`)：如果值为True，则输入和输出张量将表示为(batch, seq, feature)。当前不支持。
- **flag_seq** (`bool`)：如果值为True，输入为PackedSequence。当前不支持。
- **direction** (`bool`)：如果值为True，则方向为“BIDIRECTIONAL”，否则为“UNIDIRECTIONAL”。

## 输出说明

- **y** (`Tensor`)：4D张量。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **output_h** (`Tensor`)：4D张量。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **output_c** (`Tensor`)：4D张量。数据类型支持`torch.float16`、`torch.float32`；格式支持FRACTAL_NZ。
- **i** (`Tensor`)：输入门。数据类型支持`torch.float16`、`torch.float32`。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- **j** (`Tensor`)：候选细胞状态。数据类型支持`torch.float16`、`torch.float32`。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- **f** (`Tensor`)：遗忘门。数据类型支持`torch.float16`、`torch.float32`。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- **o** (`Tensor`)：输出门。数据类型支持`torch.float16`、`torch.float32`。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- **tanhct** (`Tensor`)：细胞状态经tanh变换后的值。数据类型支持`torch.float16`、`torch.float32`。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。

## 调用示例

示例计算单层LSTM的前向输出。输入张量依次为序列长度、批大小和输入特征数，并为需要的张量指定NPU布局。示例将两组ND权重沿第0维拼接成一个二维张量。返回值包含各时间步的输出、状态与门控中间结果。

```python
import torch
import torch_npu
format_nz = 29  # FRACTAL_NZ
format_nd = 2   # ND
input_size, hidden_size, batch_size, seq_length = 10, 5, 3, 5
x = torch_npu.npu_format_cast(torch.rand(seq_length, batch_size, input_size).npu(), format_nz).half()
weight_ih = torch_npu.npu_format_cast(torch.rand(input_size, 4 * hidden_size).npu(), format_nd).half()
weight_hh = torch_npu.npu_format_cast(torch.rand(hidden_size, 4 * hidden_size).npu(), format_nd).half()
weight = torch.cat((weight_ih, weight_hh), 0)
bias = torch_npu.npu_format_cast(torch.rand(4 * hidden_size).npu(), format_nd).half()
seq_mask = torch.tensor(seq_length, dtype=torch.int32).npu()
h = torch_npu.npu_format_cast(torch.rand(1, batch_size, hidden_size).npu(), format_nz).half()
c = torch_npu.npu_format_cast(torch.rand(1, batch_size, hidden_size).npu(), format_nz).half()
y, output_h, output_c, i, j, f, o, tanhct = torch_npu.npu_lstm(
    x, weight, bias, seq_mask, h, c, True, 1, 0.0, True, False, False, False, False)
print(y.shape, output_h.shape, output_c.shape)
```

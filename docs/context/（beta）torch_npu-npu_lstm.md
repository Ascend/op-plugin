# （beta）torch_npu.npu_lstm

## 函数原型

```
torch_npu.npu_lstm(x, weight, bias, seqMask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction)
```

## 功能说明

计算DynamicRNN。

## 参数说明

- x (Tensor) - 4D张量。数据类型：float16，float32；格式：FRACTAL_NZ。
- weight (Tensor) - 4D张量。数据类型：float16，float32；格式：FRACTAL_NZ_LSTM。
- bias (Tensor) - 1D张量。数据类型：float16，float32；格式：ND。
- seqMask (Tensor) - 张量。仅支持为FRACTAL_NZ格式的float16和ND格式的int32类型。
- h (Tensor) -  4D张量。数据类型：float16，float32；格式：FRACTAL_NZ。
- c (Tensor) -  4D张量。数据类型：float16，float32；格式：FRACTAL_NZ。
- has_biases (Bool) - 如果值为True，则存在偏差。
- num_layers (Int) - 循环层数，目前只支持单层。
- dropout (Float) - 如果值为非零，则在除去最后一层以外的每个LSTM层的输出上引入一个dropout层，丢弃概率等于dropout参数值。目前不支持。
- train (Bool，默认值为True) - 标识训练是否在op进行的bool参数。
- bidirectional (Bool) - 如果值为True，LSTM为双向。当前不支持。
- batch_first (Bool) - 如果值为True，则输入和输出张量将表示为(batch, seq, feature)。当前不支持。
- flag_seq (Bool) - 如果值为True，输入为PackedSequence。当前不支持。
- direction (Bool) - 如果值为True，则方向为“REDIRECTIONAL”，否则为“UNIDIRECTIONAL”。

## 输出说明

- y (Tensor) - 4D张量。数据类型：float16，float32；格式：FRACTAL_NZ。
- output_h (Tensor) - 4D张量。数据类型：float16，float32；格式：FRACTAL_NZ。
- output_c (Tensor) - 4D张量。数据类型：float16，float32；格式：FRACTAL_NZ。
- i (Tensor) - 4D张量。数据类型：float16，float32。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- j (Tensor) - 4D张量。数据类型：float16，float32。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- f (Tensor) - 4D张量。数据类型：float16，float32。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- o (Tensor) - 4D张量。数据类型：float16，float32。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。
- tanhct (Tensor) - 4D张量。数据类型：float16，float32。当train=True（训练模式）时，格式为FRACTAL_NZ；当train=False（推理模式）时，格式为ND。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


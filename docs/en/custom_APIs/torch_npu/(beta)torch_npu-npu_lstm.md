# (beta) torch_npu.npu_lstm

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Computes DynamicRNN.

## Prototype

```python
torch_npu.npu_lstm(x, weight, bias, seqMask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction)
```

## Parameters

- **`x`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`weight`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`bias`** (`Tensor`): This parameter must be a 1D tensor. The data type can be `float16` or `float32`. The data layout can be ND.
- **`seqMask`** (`Tensor`): Only the `float16` data type with the `FRACTAL_NZ` data layout and the `int32` data type with the ND data layout are supported.
- **`h`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`c`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`has_biases`** (`bool`): If set to `True`, biases exist.
- **`num_layers`** (`int`): Number of recurrent layers. Currently, only a single layer is supported.
- **`dropout`** (`float`): If the value is non-zero, a dropout layer is introduced on the output of each LSTM layer except the last one, and the dropout discard probability is equal to the value of the `dropout` parameter. Currently, this parameter is not supported.
- **`train`** (`bool`): Optional. Specifies whether training is executed inside the operator. The default value is `True`.
- **`bidirectional`** (`bool`): If set to `True`, the LSTM is bidirectional. Currently, this parameter is not supported.
- **`batch_first`** (`bool`): If set to `True`, the input and output tensors are represented as `(batch, seq, feature)`. Currently, this parameter is not supported.
- **`flag_seq`** (`bool`): If set to `True`, the input is a `PackedSequence`. Currently, this parameter is not supported.
- **`direction`** (`bool`): If set to `True`, the direction mode is `"REDIRECTIONAL"`. Otherwise, the direction mode is `"UNIDIRECTIONAL"`.

## Output Description

- **`y`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`output_h`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`output_c`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`i`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. When `train=True` (training mode), the data layout can be `FRACTAL_NZ`. When `train=False` (inference mode), the data layout can be ND.
- **`j`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. When `train=True` (training mode), the data layout can be `FRACTAL_NZ`. When `train=False` (inference mode), the data layout can be ND.
- **`f`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. When `train=True` (training mode), the data layout can be `FRACTAL_NZ`. When `train=False` (inference mode), the data layout can be ND.
- **`o`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. When `train=True` (training mode), the data layout can be `FRACTAL_NZ`. When `train=False` (inference mode), the data layout can be ND.
- **`tanhct`** (`Tensor`): This parameter must be a 4D tensor. The data type can be `float16` or `float32`. When `train=True` (training mode), the data layout can be `FRACTAL_NZ`. When `train=False` (inference mode), the data layout can be ND.

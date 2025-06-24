# （beta）torch_npu.npu_fused_attention_score

## 函数原型

```
torch_npu.npu_fused_attention_score(Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor attention_mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool bmm_score_transpose_a=False, bool bmm_score_transpose_b=False, bool value_transpose=False, bool dx_transpose=False) -> Tensor
```

## 功能说明

实现“Transformer attention score”的融合计算逻辑，主要将matmul、transpose、add、softmax、dropout、batchmatmul、permute等计算进行了融合。

## 参数说明

- query_layer：Tensor类型，仅支持float16。
- key_layer：Tensor类型，仅支持float16。
- value_layer：Tensor类型，仅支持float16。
- attention_mask：Tensor类型，仅支持float16。
- scale：缩放系数，浮点数标量。
- keep_prob：不做dropout的概率，0-1之间，浮点数。
- query_transpose：query是否做转置，bool类型，默认为False。
- key_transpose：key是否做转置，bool类型，默认为False。
- bmm_score_transpose_a：bmm计算中左矩阵是否做转置，bool类型，默认为False。
- bmm_score_transpose_b：bmm计算中右矩阵是否做转置，bool类型，默认为False。
- value_transpose：value是否做转置，bool类型，默认为False。
- dx_transpose：反向计算时dx是否做转置，bool类型，默认为False。

## 约束说明

输入tensor的格式编号必须均为29，数据类型为FP16。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> query_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> key_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> value_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> attention_mask = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 512).npu(), 29).half()
>>> scale = 0.125
>>> keep_prob = 0.5
>>> context_layer = torch_npu.npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
>>> print(context_layer)
tensor([[0.5010, 0.4709, 0.4841,  ..., 0.4321, 0.4448, 0.4834],
        [0.5107, 0.5049, 0.5239,  ..., 0.4436, 0.4375, 0.4651],
        [0.5308, 0.4944, 0.5005,  ..., 0.5010, 0.5103, 0.5303],
        ...,
        [0.5142, 0.5068, 0.5176,  ..., 0.5498, 0.4868, 0.4805],
        [0.4941, 0.4731, 0.4863,  ..., 0.5161, 0.5239, 0.5190],
        [0.5459, 0.5107, 0.5415,  ..., 0.4641, 0.4688, 0.4531]],
       device='npu:0', dtype=torch.float16)
```


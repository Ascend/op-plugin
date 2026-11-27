# （beta）torch_npu.contrib.npu_fused_attention

> [!NOTICE]  
> 该接口已废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议组合使用torch_npu.npu_linear和torch_npu.npu_fusion_attention。

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

BERT自注意力的融合实现。

## 函数原型

```python
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## 参数说明

- **hidden_states** (`Tensor`)：必选参数，最后一层的hidden_states。数据格式支持FRACTAL_NZ。shape为2维，其中第0维需为32的整数倍，第1维支持768或1024。
- **attention_mask** (`Tensor`)：必选参数，掩码张量，用于屏蔽不需要参与注意力计算的token位置（如padding位置）。数据格式支持FRACTAL_NZ。shape为4维，形状为$(B, 1, S, S)$，其中B为batch size，S为序列长度。
- **query_kernel** (`Tensor`)：必选参数，query的权重矩阵。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **key_kernel** (`Tensor`)：必选参数，key的权重矩阵。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **value_kernel** (`Tensor`)：必选参数，value的权重矩阵。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **query_bias** (`Tensor`)：必选参数，query的偏置值。数据格式支持ND。
- **key_bias** (`Tensor`)：必选参数，key的偏置值。数据格式支持ND。
- **value_bias** (`Tensor`)：必选参数，value的偏置值。数据格式支持ND。
- **scale** (`float`)：可选参数，应用于注意力分数的缩放系数，默认值为`1`。算子以`float16`传入该系数。缩放点积注意力通常取`1 / sqrt(head_dim)`，其中`head_dim`为注意力头的维度。
- **keep_prob** (`float`)：表示dropout中保留元素的概率，等于`1 - drop_rate`。默认值为`0`。

## 返回值说明

`Tensor`

自注意力的结果。

## 调用示例

下例构造长度为32的一个序列、768维隐藏状态、全零掩码和Q/K/V投影所需的张量。

```python
import torch
import torch_npu

torch.npu.config.allow_internal_format = True
batch, seq_len, hidden_size = 1, 32, 768
options = {"device": "npu", "dtype": torch.float16}

hidden = torch_npu.npu_format_cast(
    torch.randn(batch * seq_len, hidden_size, **options), 29
)
mask = torch_npu.npu_format_cast(
    torch.zeros(batch, 1, seq_len, seq_len, **options), 29
)
weight = torch_npu.npu_format_cast(torch.eye(hidden_size, **options), 29)
bias = torch.zeros(hidden_size, **options)
```

调用时，Q/K/V分别传入对应的权重和偏置。上述示例可以将`weight`和`bias`各传入三次。头维是64，因此缩放系数可传`0.125`。无需dropout时，最后一个位置参数传`1.0`。

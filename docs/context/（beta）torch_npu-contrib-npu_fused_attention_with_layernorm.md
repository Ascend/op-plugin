# （beta）torch_npu.contrib.npu_fused_attention_with_layernorm

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch_npu.npu_fusion_attention`与`torch.nn.LayerNorm`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

bert自注意力与层归一化的融合实现。

## 函数原型

```
torch_npu.contrib.npu_fused_attention_with_layernorm(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, gamma, beta, scale=1, keep_prob=0)
```

## 参数说明

- **hidden_states**（`Tensor`）：最后一层的hidden_states。
- **attention_mask**（`Tensor`）：attention mask。
- **query_kernel**（`Tensor`）：query的权重。
- **key_kernel**（`Tensor`）：key的权重。
- **value_kernel**（`Tensor`）：value的权重。
- **query_bias**（`Tensor`）：query的偏差值。
- **key_bias**（`Tensor`）：key的偏差值。
- **value_bias**（`Tensor`）：value的偏差值。
- **gamma**（`Tensor`）：torch.nn.LayerNorm.weight类型的tensor。
- **beta**（`Tensor`）：torch.nn.LayerNorm.bias类型的tensor。
- **scale**（`double`）：计算score的缩放系数。
- **keep_prob**：计算中保留数据的概率，值等于1-drop rate。

## 返回值说明

`torch.Tensor`

self attention的结果。


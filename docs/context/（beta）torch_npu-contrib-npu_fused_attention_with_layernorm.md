# （beta）torch_npu.contrib.npu_fused_attention_with_layernorm

>**须知：**<br>
>该接口计划废弃，可以使用torch_npu.npu_fusion_attention与torch.nn.LayerNorm接口进行替换。

## 函数原型

```
torch_npu.contrib.npu_fused_attention_with_layernorm(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, gamma, beta, scale=1, keep_prob=0)
```

## 功能说明

bert自注意力与前层规范的融合实现。

## 参数说明

- hidden_states (Tensor)：最后一层的hidden_states。
- attention_mask (Tensor)：attention mask。
- query_kernel (Tensor)：query的权重。
- key_kernel (Tensor)：key的权重。
- value_kernel (Tensor)：value的权重。
- query_bias (Tensor)：query的偏差值。
- key_bias (Tensor)：key的偏差值。
- value_bias (Tensor)：value的偏差值。
- gamma (Tensor)：torch.nn.LayerNorm.weight类型的tensor。
- beta (Tensor)：torch.nn.LayerNorm.bias类型的tensor。
- scale=1 (double)：计算score的缩放系数。
- keep_prob=0：计算中保留数据的概率，值等于1 - drop rate。

## 输出说明

torch.Tensor：self attention的结果。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

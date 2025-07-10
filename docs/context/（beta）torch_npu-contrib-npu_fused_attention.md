# （beta）torch_npu.contrib.npu_fused_attention

## 函数原型

```
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## 功能说明

bert自注意力的融合实现。

## 参数说明

- hidden_states (Tensor)：最后一层的hidden_states。
- attention_mask (Tensor)：attention mask。
- query_kernel (Tensor): query的权重。
- key_kernel (Tensor)：key的权重。
- value_kernel (Tensor)：value的权重。
- query_bias (Tensor)：query的偏差值。
- key_bias (Tensor)：key的偏差值。
- value_bias (Tensor)：value的偏差值。
- scale=1 (double)：计算score的缩放系数。
- keep_prob=0：计算中保留数据的概率，值等于1 - drop rate。

## 输出说明

torch.Tensor：self attention的结果。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


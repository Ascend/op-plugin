# （beta）torch_npu.npu_multi_head_attention

## 函数原型

```
torch_npu.npu_multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor attn_mask, Tensor out_proj_weight, Tensor query_bias, Tensor key_bias, Tensor value_bias, Tensor out_proj_bias, Tensor dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 功能说明

实现Transformer模块中的MultiHeadAttention计算逻辑。

## 参数说明

- query: Tensor类型，仅支持float16。
- key: Tensor类型，仅支持float16。
- value: Tensor类型，仅支持float16。
- query_weight: Tensor类型，仅支持float16。
- key_weight: Tensor类型，仅支持float16。
- value_weight: Tensor类型，仅支持float16。
- attn_mask: Tensor类型，仅支持float16。
- out_proj_weight: Tensor类型，仅支持float16。
- query_bias: Tensor类型，仅支持float16。
- key_bias: Tensor类型，仅支持float16。
- value_bias: Tensor类型，仅支持float16。
- out_proj_bias: Tensor类型，仅支持float16。
- dropout_mask_input: Tensor类型，仅支持float16。
- attn_head_num： Attention Head numbers，Int型。
- attn_dim_per_head：Attention dim of a Head，Int型。
- src_len：source length，Int型。
- tgt_len：target length，Int型。
- keep_prob：dropout keep probability，Float型。
- softmax_use_float：SoftMax Use Float32 to keep precision，Bool型。

## 输出说明

- y: Tensor类型，仅支持float16。
- dropout_mask: Tensor类型，仅支持float16。
- query_res: Tensor类型，仅支持float16。
- key_res: Tensor类型，仅支持float16。
- value_res: Tensor类型，仅支持float16。
- attn_scores: Tensor类型，仅支持float16。
- attn_res: Tensor类型，仅支持float16。
- context: Tensor类型，仅支持float16。

## 约束说明

Attr attn_head_num：需16整数倍对齐。

Attr attn_dim_per_head：需16整数倍对齐。

Attr src_len：需16整数倍对齐。

Attr tgt_len：需16整数倍对齐。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
import torch
import torch_npu
import numpy as np
 
batch = 8
attn_head_num = 16
attn_dim_per_head = 64
src_len = 64
tgt_len = 64
dropout_prob = 0.0
softmax_use_float = True
 
weight_col = attn_head_num * attn_dim_per_head
query = torch.from_numpy(np.random.uniform(-1, 1, (batch * tgt_len, weight_col)).astype("float16")).npu()
key = torch.from_numpy(np.random.uniform(-1, 1, (batch * src_len, weight_col)).astype("float16")).npu()
value = torch.from_numpy(np.random.uniform(-1, 1, (batch * tgt_len, weight_col)).astype("float16")).npu()
query_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
key_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
value_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
out_proj_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
attn_mask = torch.from_numpy(np.random.uniform(-1, 1, (batch, attn_head_num, tgt_len, src_len)).astype("float16")).npu()
query_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
key_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
value_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
out_proj_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
dropout_mask_input = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
            
npu_result, npu_dropout_mask, npu_query_res, npu_key_res, npu_value_res, npu_attn_scores, npu_attn_res, npu_context = torch_npu.npu_multi_head_attention (query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias,  dropout_mask_input, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)
 
print(npu_result)
 
 
 
tensor([[ 623.5000,   75.5000,  307.0000,  ...,   25.3125, -418.7500,
           35.9688],
        [-254.2500, -165.6250,  176.2500,  ...,   87.3750,   78.0000,
           65.2500],
        [ 233.2500,  207.3750,  324.7500,  ...,   38.6250, -264.2500,
          153.7500],
        ...,
        [-110.2500,  -92.5000,  -74.0625,  ...,  -68.0625,  195.6250,
         -157.6250],
        [ 300.0000, -184.6250,   -6.0039,  ...,  -15.7969, -299.0000,
          -93.1875],
        [  -2.5996,   36.8750,  100.0625,  ...,  112.7500,  202.0000,
         -166.3750]], device='npu:0', dtype=torch.float16)
```


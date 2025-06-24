# torch_npu.npu_nsa_select_attention
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>                  |    √     |
|<term>Atlas A2 训练系列产品</term> |    √     |


## 功能说明

Native Sparse Attention算法中训练场景下，实现选择注意力的计算。

## 函数原型

```
torch_npu.npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **query** (`Tensor`)：必选参数，shape支持$[T1,N1,D1]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **key** (`Tensor`)：必选参数，shape支持$[T2,N2,D1]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **value** (`Tensor`)：必选参数，shape支持$[T2,N2,D2]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **topk_indices** (`Tensor`)：必选参数，shape为$[T1, N2, select\_block\_count]$，数据类型支持`int32`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **scale_value** (`double`)：必选参数，表示缩放系数。
- **head_num** (`int`)：必选参数，表示单卡的head个数，即query的N1轴长度。
- **select_block_size** (`int`)：必选参数，表示select窗口的大小。
- **select_block_count** (`int`)：必选参数，表示select窗口的数量。
- **atten_mask** (`Tensor`)：可选参数，当前暂不支持。
- **actual_seq_qlen**(`list[int]`)：必选参数，长度表示`query`有多少个batch，值表示各batch的token长度的前缀和，例如，`actual_seq_qlen[0]=s0,actual_seq_qlen[1]=s0+s1，...，actual_seq_qlen[-1]=T1`。
- **actual_seq_kvlen**(`list[int]`)：必选参数，，长度表示`key`或`value`有多少个batch，值表示各batch的token长度的前缀和，例如，`actual_seq_kvlen[0]=s0,actual_seq_kvlen[1]=s0+s1，...，actual_seq_kvlen[-1]=T2`。

## 返回值说明
- **Tensor**：代表经过选择后的注意力attention结果。

- **Tensor**：代表softmax计算的max中间结果，用于反向计算。

- **Tensor**：代表softmax计算的sum中间结果，用于反向计算。

## 约束说明

- `query`、`key`、`value`的数据类型必须一致，同时layout必须一致。
- `query`的第2维（D1）必须等于`key`的第2维，并且`key`的第2维（D1）必须大于等于`value`的第2维（D2）。
- `query`和`key`第2维（D1）只支持192，`value`的第2维（D2）只支持128。
- `select_block_size`目前仅支持64，`select_block_count`仅支持16。


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> query = torch.randn(256, 16, 192, dtype=torch.float16).npu()
>>> key = torch.randn(3072, 4, 192, dtype=torch.float16).npu()
>>> value = torch.randn(3072, 4, 128, dtype=torch.float16).int().npu()
>>> topk_indices = torch.randn(256, 4, 16).int().npu()
>>> scale_value = 1.0
>>> head_num = 16
>>> select_block_size = 64
>>> select_block_count = 16
>>> atten_mask = torch.randn(512, 2048).bool().npu()
>>> actual_seq_qlen = [128, 256]
>>> actual_seq_kvlen = [2048, 3072]
>>> torch_npu.npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)

```


# torch_npu.npu_nsa_compress
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>                  |    √     |
|<term>Atlas A2 训练系列产品</term> |    √     |


## 功能说明

Native Sparse Attention算法中训练场景下，实现压缩功能的计算。

## 函数原型

```
torch_npu.npu_nsa_compress(input, weight, compress_block_size, compress_stride, actual_seq_len=None) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数，待压缩张量，shape支持$[T,N,D]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **weight** (`Tensor`)：必选参数，压缩的权重，shape支持$[compress\_block\_size, N]$，`weight`和`input`的shape满足broadcast关系，数据类型支持`bfloat16`、`float16`，数据类型与`input`保持一致，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **compress_block_size** (`int`)：必选参数，压缩滑窗的大小。
- **compress_stride**(`int`)：必选参数，两次压缩滑窗间隔大小。
- **actual_seq_len**(`list[int]`)：必选参数，长度表示`query`有多少个batch，值表示各batch的token长度的前缀和，例如，`actual_seq_len[0]=s0,actual_seq_len[1]=s0+s1，...，actual_seq_len[-1]=T`。

## 返回值说明
`Tensor`

代表压缩后的结果。

## 约束说明

1. `input.shape[0]`等于`actual_seq_len[-1]`。
2. `input.shape[1]`等于`weight.shape[1]`，且小于等于128。
3. `input.shape[2]`需为16的整数倍，上限256。
4. `weight.shape[0]`等于`compress_block_size`，需为16的整数倍，上限128。
5. `compress_stride`需为16的整数倍，且小于等于`compress_block_size`。


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> actual_seq_len = np.random.randint(0, 100, [48])
>>> actual_seq_len = np.cumsum(actual_seq_len).astype(np.int64)
>>> head_num = 4
>>> head_dim = 128
>>> compress_block_size = 16
>>> compress_stride = 16
>>> input = torch.randn(actual_seq_len[-1], head_num, head_dim, dtype=torch.float16).npu()
>>> weight = torch.randn(compress_block_size, head_num, dtype=torch.float16).npu()
>>> torch_npu.npu_nsa_compress(input, weight, compress_block_size, compress_stride, actual_seq_len=actual_seq_len)

```


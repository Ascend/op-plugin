# torch_npu.npu_nsa_compress_attention
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>                 |    √     |
|<term>Atlas A2 训练系列产品</term> |    √     |


## 功能说明

Native Sparse Attention算法中训练场景下，实现压缩注意力的计算。


## 函数原型

```
torch_npu.npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask=None, atten_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **query** (`Tensor`)：必选参数，shape支持$[T,N,D]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **key** (`Tensor`)：必选参数，shape支持$[T,N2,D]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **value** (`Tensor`)：必选参数，shape支持$[T,N2,D2]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **scale_value** (`double`)：必选参数，表示缩放系数。
- **head_num** (`int`)：必选参数，表示`query`的head个数。
- **compress_block_size**(`int`)：必选参数，压缩滑窗的大小。
- **compress_stride**(`int`)：必选参数，两次压缩滑窗间隔大小。
- **select_block_size** (`int`)：必选参数，表示select窗口的大小。
- **select_block_count** (`int`)：必选参数，表示select窗口的数量。
- **topk_mask**(`Tensor`)：可选参数，shape支持$[S,S]$，SS分别是max_sq和max_skv，数据类型支持`bool`。
- **atten_mask** (`Tensor`)：可选参数，取值为1代表该位不参与计算（不生效），为0代表该位参与计算，数据类型支持`bool`，数据格式支持ND，输入shape类型支持$[S,S]$格式，SS分别是maxSq和maxSkv。
- **actual_seq_qlen** (`list[int]`)：必选参数，长度表示`query`有多少个batch，值表示各batch的token长度的前缀和，例如，`actual_seq_qlen[0]=s0,actual_seq_qlen[1]=s0+s1，...，actual_seq_qlen[-1]=T`。
- **actual_cmp_seq_kvlen** (`list[int]`)：必选参数，长度表示compress attention的`key`或`value`有多少个batch，值表示各batch的token长度的前缀和，例如，`actual_cmp_seq_kvlen[0]=s0,actual_cmp_seq_kvlen[1]=s0+s1，...，actual_cmp_seq_kvlen[-1]=T`。
- **actual_sel_seq_kvlen** (`list[int]`)：必选参数，长度表示select attention的key/value有多少个batch，值表示各batch的token长度的前缀和，例如，`actual_sel_seq_kvlen[0]=s0,actual_sel_seq_kvlen[1]=s0+s1，...，actual_sel_seq_kvlen[-1]=T`。

## 返回值说明
- **Tensor**：代表压缩注意力attention的结果。

- **Tensor**：代表选择出的topk。

- **Tensor**：代表softmax计算的max中间结果，用于反向计算。

- **Tensor**：代表softmax计算的sum中间结果，用于反向计算。

## 约束说明

- `compress_block_size`、`compress_stride`和`select_block_size`必须是16的整数倍，并且`compress_block_size`大于等于`compress_stride`。
- `query`、`key`、`value`的数据类型必须一致，同时layout必须一致。
- `query`的第2维必须等于key的第2维，并且key的第2维必须大于等于`value`的第2维。
- `query`、`key`、`value`的batch size必须相等。


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> query = torch.randn(65536, 64, 192, dtype=torch.bfloat16).npu()
>>> key = torch.randn(4096, 4, 192, dtype=torch.bfloat16).npu()
>>> value = torch.randn(4096, 4, 128, dtype=torch.bfloat16).npu()
>>> scale_value = 1 / (192 ** 0.5)
>>> head_num = 64
>>> compress_block_size = 32
>>> compress_stride = 16
>>> select_block_size = 64
>>> select_block_count = 16
>>> actual_seq_qlen = [65536]
>>> actual_cmp_seq_kvlen = [4096]
>>> actual_sel_seq_kvlen = [1024]
>>> torch_npu.npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=actual_sel_seq_kvlen)

```


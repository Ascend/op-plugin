# torch_npu.npu_nsa_compress_infer
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                  |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |


## 功能说明

Native Sparse Attention算法中推理场景下，实现对KV压缩的计算。

## 函数原型

```
torch_npu.npu_nsa_compress_infer(input, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table=None, actual_seq_len=None, cache) -> Tensor(a!)
```

## 参数说明

- **input** (`Tensor`)：必选输入，待压缩张量，shape支持$[block\_num,page\_block\_size,head\_num,head\_dim]$，数据类型支持`bfloat16`、`float16`，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **weight** (`Tensor`)：必选输入，压缩的权重，shape支持$[compress\_block\_size, head\_num]$，`weight`和`input`的shape满足broadcast关系，数据类型支持`bfloat16`、`float16`，数据类型与`input`保持一致，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
- **slot_mapping**(`Tensor`)：必选输入，表示每个batch尾部压缩数据存储的位置的索引，shape支持$[batch\_num]$，数据类型支持`int32`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **compress_block_size** (`int`)：必选输入，压缩滑窗的大小。
- **compress_stride**(`int`)：必选输入，两次压缩滑窗间隔大小。
- **page_block_size**(`int`)：必选输入，page_attention场景下page的block_size大小。
- **block_table**(`Tensor`)：可选输入，page_attention场景下kv缓存使用的block映射表，不支持非连续的Tensor。
- **actual_seq_len**(`list[int]`)：必选输入，表示每个batch对应的token的长度。
- **cache**(`Tensor`)：必选输入，推理场景下的kv缓存，支持非连续的Tensor，不支持空Tensor。

## 返回值说明
`Tensor`

代表对KV压缩计算后的结果。

## 约束说明

- `compress_block_size`和`compress_stride`必须是16的整数倍，并且$compress_block_size>=compress_stride$，且`compress_block_size`小于128。
- `page_block_size`只支持64和128。
- `head_dim`必须是16的整数倍，`head_num`小于64。
- `slot_mapping`里的值无重复，否则会导致计算结果不稳定。
- `block_table`里的值小于`block_num`。


## 调用示例

   ```python
   >>> import torch
   >>> import torch_npu
   >>> input = torch.randn(1, 128, 1, 192, dtype=torch.float16).npu()
   >>> weight = torch.randn(32, 1, dtype=torch.float16).npu()
   >>> slot_mapping = torch.randn([1]).int().npu()
   >>> compress_block_size = 32
   >>> compress_stride = 16
   >>> page_block_size = 128
   >>> act_seq_lens = [43]
   >>> block_table = torch.randn([1, 1]).int().npu()
   >>> cache = torch.zeros([1, 1, 192],dtype=torch.float16).npu()
   >>> torch_npu.npu_nsa_compress_infer(input, weight,slot_mapping,compress_block_size,compress_stride,page_block_size,actual_seq_len=act_seq_lens,block_table=block_table,cache=cache)
   
   ```


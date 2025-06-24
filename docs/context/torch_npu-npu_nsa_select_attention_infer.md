# torch_npu.npu_nsa_select_attention_infer
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                  |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

Native Sparse Attention算法中推理场景下，实现选择注意力的计算。

## 函数原型

```
torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout='BSND', atten_mask=None, block_table=None, actual_seq_qlen=None, actual_seq_kvlen=None) -> Tensor
```

## 参数说明

- **query** (`Tensor`)：必选输入，shape支持3维或者4维，数据类型支持`bfloat16`、`float16`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **key** (`Tensor`)：必选输入，shape支持3维或者4维，数据类型支持`bfloat16`、`float16`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **value** (`Tensor`)：必选输入，shape支持3维或者4维，数据类型支持`bfloat16`、`float16`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **topk_indices** (`Tensor`)：必选输入，shape为`[batch_size, key_value_head_num, select_block_count]`，数据类型支持`int32`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **scale_value** (`double`)：必选输入，表示缩放系数。
- **head_num** (`int`)：必选输入，表示`query`的head个数。
- **key_value_head_num** (`int`)：必选输入，表示`key`或者`value`的head个数。
- **select_block_size** (`int`)：必选输入，表示选择窗口的大小。
- **select_block_count** (`int`)：必选输入，表示选择窗口的数量。
- **page_block_size**(`int`)：必选输入，page_attention场景下page的block_size大小。
- **atten_mask** (`Tensor`)：可选输入，当前暂不支持。
- **block_table**(`Tensor`)：可选输入，page_attention场景下kv缓存使用的block映射表，数据类型支持`int32`，不支持非连续的Tensor，不支持空tensor。
- **layout**(`str`)：可选输入，表示输入的数据排布格式，支持`BSH`、`BSND`，默认为`BSND`。
- **actual_seq_qlen**(`list[int]`)：可选输入，当前暂不支持。
- **actual_seq_kvlen**(`list[int]`)：必选输入，表示`key`/`value`每个S的长度。

## 返回值说明
`Tensor`

代表经过选择后的注意力结果。

## 约束说明

- `query`的数据排布格式中，B即Batch，S即Seq-Length，N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足$D=H/N$。`key`和`value`的数据排布格式当前（paged attention场景）支持$(block\_num, block\_size, H)$或$(block\_num, block\_size, N, D)$，H（Head-Size）表示隐藏层的大小，$H = N * D$。

- 参数`query`中的N和head_num值相等，`key`、`value`的N和`key_value_head_num`值相等，并且`head_num`是`key_value_head_num`的倍数关系。
- 参数`query`中的D和`key`的D$(H/key\_value\_head\_num)$值相等。
- `query`，`key`，`value`输入，功能使用限制如下：
  -   支持B轴小于等于3072；
  -   支持`query`的N轴与`key`/`value`的N轴（H/D）小于等于128；
  -   支持`query`的N轴与`key`/`value`的N轴（H/D）的比值小于等于128，且能够被128整除；
  -   支持`query`与`Key`的D轴小于等于192；
  -   支持`value`的D轴小于等于128；
  -   支持`query`与`Key`的D轴大于等于`value`的D轴；
  -   支持`query`与`Key`的block_size小于等于128且被16整除；
  -   仅支持`query`的S轴等于1。
  -   仅支持`key`/`value`的S轴小于等于8192。
  -   仅支持`select_block_size`、`page_block_size`取值为16的整数倍。
  -   `selectBlockCount`上限满足$select\_block\_count * select\_block\_size <= MaxKvSeqlen$，$MaxKvSeqlen = Max(actual\_seq\_kvlen)$。


## 调用示例

   ```python
   >>> import torch
   >>> import torch_npu
   >>> query = torch.randn(17, 1, 126, 192, dtype=torch.float16).npu()
   >>> key = torch.randn(187, 128, 9, 192, dtype=torch.float16).npu()
   >>> value = torch.randn(187, 128, 9, 128, dtype=torch.float16).npu()
   >>> topk_indices = torch.randn(17, 9, 21).int().npu()
   >>> scale_value = 2.0
   >>> head_num = 126
   >>> key_value_head_num = 9
   >>> select_block_size = 64
   >>> select_block_count = 21
   >>> page_block_size = 128
   >>> block_table = torch.randn(17, 11).int().npu()
   >>> actual_seq_qlen = [1]
   >>> actual_kv_seqlen = [1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328, 1328]
   >>> layout = 'BSND'
   >>> torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
   
   ```


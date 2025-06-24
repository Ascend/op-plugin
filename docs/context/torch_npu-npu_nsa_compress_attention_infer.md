# torch_npu.npu_nsa_compress_attention_infer
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                  |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |


## 功能说明

Native Sparse Attention算法中推理场景下，实现压缩注意力的计算。

## 函数原型

```
torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, atten_mask=None, block_table=None, topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None) -> (Tensor, Tensor)
```

## 参数说明

- **query** (`Tensor`)：必选输入，shape支持3维输入，数据排布格式支持TND，数据类型支持`bfloat16`、`float16`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **key** (`Tensor`)：必选输入，shape支持3维输入，数据类型支持`bfloat16`、`float16`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **value** (`Tensor`)：必选输入，shape支持3维输入，数据类型支持`bfloat16`、`float16`，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
- **scale_value** (`double`)：必选输入，表示缩放系数。
- **head_num** (`int`)：必选输入，表示`query`的head个数。
- **key_value_head_num** (`int`)：必选输入，表示`key`或者`value`的head个数。
- **select_block_size** (`int`)：必选输入，表示选择窗口的大小。
- **select_block_count** (`int`)：必选输入，表示选择窗口的数量。
- **page_block_size**(`int`)：必选输入，page_attention场景下page的block_size大小。
- **compress_block_size**(`int`)：必选输入，压缩滑窗的大小。
- **compress_stride**(`int`)：必选输入，两次压缩滑窗间隔大小。
- **atten_mask** (`Tensor`)：可选输入，当前不支持。
- **block_table**(`Tensor`)：可选输入，page_attention场景下kv缓存使用的block映射表，不支持非连续的Tensor，不支持空tensor。
- **topk_mask**(`Tensor`)：可选输入，当前不支持。
- **actual_seq_qlen** (`list[int]`)：可选输入，当前不支持。
- **actual_cmp_seq_kvlen** (`list[int]`)：必选输入，表示压缩注意力的`key`/`value`的每个S的长度。
- **actual_sel_seq_kvlen** (`list[int]`)：可选输入，当前不支持。

## 返回值说明
**Tensor**：代表压缩注意力的结果。

**Tensor**：代表select topk的计算结果。

## 约束说明


- `query`的数据排布格式中，T代表B（Batch）与S（Seq-Length）合轴后的结果、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足$D=H/N$。`key`和`value`的数据排布格式当前（paged attention）支持$（block\_num, block\_size, H）$，H（Head-Size）表示隐藏层的大小，$H = N * D$。

- 参数`query`中的N和head_num值相等，`key`、`value`的N和key_value_head_num值相等，并且head_num是key_value_head_num的倍数关系。
- 参数`query`中的D和`key`的D(H/key_value_head_num)值相等。
- 参数`query`中的B、`block_table`的B、a`ctual_cmp_seq_kvlen`的shape值相等。
- 参数`key`中的block_num和参数`value`中的block_num值相等。
- 参数`key`中的block_size、参数`value`中的block_size和page_block_size值相等。
- `query`，`key`，`value`输入，功能使用限制如下：
  -   支持`query`的N轴必须是`key`/`value`的N轴（H/D）的整数倍；
  -   支持`query`的N轴与`key`/`value`的N轴（H/D）的比值小于等于128，且128是group的整数倍；
  -   支持`query`与`Key`的D轴小于等于192；
  -   支持`value`的D轴小于等于128；
  -   支持`query`与`Key`的D轴大于等于value的D轴；
  -   支持`key`与`value`的block_size小于等于128，且是16的整数倍；
  -   仅支持`query`的S轴等于1。
  -   仅支持paged attention。
  -   仅支持`key`/`value`的S轴小于等于8192。
  -   仅支持`compress_block_size`取值16、32、64。
  -   仅支持`compress_stride`取值16、32、64。
  -   仅支持`select_block_size`取值16、32、64。
  -   仅支持`compress_block_size`大于等于`compress_stride` , `select_block_size`大于等于`compress_block_size` , `select_block_size`是`compress_stride`的整数倍。
  -   压缩前的kvSeqlen的上限可以表示为：$NoCmpKvSeqlenCeil =（cmpKvSeqlen - 1）* compress_block_stride + compress_block_siz$e，需要满足$NoCmpKvSeqlenCeil / select_block_size <= 4096$，且需要满足$select_block_count <= NoCmpKvSeqlenCeil / select_block_size$。


## 调用示例

   ```python
   >>> import torch
   >>> import torch_npu
   >>> query = torch.randn([20, 8, 192], dtype=torch.bfloat16).npu()
   >>> key = torch.randn([160, 128, 384], dtype=torch.bfloat16).npu()
   >>> value = torch.randn([160, 128, 256], dtype=torch.bfloat16).npu()
   >>> scale_value = 1.0
   >>> head_num = 8
   >>> key_value_head_num = 2
   >>> compress_block_size = 32
   >>> compress_stride = 16
   >>> select_block_size = 64
   >>> select_block_count = 16
   >>> page_block_size = 128
   >>> block_table = torch.randn([20, 8]).int().npu()
   >>> actual_cmp_seq_kvlen = [1024] * 20
   >>> torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen)
   
   ```

 


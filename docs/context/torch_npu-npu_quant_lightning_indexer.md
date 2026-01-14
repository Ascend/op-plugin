# torch_npu.npu_quant_lightning_indexer

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 推理系列产品</term>   | √  |

## 功能说明

-   API功能：QuantLightningIndexer是推理场景下，SparseFlashAttention（SFA）前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

-   计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{INT8}@{\left(K_{index}^{INT8}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{INT8}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{INT8}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为SparseFlashAttention的输入。

## 函数原型

```
torch_npu.npu_quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1) -> Tensor
```

## 参数说明
> [!NOTE]
> - query、key、weights、query_dequant_scale、key_dequant_scale参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - 使用S1和S2分别表示query和key的输入样本序列长度，N1和N2分别表示query和key对应的多头数，k表示最后选取的索引个数。参数query中的D和参数key中的D值相等为128。T1和T2分别表示query和key的输入样本序列长度的累加和。
-   **query**（`Tensor`）：必选参数，表示输入Index Query，对应公式中的$Q_{index}^{INT8}\in\R^{g\times d}$。不支持非连续，数据格式支持$ND$，数据类型支持`int8`。`layout_query`为BSND时shape为[B,S1,N1,D]，当`layout_query`为TND时shape为[T1,N1,D]，N1仅支持64。
    
-   **key**（`Tensor`）：必选参数，表示输入Index Key，对应公式中的$K_{index}^{INT8}\in\R^{S_{k}\times d}$。不支持非连续，数据格式支持$ND$，数据类型支持`int8`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2, D]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数，block\_size取值为16的整数倍，最大支持到1024。`layout_key`为BSND时shape为[B, S2, N2, D]，`layout_key`为TND时shape为[T2, N2, D]，N2仅支持1。
    
-   **weights**（`Tensor`）：必选参数，表示权重系数，对应公式中的$W$。不支持非连续，数据格式支持$ND$，数据类型支持`float16`，支持输入shape[B,S1,N1]、[T,N1]。

-   **query_dequant_scale**（`Tensor`）：必选参数，表示Index Query的反量化系数$Scale_Q$ 。不支持非连续，数据格式支持$ND$，数据类型支持`float16`，支持输入shape[B,S1,N1]、[T,N1]。

-   **key_dequant_scale**（`Tensor`）：必选参数，表示Index Key的反量化系数，对应公式中的$Scale_K^T$。不支持非连续，数据格式支持$ND$，数据类型支持`float16`，layout\_key为PA_BSND时shape为[block\_count, block\_size, N2]，其中block\_count为PageAttention时block总数，block\_size为一个block的token数。

-   **query\_quant\_mode**（`int`）：可选参数，用于标识输入`query`的量化模式，当前仅支持Per-Token-Head量化模式，当前仅支持传入0。

-   **key\_quant\_mode**（`int`）：可选参数，用于标识输入`key`的量化模式，当前仅支持Per-Token-Head量化模式，当前仅支持传入0。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入；之后的参数是可选参数，位置无关，不赋值会使用默认值。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。不能出现负值。

-   **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。当`layout_key`为TND或PA_BSND时，该入参必须传入，`layout_key`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持$ND$，数据类型支持`int32`。PageAttention场景下，block\_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为每个batch中最大actual\_seq\_lengths\_key对应的block数量)，支持block_size取值为16的整数倍，最大支持到1024。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，当前支持PA_BSND、BSND、TND，默认值"BSND"。在非PageAttention场景下，layout\_key应与layout\_query保持一致。

-   **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持[1, 2048]，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`int32`。 sparse\_mode为0时，代表defaultMask模式。sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

## 返回值说明
`Tensor`

代表公式中的输出Out。数据格式支持$ND$，数据类型支持`int32`，支持输出shape[B,S1,N2,k]或[T,N2,k]。

## 约束说明
-   该接口支持图模式。
-   该接口要求$W \odot Scale_Q$的结果在`float16`的表示范围内。
-   该接口的TopK过程对NAN排序是未定义行为。

## 调用示例

-   单算子模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 0
    sparse_count = 2048
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
    key_dequant_scale = key_dequant_scale.to(torch.float16)
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
    
    npu_out = torch_npu.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                    key_dequant_scale.npu(),
                                                    actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                    actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                    block_table=block_table.npu(),
                                                    query_quant_mode=query_quant_mode,
                                                    key_quant_mode=key_quant_mode,
                                                    layout_query=layout_query,
                                                    layout_key=layout_key, sparse_count=sparse_count,
                                                    sparse_mode=sparse_mode)
    ```
-   图模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 0
    sparse_count = 2048
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
    key_dequant_scale = key_dequant_scale.to(torch.float16)
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
    
    class LIQuantNetwork(nn.Module):
        def __init__(self):
            super(LIQuantNetwork, self).__init__()

        def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query=None, 
                    actual_seq_lengths_key=None, block_table=None, query_quant_mode=0, key_quant_mode=0,
                    layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3):

            out = torch_npu.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),       
                                                        key_dequant_scale.npu(),
                                                        actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                        actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                        block_table=block_table.npu(),
                                                        query_quant_mode=query_quant_mode,
                                                        key_quant_mode=key_quant_mode,
                                                        layout_query=layout_query,
                                                        layout_key=layout_key, sparse_count=sparse_count,
                                                        sparse_mode=sparse_mode)
            return out
    
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(LIQuantNetwork().npu(), fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale,
                        actual_seq_lengths_query=actual_seq_lengths_query,
                        actual_seq_lengths_key=actual_seq_lengths_key,
                        block_table=block_table, query_quant_mode=query_quant_mode,
                        key_quant_mode=key_quant_mode, layout_query=layout_query,
                        layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
    ```
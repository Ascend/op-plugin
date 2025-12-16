# torch_npu.npu_sparse_flash_attention

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明
- API功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。
    本次公布的`sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 函数原型

```
torch_npu.npu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, attention_mode=0, return_softmax_lse=False) -> (Tensor, Tensor, Tensor)
```

## 参数说明

> [!NOTE]  
>- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N和N1表示num\_query\_heads，KV\_N和N2表示num\_key\_value\_heads，T1表示query shape中的T，T2表示key shape中的输入样本序列长度的累加和。
-   **query**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。`layout_query`为BSND时shape为[B,S1,N1,D]，当`layout_query`为TND时shape为[T1,N1,D]，其中N1支持1/2/4/8/16/32/64/128。
-   **key**（`Tensor`）：必选参数，对应公式中的$\tilde{K}$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`，`layout_kv`时shape为[block\_num, block\_size, KV\_N, D]，其中block\_num为PageAttention时block总数，block\_size为一个block的token数，block\_size取值为16的倍数，最大支持1024。`layout_kv`为BSND时shape为[B, S2, KV\_N, D]，`layout_kv`为TND时shape为[T2, KV\_N, D]，其中KV\_N只支持1。

-   **value**（`Tensor`）：必选参数，不支持非连续，对应公式中的$\tilde{V}$，维度N只支持1，数据格式支持ND，数据类型支持`bfloat16`和`float16`，shape与`key`的shape一致。
    
-   **sparse\_indices**（`Tensor`）：必选参数，代表离散取kvCache的索引，不支持非连续，数据格式支持ND,数据类型支持`int32`。当`layout_query`为BSND时，shape需要传入[B, Q\_S, KV\_N, sparse\_size]，当`layout_query`为TND时，shape需要传入[Q\_T, KV\_N, sparse\_size]，其中sparse\_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse\_size大于0。

-   **scale\_value**（`double`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持`float`。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中kvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **actual\_seq\_lengths\_kv**（`Tensor`）：可选参数，表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`。如果不指定None，表示和key的shape的S长度相同。该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **query\_rope**（`Tensor`）：可选参数，表示MLA结构中的query的rope信息，不支持非连续，数据格式支持ND,数据类型支持`bfloat16`和`float16`。
    
-   **key\_rope**（`Tensor`）：可选参数，表示MLA结构中的key的rope信息，不支持非连续，数据格式支持ND,数据类型支持`bfloat16`和`float16`。

-   **sparse\_block\_size**（`int`）：可选参数，代表sparse阶段的block大小，在计算importance score时使用，数据类型支持`int64`，取值范围为[1,128]，且为2的幂次方。
    -   sparse_block_size为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。
    -   sparse_block_size为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，用户不特意指定时可传入默认值"BSND"，支持传入BSND和TND。

-   **layout\_kv**（`str`）：可选参数，用于标识输入`key`的数据排布格式，用户不特意指定时可传入默认值"BSND"，支持传入TND、BSND和PA\_BSND，其中PA\_BSND在使能PageAttention时使用。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式。数据类型支持`int64`。
    -   sparse\_mode为0时，代表全部计算。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

-   **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **attention\_mode**（`int`）：可选参数，表示attention的模式，数据类型支持`int64`，仅支持传入2，表示MLA-absorb模式，即计算过程中会将query和key的nope部分分别和query_rope和key_rope的rope部分沿头维度（D）拼接，合并形成最终的query和key用于后续计算，且key和value共享同一份底层张量数据。

-   **return\_softamx\_lse**（`bool`）：可选参数，用于表示是否返回softmax_max和softmax_sum。True表示返回，但图模式下不支持，False表示不返回；默认值为False。该参数仅在训练且`layout_kv`不为PA_BSND场景支持。

## 返回值说明

-   **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`和`float16`。当layout\_query为BSND时shape为[B,S1,N1,D]，当layout\_query为TND时shape为[T1,N1,D]。
-   **softmax\_max**（`Tensor`）：可选输出，Attention算法对query乘key的结果，取max得到softmax_max，数据类型支持`float`。当layout\_query为BSND时shape为[B,N2,S1,N1/N2]，当layout\_query为TND时shape为[N2,T1,N1/N2]。
-   **softmax\_sum**（`Tensor`）：可选输出，Attention算法query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum，数据类型支持`float`。当layout\_query为BSND时shape为[B,N2,S1,N1/N2]，当layout\_query为TND时shape为[N2,T1,N1/N2]。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   参数query中的D和key、value的D值相等为512，参数query\_rope中的D和key\_rope的D值相等为64。
-   参数query、key、value的数据类型必须保持一致。
-   支持sparse\_block\_size整除block\_size。
-   layout\_query为TND且layout\_kv为BSND场景不支持，在非PageAttention场景下，该参数值应与layout_query值保持一致。

## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    
    query_type = torch.float16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    sparse_block_count = 2048
    t = 10
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    tile_size = 128
    block_size = 256
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
        to(torch.int32)
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
    act_seq_q = [s1] * b
    act_seq_kv = [s2_act] * b

    query = query.npu()
    key = key.npu()
    value = value.npu()
    sparse_indices = sparse_indices.npu()
    query_rope = query_rope.npu()
    key_rope = key_rope.npu()
    act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

    attention_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query, key, value, sparse_indices, scale_value, block_table=None, 
        actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1,
        attention_mode = attention_mode, return_softmax_lse = return_softmax_lse)
    ```
- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import torch.nn as nn
    import numpy as np
    import random

    query_type = torch.float16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    sparse_block_count = 2048
    t = 10
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    tile_size = 128
    block_size = 256
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
        to(torch.int32)
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
    act_seq_q = [s1] * b
    act_seq_kv = [s2_act] * b

    query = query.npu()
    key = key.npu()
    value = value.npu()
    sparse_indices = sparse_indices.npu()
    query_rope = query_rope.npu()
    key_rope = key_rope.npu()
    act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, query, key, value, sparse_indices, scale_value, 
            block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
            query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
            sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse):
            
            attention_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
                query, key, value, sparse_indices, scale_value, block_table=None, 
                actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_kv=actual_seq_lengths_kv,
                query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
                layout_query=layout_query, layout_kv=layout_kv, sparse_mode=sparse_mode, 
                pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1, attention_mode = attention_mode, 
                return_softmax_lse = return_softmax_lse)
            return attention_out, softmax_max, softmax_sum

    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)

    attention_out, softmax_max, softmax_sum = mod(query, key, value, sparse_indices, 
        scale_value, block_table=None, actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1,
        attention_mode = attention_mode, return_softmax_lse = return_softmax_lse)
    ```


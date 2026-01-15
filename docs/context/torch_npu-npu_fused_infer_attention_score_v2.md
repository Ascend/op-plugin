# torch\_npu.npu\_fused\_infer\_attention\_score\_v2<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas A2 推理系列产品</term> |    √     |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

-   API功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当不涉及system prefix、左padding、kv量化参数合一、pertensor全量化的场景，推荐使用本接口，否则使用老接口`npu_fused_infer_attention_score`。
-   计算公式：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，$d$表示隐藏层最小的单元尺寸。为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵，$n$为输出矩阵的行数。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
torch_npu.npu_fused_infer_attention_score_v2(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, learnable_sink=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

> [!NOTE]   
> - query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示隐藏层的大小、N（Head Num）表示多头数、D（Head Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - Q_S和S1表示query shape中的S，KV_S和S2表示key和value shape中的S，Q_N表示num\_query\_heads，KV_N表示num\_key\_value\_heads。

-   **query**（`Tensor`）：必选参数，表示attention结构的Query输入，对应公式中的`Q`。不支持非连续的Tensor，数据类型支持`float16`、`bfloat16`，数据格式支持ND。    
    
-   **key**（`Tensor`）：必选参数，表示attention结构的Key输入，对应公式中的`K`。不支持非连续的Tensor，数据类型支持`float16`、`bfloat16`、`int8`、`int4`（`int32`），数据格式支持ND。   
     
-   **value**（`Tensor`）：必选参数，表示attention结构的Value输入，对应公式中的`V`。不支持非连续的Tensor，数据类型支持`float16`、`bfloat16`、`int8`、`int4`（`int32`），数据格式支持ND。    
    
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **query\_rope**（`Tensor`）：可选参数，表示MLA（Multi-head Latent Attention）结构中`query`的rope信息，数据类型支持`float16`、`bfloat16`，不支持非连续的Tensor，数据格式支持ND。
-   **key\_rope**（`Tensor`）：可选参数，表示MLA（Multi-head Latent Attention）结构中的`key`的rope信息，数据类型支持`float16`、`bfloat16`，不支持非连续的Tensor，数据格式支持ND。
-   **pse\_shift**（`Tensor`）：可选参数，表示attention结构内部的位置编码参数，数据类型支持`float16`、`bfloat16`，数据类型与`query`数据类型需满足类型推导规则。不支持非连续的Tensor，数据格式支持ND。如不使用该功能可传入None。

    -   Q\_S大于1，当`pse_shift`为`float16`类型时，要求`query`为float16或int8类型；当`pse_shift`为`bfloat16`类型时，要求`query`为`bfloat16`类型。输入shape类型需为\(B, Q\_N, Q\_S, KV\_S\)或\(1, Q\_N, Q\_S, KV\_S\)。对于`pse_shift`的KV\_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
    -   Q\_S为1，当`pse_shift`为`float16`类型时，要求`query`为`float16`类型；当`pse_shift`为`bfloat16`类型时，要求`query`为`bfloat16`类型。输入shape类型需为\(B, Q\_N, 1, KV\_S\)或\(1, Q\_N, 1, KV\_S\)。对于`pse_shift`的KV\_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。

-   **atten\_mask**（`Tensor`）：可选参数，对QK结果进行mask，用来指示是否计算Token间的相关性。数据类型支持`bool`、`int8`和`uint8`。不支持非连续的Tensor，数据格式支持ND。如不使用该功能可传入None。
    - `sparse_mode`为0、1时
        - 支持shape传入(1,Q_S,KV_S)、(B,1,Q_S,KV_S)、(1,1,Q_S,KV_S)。
        - 当输入`input_layout`为BSH、BSND、BNSD、BNSD_BSND时，且query、key、value的D相等，并且不传`query_rope`和`key_rope`时，Q_S为1可支持传入(B,KV_S)，Q_S大于1时可支持传入(Q_S,KV_S)。
        - 如果Q\_S、KV\_S非16或32对齐，可以向上取到对齐的S。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
    - `sparse_mode`为2、3、4时，shape输入支持(2048,2048)或(1,2048,2048)或(1,1,2048,2048)。    
-   **actual\_seq\_qlen**（`List[Int]`）：可选参数，表示不同Batch中`query`的有效seqlen，数据类型支持`int64`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。
    该入参中每个Batch的有效seqlen不超过`query`中对应batch的seqlen。当seqlen传入长度为1时，每个Batch使用相同seqlen；当seqlen传入长度>=Batch时，取seqlen的前Batch个数；其他长度不支持。当`query`的input\_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须>=前一个元素的值，且不能出现负值。

-   **actual\_seq\_kvlen**（`List[Int]`）：可选参数，表示不同Batch中`key`/`value`的有效seqlenKv，数据类型支持`int64`。如果不指定None，表示和key/value的shape的S长度相同。不同O\_S值有不同的约束，具体参见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中KV存储使用的block映射表，数据类型支持`int32`。数据格式支持ND。如不使用该功能可传入None。
-   **dequant\_scale\_query**（`Tensor`）：可选参数，表示`query`的反量化参数，仅支持pertoken叠加perhead。数据类型支持`float32`。数据格式支持ND，如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
-   **dequant\_scale\_key**（`Tensor`）：可选参数，kv伪量化参数分离时表示`key`的反量化因子。数据类型支持`float16`、`bfloat16`、`float32`，数据格式支持ND。通常支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理scale、pertoken叠加per head并使用page attention模式管理scale。如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。    
    
-   **dequant\_offset\_key**（`Tensor`）：可选参数，kv伪量化参数分离时表示`key`的反量化偏移。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持ND。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理offset、pertoken叠加perhead并使用page attention模式管理offset。如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
-   **dequant\_scale\_value**（`Tensor`）：可选参数，kv伪量化参数分离时表示`value`的反量化因子。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持ND。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理scale、pertoken叠加perhead并使用page attention模式管理scale。如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
    
-   **dequant\_offset\_value**（`Tensor`）：可选参数，kv伪量化参数分离时表示`value`的反量化偏移。数据类型支持`float16`、`bfloat16`、`float32`。数据格式支持ND。支持perchannel、pertensor、pertoken、pertensor叠加perhead、pertoken叠加perhead、pertoken叠加使用page attention模式管理offset、pertoken叠加perhead并使用page attention模式管理offset。如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
-   **dequant\_scale\_key\_rope**（`Tensor`）：可选参数，**预留参数，暂未使用，使用默认值即可。**
-   **quant\_scale\_out**（`Tensor`）：可选参数，表示输出的量化因子。数据类型支持`float32`、`bfloat16`。数据格式支持ND。支持pertensor、perchannel。当输入为`bfloat16`时，同时支持`float32`、`bfloat16`，否则仅支持`float32`。perchannel格式，当输出layout为BSH时，要求`quant_scale_out`所有维度的乘积等于H；其他layout要求乘积等于Q\_N\*D（建议输出layout为BSH时，quant\_scale\_out shape传入\(1, 1, H\)或\(H,\)；输出为BNSD时，建议传入\(1, Q\_N, 1, D\)或\(Q\_N, D\)；输出为BSND时，建议传入\(1, 1, Q\_N, D\)或\(Q\_N, D\)）。如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
-   **quant\_offset\_out**（`Tensor`）：可选参数，表示输出的量化偏移。数据类型支持`float32`、`bfloat16`。数据格式支持ND。支持pertensor、perchannel。若传入`quant_offset_out`，需保证其类型和shape信息与`quant_scale_out`一致。如不使用该功能可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
-   **learnable_sink**（`Tensor`）：可选参数，表示通过可学习的“Sink Token”起到吸收Attention Score的作用，数据类型支持`bfloat16`，数据格式支持ND，shape输入为(Q_N,)。默认值为None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

-   **num\_query\_heads**（`int`）：可选参数，代表query的head个数，数据类型支持`int64`，在BNSD场景下，需要与shape中的`query`的N轴shape值相同，否则执行异常。
-   **num\_key\_value\_heads**（`int`）：可选参数，代表`key`、`value`中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，数据类型支持`int64`。默认值为0，表示`key`/`value`/`query`的head个数相等，需要满足`num_query_heads`整除`num_key_value_heads`，`num_query_heads`与`num_key_value_heads`的比值不能大于64。在BSND、BNSD、BNSD\_BSND（仅支持Q\_S大于1）场景下，还需要与shape中的`key`/`value`的N轴shape值相同，否则执行异常。
-   **softmax\_scale**（`float`）：可选参数，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持`float32`。数据类型与`query`数据类型需满足数据类型推导规则。默认值为1.0。
-   **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`。默认值为2147483647，Q\_S为1时该参数无效。
-   **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`。默认值为2147483647，Q\_S为1时该参数无效。
-   **input\_layout**（`str`）：可选参数，用于标识输入`query`、`key`、`value`的数据排布格式，默认值为"BSH"。

    > [!NOTE]   
    > 注意排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出output的格式，算子内部会进行layout转换。

    支持BSH、BSND、BNSD、BNSD\_BSND（输入为BNSD时，输出格式为BSND，仅支持Q\_S大于1）、BSH\_NBSD、BSND\_NBSD、BNSD\_NBSD（输出格式为NBSD时，仅支持Q\_S大于1且小于等于16）、TND、TND\_NTD、NTD\_TND（TND相关场景综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)）。其中BNSD\_BSND含义指当输入为BNSD，输出格式为BSND，仅支持Q\_S大于1。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式。数据类型支持`int64`。Q\_S为1且不带rope输入时该参数无效。input\_layout为TND、TND\_NTD、NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
    
    -   `sparse_mode`为0时，代表defaultMask模式，如果atten\_mask未传入则不做mask操作，忽略pre\_tokens和next\_tokens（内部赋值为INT\_MAX）；如果传入，则需要传入完整的atten\_mask矩阵（S1\*S2），表示pre\_tokens和next\_tokens之间的部分需要计算。
    -   `sparse_mode`为1时，代表allMask，必须传入完整的atten\_mask矩阵（S1\*S2）。
    -   `sparse_mode`为2时，代表leftUpCausal模式的mask，需要传入优化后的atten\_mask矩阵（2048\*2048）。
    -   `sparse_mode`为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的atten\_mask矩阵（2048\*2048）。
    -   `sparse_mode`为4时，代表band模式的mask，需要传入优化后的atten\_mask矩阵（2048\*2048）。
    -   `sparse_mode`为5、6、7、8时，分别代表prefix、global、dilated、block\_local，均暂不支持。默认值为0。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
    
-   **block\_size**（`int`）：可选参数，表示PageAttention中KV存储每个block中最大的token个数，默认为0，数据类型支持`int64`。
-   **query\_quant\_mode**（`int`）：可选参数， 表示query的伪量化方式。仅支持传入3，代表模式3：pertoken叠加perhead模式。
-   **key\_quant\_mode**（`int`）：可选参数，表示key的伪量化方式，默认值为0。取值除了`key_quant_mode`为0且`value_quant_mode`为1的场景外，其他场景取值需要与`value_quant_mode`一致。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    当Q\_S>=2时，仅支持传入值为0、1；当Q\_S=1时，支持取值0、1、2、3、4、5。

    -   `key_quant_mode`为0时，代表perchannel模式（perchannel包含pertensor）。
    -   `key_quant_mode`为1时，代表pertoken模式。
    -   `key_quant_mode`为2时，代表pertensor叠加perhead模式。
    -   `key_quant_mode`为3时，代表pertoken叠加perhead模式。
    -   `key_quant_mode`为4时，代表pertoken叠加使用page attention模式管理scale/offset模式。
    -   `key_quant_mode`为5时，代表pertoken叠加per head并使用page attention模式管理scale/offset模式。

- **value\_quant\_mode**（`int`）：可选参数，表示`value`的伪量化方式，模式编号与`key_quant_mode`一致，默认值为0。取值除了`key_quant_mode`为0且`value_quant_mode`为1的场景外，其他场景取值需要与`key_quant_mode`一致。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    当Q\_S>=2时，仅支持传入值为0、1；当Q\_S=1时，支持取值0、1、2、3、4、5。

-   **inner\_precise**（`int`）：可选参数，数据类型支持`int64`，支持4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。当Q\_S\>1时，sparse\_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效；Q\_S为1时该参数仅支持取0和1。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    -   inner\_precise为0时，代表开启高精度模式，且不做行无效修正。
    -   inner\_precise为1时，代表高性能模式，且不做行无效修正。
    -   inner\_precise为2时，代表开启高精度模式，且做行无效修正。
    -   inner\_precise为3时，代表高性能模式，且做行无效修正。

    > [!NOTE]   
    > bfloat16和int8不区分高精度和高性能，行无效修正对`float16`、`bfloat16`和`int8`均生效。当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。

-   **return\_softmax\_lse**（`bool`）：可选参数，表示是否输出`softmax_lse`，支持S轴外切（增加输出）。true表示输出，false表示不输出；默认值为false。
-   **query_dtype**（`int`）：可选参数，表示`query`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **key_dtype**（`int`）：可选参数，表示`key`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **value_dtype**（`int`）：可选参数，表示`value`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **query_rope_dtype**（`int`）：可选参数，表示`query_repo`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **key_rope_dtype**（`int`）：可选参数，表示`key_rope`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **key_shared_prefix_dtype**（`int`）：可选参数，表示key_shared_prefix的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **value_shared_prefix_dtype**（`int`）：可选参数，表示value_shared_prefix的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **dequant_scale_query_dtype**（`int`）：可选参数，表示`dequant_scale_query`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **dequant_scale_key_dtype**（`int`）：可选参数，表示`dequant_scale_key`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **dequant_scale_value_dtype**（`int`）：可选参数，表示`dequant_scale_value`的数据类型，**预留参数，暂未使用，使用默认值即可。**
-   **dequant_scale_key_rope_dtype**（`int`）：可选参数，表示`dequant_scale_key_rope`的数据类型，**预留参数，暂未使用，使用默认值即可。**

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **attention\_out**（`Tensor`）：公式中的输出，数据类型支持`float16`、`bfloat16`、`int8`。数据格式支持ND。限制：该入参的D维度与`value`的D保持一致，其余维度需要与入参`query`的shape保持一致。
-   **softmax\_lse**（`Tensor`）：ring attention算法对query乘key的结果先取max得到softmax\_max，query乘key的结果减去softmax\_max，再取exp，最后取sum，得到softmax\_sum，最后对softmax\_sum取log，再加上softmax\_max得到的结果。数据类型支持`float32`，当`return_softmax_lse`为True时，一般情况下输出shape为\(B, Q\_N, Q\_S, 1\)，若input\_layout为TND/NTD\_TND时，输出shape为\(T,Q\_N,1\)；当`return_softmax_lse`为False时，输出shape为\[1\]的值为0的Tensor。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则填充全零的对应shape的输出（填充attention\_out）。attention\_out为空Tensor时，框架会处理。
-   参数key、value中对应tensor的shape需要完全一致；非连续场景下key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。
-   int8量化相关入参数量与输出数据格式的综合限制：
    -   输出为int8的场景：入参quant\_scale\_out需要存在，quant\_offset\_out可选，不传时默认为0。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输入为int8。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：输入为int8。

    -   输出为float16的场景：若存在入参quant\_offset\_out或quant\_scale\_out（即不为None），则报错并返回。
    -   入参quant\_offset\_out和quant\_scale\_out支持pertensor或perchannel格式，数据类型支持float32、bfloat16。

-   query\_rope和key\_rope输入时即为MLA场景，参数约束如下：
    -   query\_rope的数据类型、数据格式与query一致。
    -   key\_rope的数据类型、数据格式与key一致。
    -   query\_rope和key\_rope要求同时配置或同时不配置，不支持只配置其中一个。
    -   当query\_rope和key\_rope非空时，query的D只支持512、128；
        -   当query的D等于512时：
            - sparse：支持0/3/4；
            - query\_rope配置时要求query的N为1/2/4/8/16/32/64/128，query\_rope的shape中D为64，其余维度与query一致；
            - key\_rope配置时要求key的N为1、D为512，key\_rope的shape中D为64，其余维度与key一致；
            - 支持key、value、key\_rope的数据格式为ND或NZ。当数据格式为NZ时，若数据类型为float16或bfloat16，输入参数key和value的格式为\[blockNum, KV\_N, D/16, blockSize, 16\]；若输入数据类型为int8，输入参数key和value的格式为\[blockNum, KV\_N, D/32, blockSize, 32\]；
            - input\_layout形状支持BSH、BSND、BNSD、BNSD\_NBSD、BSND\_NBSD、BSH\_NBSD、TND、TND\_NTD；
            - 支持开启page attention，此时block\_size支持16的倍数且不大于1024；
            - 不支持开启左padding、tensorlist、pse、prefix、伪量化、后量化、空Tensor。
            - 支持全量化场景，即输入query/key/value全为int8，query\_rope和key\_rope为bfloat16，输出为bfloat16的场景：
                - 入参dequant\_scale\_query、dequant\_scale\_key、dequant\_scale\_value需要同时存在，且其数据类型仅支持float32。
                - 不支持传入quant\_scale\_out、quant\_offset\_out、dequant\_offset\_key、dequant\_offset\_value，否则报错并返回。
                - query\_quant\_mode仅支持pertoken叠加perhead模式，key\_quant\_mode和value\_quant\_mode仅支持pertensor模式。
                - 支持key、value、key\_rope的input\_layout格式为NZ。
        -   当query的D等于128时：
            - input\_layout：BSH、BSND、TND、BNSD、NTD、BSH\_BNSD、BSND\_BNSD、BNSD\_BSND、NTD\_TND。    
            - query\_rope配置时要求query\_rope的shape中D为64，其余维度与query一致。  
            - key\_rope配置时要求key\_rope的shape中D为64，其余维度与key一致。  
            - 不支持开启左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor。
            - 其余约束同TND、NTD\_TND场景下的综合限制保持一致。

    -   TND、TND\_NTD、NTD\_TND场景下query、key、value输入的综合限制：
        -   actual\_seq\_qlen和actual\_seq\_kvlen必须传入，且以该入参元素数量作为Batch值（注意入参元素数量要小于等于4096）。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值；
        -   当query的D等于512时：
            -   sparse：支持0/3/4；
            -   支持TND、TND\_NTD；
            -   支持开启page attention，此时actual\_seq\_kvlen长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV\_S；
            -   要求query的N为1/2/4/8/16/32/64/128，key、value的N为1；
            -   要求query\_rope和key\_rope不等于空，query\_rope和key\_rope的D为64；
            -   不支持开启左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor。

        -   当query的D不等于512时：
            -   当query\_rope和key\_rope为空时：TND场景，要求Q\_D、K\_D、V\_D等于128，或者Q\_D、K\_D等于192，V\_D等于128/192；NTD场景，不支持V\_D等于192；NTD\_TND场景，要求Q\_D、K\_D等于128/192，V\_D等于128。当query\_rope和key\_rope不为空时，要求Q\_D、K\_D、V\_D等于128；GQA和PA场景不支持V_D等于192;
            -   支持TND、NTD、NTD\_TND；
            -   page attention场景下仅支持blocksize为16对齐且小于等于1024;
            -   不支持开启左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor；
-   GQA伪量化场景下KV为NZ格式时的参数约束如下：
    - 支持perchannel和pertoken模式，query数据类型固定为bfloat16，key&value固定为int8；query&key&value的D仅支持128；query Sequence Length仅支持1-16；
    - input\_layout仅支持BSH、BSND、BNSD；
    - 仅支持page_attention场景，blockSize仅支持128或512；
    - key&value仅支持NZ输入，输入格式为[blockNum, KV\_N, D/32, blockSize, 32]；
    - dequant\_scale\_key和dequant\_scale\_value的dtype：perchannel模式下，仅支持bfloat16类型；pertoken模式下，仅支持float32类型；
    - dequant\_scale\_key和dequant\_scale\_value的shape：perchannel模式下，当layout为BSH时，必须传入[H]；layout为BNSD时，必须传入[KV\_N,1,D]；输出为BSND时，必须传入[KV\_N, D]；pertoken模式下，必须传入[B,KV\_S]，S需要大于等于blockTable的第二维*blockSize；
    - 仅支持KV分离；
    - 仅支持高性能模式；
    - 当MTP等于0时，支持sparse\_mode=0且不传mask；当MTP大于0、小于16时，支持sparse\_mode=3且传入优化后的atten\_mask矩阵，atten\_mask矩阵shape必须传入（2048\*2048）；
    - 不支持配置dequant\_offset\_key和dequant\_offset\_value;
    - 不支持配置query\_rope和key\_rope；
    - 不支持左padding、tensorlist、pse、prefix、后量化；
    - num\_query\_heads与num\_key\_value\_heads支持组合有(10, 1)、(64, 8)、(80, 8)、(128, 16)。
-   learnable_sink的参数约束如下：
    - 仅支持TND、NTD\_TND；
    - 仅支持value的d小于等于128；
-   **当Q\_S大于1时：**
    -   query、key、value输入，功能使用限制如下：
        -   支持B轴小于等于65536，D轴32byte不对齐时仅支持到128。
        -   支持N轴小于等于256，支持D轴小于等于512；input\_layout为BSH或者BSND时，建议N\*D小于65535。
        -   S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致PFA算子执行超时（aicore error类型报错，errorStr为timeout or trap error），此场景下建议做S切分处理（注：这里计算量会受B、S、N、D等的影响，值越大计算量越大），典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
            -   B=1，Q\_N=20，Q\_S=2097152，D=256，KV\_N=1，KV\_S=2097152。
            -   B=1，Q\_N=2，Q\_S=20971520，D=256，KV\_N=2，KV\_S=20971520。
            -   B=20，Q\_N=1，Q\_S=2097152，D=256，KV\_N=1，KV\_S=2097152。
            -   B=1，Q\_N=10，Q\_S=2097152，D=512，KV\_N=1，KV\_S=2097152。

        -   query、key、value输入类型包含int8时，D轴需要32对齐；输入类型全为`float16`、`bfloat16`时，D轴需16对齐。
        -   D轴限制：
            -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：query、key、value输入类型包含`int8`时，D轴需要32对齐；query、key、value或attentionOut类型包含`int4`时，D轴需要64对齐；输入类型全为`float16`、`bfloat16`时，D轴需16对齐。

    -   actual\_seq\_qlen：
    
        <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于query中对应batch的Sequence Length。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。当query的input\_layout为TND/NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   actual\_seq\_kvlen：
    
        <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的input\_layout为TND/NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   参数sparse\_mode当前仅支持值为0、1、2、3、4的场景，取其它值时会报错。
        
        -   sparse\_mode=0时，atten\_mask如果为None，则忽略入参pre\_tokens、next\_tokens（内部赋值为INT\_MAX）。
        -   sparse\_mode=2、3、4时，atten\_mask的shape需要为\(S, S\)或\(1, S, S\)或\(1, 1, S, S\)，其中S的值需要固定为2048，且需要用户保证传入的atten\_mask为下三角，不传入atten\_mask或者传入的shape不正确报错。        
        -   sparse\_mode=1、2、3的场景忽略入参pre\_tokens、next\_tokens并按照相关规则赋值。

    -   page attention场景：
        -   page attention的使能必要条件是block\_table存在且有效，同时key、value是按照block\_table中的索引在一片连续内存中排布，支持key、value数据类型为`float16`、`bfloat16`。在该场景下key、value的input\_layout参数无效。block\_table中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
        -   block\_size是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，block\_size最小为128，最大为512，且要求是128的倍数。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
    
        -   page attention场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且KV\_N\*D超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小KV\_N）或调整kv cache排布格式为（blocknum, KV\_N, blocksize, D）解决。当query的input\_layout为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV\_N, blocksize, D）两种格式，当query的input\_layout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actual\_seq\_kvlen和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
        -   page attention不支持伪量化场景，不支持tensorlist场景。
        -   page attention场景下，必须传入actual\_seq\_kvlen。
        -   page attention场景下，block\_table必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actual\_seq\_kvlen对应的block数量）。
        -   page attention场景下，支持两种格式和float32/bfloat16，不支持输入query为int8的场景。
        -   page attention使能场景下，以下场景输入需满足KV\_S\>=maxBlockNumPerSeq\*blockSize：
            -   传入atten\_mask时，如mask shape为（B, 1, Q\_S, KV\_S）。
            -   传入pse\_shift时，如pse\_shift shape为（B, Q\_N, Q\_S, KV\_S）。

    -   入参quant\_scale\_out和quant\_offset\_out支持pertensor、perchannel量化，支持float32、bfloat16类型。若传入quant\_offset\_out，需保证其类型和shape信息与quant\_scale\_out一致。当输入为bfloat16时，同时支持float32和bfloat16，否则仅支持float32。perchannel场景下，当输出layout为BSH时，要求quant\_scale\_out所有维度的乘积等于H；其他layout要求乘积等于Q\_N\*D。当输出layout为BSH时，quant\_scale\_out shape建议传入\(1, 1, H\)或\(H,\)；当输出layout为BNSD时，建议传入\(1, Q\_N, 1, D\)或\(Q\_N, D\)；当输出为BSND时，建议传入\(1, 1, Q\_N, D\)或\(Q\_N, D)。
    -   输出为int8，quant\_scale\_out和quant\_offset\_out为perchannel时，暂不支持Ring Attention或者D非32Byte对齐的场景。
    -   输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。
    -   pse\_shift功能使用限制如下：
        
        -   支持query数据类型为float16、bfloat16、int8场景下使用该功能。
        -   query、key、value数据类型为float16且pse\_shift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
        -   Q\_S需大于等于query的S长度，KV\_S需大于等于key的S长度。

    -   输出为int8，入参quant\_offset\_out传入非None和非空tensor值，并且sparse\_mode、pre\_tokens和next\_tokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截：
        -   sparse\_mode=0，atten\_mask如果非None，每个batch actual\_seq\_qlen-actual\_seq\_kvlen-pre\_tokens\>0或next\_tokens<0时，满足拦截条件。
        -   sparse\_mode=1或2，不会出现满足拦截条件的情况。
        -   sparse\_mode=3，每个batch actual\_seq\_kvlen-actual\_seq\_qlen<0，满足拦截条件。
        -   sparse\_mode=4，pre\_tokens<0或每个batch next\_tokens+actual\_seq\_kvlen-actual\_seq\_qlen<0时，满足拦截条件。

    -   kv伪量化参数分离：
        -   当伪量化参数和KV分离量化参数同时传入时，以KV分离量化参数为准。     
        -   key\_quant\_mode和value\_quant\_mode取值需要保持一致。
        -   dequant\_scale\_key和dequant\_scale\_value要么都为空，要么都不为空；dequant\_offset\_key和dequant\_offset\_value要么都为空，要么都不为空。
        -   dequant\_scale\_key和dequant\_scale\_value都不为空时，其shape需要保持一致；dequant\_offset\_key和dequant\_offset\_value都不为空时，其shape需要保持一致。  
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：   
            -   仅支持pertoken和perchannel模式，pertoken模式下要求两个参数的shape均为\(B, KV\_S\)，数据类型固定为float32；perchannel模式下要求两个参数的shape为（KV\_N, D），\(KV\_N, D\)，\(H\)，数据类型固定为bfloat16,H为KV\_N*D。
            -   dequant\_scale\_key与dequant\_scale\_value非空场景，要求query的s小于等于16；要求query的dtype为bfloat16，key、value的dtype为int8，输出的dtype为bfloat16；不支持tensorlist、page attention特性。
        
        -   管理scale/offset的量化模式如下：
        
            > [!NOTE]   
            > 注意scale、offset具体指dequant\_scale\_key、dequant\_scale\_key、dequant\_offset\_value、dequant\_offset\_value参数。

            <a name="zh-cn_topic_0000001832267082_table3276159203213"></a>
            <table><thead align="left"><tr id="zh-cn_topic_0000001832267082_row192767598320"><th class="cellrowborder" valign="top" width="16.950000000000003%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000001832267082_p19276135910323"><a name="zh-cn_topic_0000001832267082_p19276135910323"></a><a name="zh-cn_topic_0000001832267082_p19276135910323"></a>量化模式</p>
            </th>
            <th class="cellrowborder" valign="top" width="23.09%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000001832267082_p1627615594327"><a name="zh-cn_topic_0000001832267082_p1627615594327"></a><a name="zh-cn_topic_0000001832267082_p1627615594327"></a>该场景下scale和offset条件</p>
            </th>
            <th class="cellrowborder" valign="top" width="46.660000000000004%" id="mcps1.1.5.1.3"><p id="zh-cn_topic_0000001832267082_p17276195963213"><a name="zh-cn_topic_0000001832267082_p17276195963213"></a><a name="zh-cn_topic_0000001832267082_p17276195963213"></a>该场景下key和value条件</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.3%" id="mcps1.1.5.1.4"><p id="zh-cn_topic_0000001832267082_p227695933219"><a name="zh-cn_topic_0000001832267082_p227695933219"></a><a name="zh-cn_topic_0000001832267082_p227695933219"></a>支持的产品</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="zh-cn_topic_0000001832267082_row172761159123213"><td class="cellrowborder" valign="top" width="16.950000000000003%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p14277165915327"><a name="zh-cn_topic_0000001832267082_p14277165915327"></a><a name="zh-cn_topic_0000001832267082_p14277165915327"></a>perchannel模式</p>
            </td>
            <td class="cellrowborder" valign="top" width="23.09%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p113847501392"><a name="zh-cn_topic_0000001832267082_p113847501392"></a><a name="zh-cn_topic_0000001832267082_p113847501392"></a>两个参数shape支持(KV_N, 1, D)，(KV_N, D)，(H)，(1, KV_N, 1, D)，(1, KV_N, D)，(1, H)数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" width="46.660000000000004%" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul2277759183216"></a><a name="zh-cn_topic_0000001832267082_ul2277759183216"></a><ul id="zh-cn_topic_0000001832267082_ul2277759183216"><li><span id="zh-cn_topic_0000001832267082_ph112776597327"><a name="zh-cn_topic_0000001832267082_ph112776597327"></a><a name="zh-cn_topic_0000001832267082_ph112776597327"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_19"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_19"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_19"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>：仅支持key、value数据类型为int8。</li><li><span id="zh-cn_topic_0000001832267082_ph327717592325"><a name="zh-cn_topic_0000001832267082_ph327717592325"></a><a name="zh-cn_topic_0000001832267082_ph327717592325"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_19"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_19"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_19"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：仅支持key、value数据类型为int8。</li></ul>
            </td>
            <td class="cellrowborder" rowspan="2" valign="top" width="13.3%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000001832267082_ul7277159143219"></a><a name="zh-cn_topic_0000001832267082_ul7277159143219"></a><ul id="zh-cn_topic_0000001832267082_ul7277159143219"><li><span id="zh-cn_topic_0000001832267082_ph527775920323"><a name="zh-cn_topic_0000001832267082_ph527775920323"></a><a name="zh-cn_topic_0000001832267082_ph527775920323"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_20"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_20"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_20"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></li><li><span id="zh-cn_topic_0000001832267082_ph22772059103210"><a name="zh-cn_topic_0000001832267082_ph22772059103210"></a><a name="zh-cn_topic_0000001832267082_ph22772059103210"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_20"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_20"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_20"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row027816595321"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p19936325114616"><a name="zh-cn_topic_0000001832267082_p19936325114616"></a><a name="zh-cn_topic_0000001832267082_p19936325114616"></a>pertoken模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p39361225144616"><a name="zh-cn_topic_0000001832267082_p39361225144616"></a><a name="zh-cn_topic_0000001832267082_p39361225144616"></a>两个参数的shape均为(B, KV_S)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul7475108135214"></a><a name="zh-cn_topic_0000001832267082_ul7475108135214"></a><ul id="zh-cn_topic_0000001832267082_ul7475108135214"><li><span id="zh-cn_topic_0000001832267082_ph1947698135214"><a name="zh-cn_topic_0000001832267082_ph1947698135214"></a><a name="zh-cn_topic_0000001832267082_ph1947698135214"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_21"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_21"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_21"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>：仅支持key、value数据类型为int8。</li><li><span id="zh-cn_topic_0000001832267082_ph12476287523"><a name="zh-cn_topic_0000001832267082_ph12476287523"></a><a name="zh-cn_topic_0000001832267082_ph12476287523"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_21"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_21"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_21"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：仅支持key、value数据类型为int8。</li></ul>
            </td>
            </tr>
            </tbody>
            </table>
    
-   **当Q\_S等于1时：**
    -   query、key、value输入，功能使用限制如下：
        -   支持B轴小于等于65536，支持N轴小于等于256，支持S轴小于等于262144，支持D轴小于等于512。
        -   query、key、value输入类型均为int8的场景暂不支持。
        -   在int4（int32）伪量化场景下，PyTorch入图调用仅支持KV int4拼接成int32输入（建议通过dynamicQuant生成int4格式的数据，因为dynamicQuant就是一个int32包括8个int4）。
        -   在int4（int32）伪量化场景下，若KV int4拼接成int32输入，那么KV的N、D或者H是实际值的八分之一。并且，int4伪量化仅支持D 64对齐（int32支持D 8对齐）。

    -   actual\_seq\_qlen：
    
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当query的input\_layout不为TND时，Q\_S为1时该参数无效。当query的input\_layout为TND/TND\_NTD时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   actual\_seq\_kvlen：
    
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的input\_layout为TND/TND\_NTD时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   page attention场景：
        -   使能必要条件是block\_table存在且有效，同时key、value是按照block\_table中的索引在一片连续内存中排布，在该场景下key、value的input\_layout参数无效。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
            -   支持key、value数据类型为float16、bfloat16、int8。
            -   不支持Q为`bfloat16`、float16、key、value为int4（int32）的场景。
    
        -   该场景下，block\_size是用户自定义的参数，该参数的取值会影响page attention的性能。key、value输入类型为float16、bfloat16时需要16对齐，key、value输入类型为int8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
        -   参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。
        -   page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actual\_seq\_kvlen对应的block数量）。
        -   page attention场景下，当query的input\_layout为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV\_N, blocksize, D）两种格式，当query的input\_layout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actual\_seq\_kvlen和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
        -   page attention场景下，kv cache排布为（blocknum, KV\_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV\_N, blocksize, D）格式。
        -   page attention使能场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且numKvHeads \* headDim 超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 numKvHeads）或调整kv cache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
        -   page attention场景的参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。

    -   kv伪量化参数分离：
        -   除了key\_quant\_mode为0并且value\_quant\_mode为1的场景外，key\_quant\_mode和value\_quant\_mode取值需要保持一致。   
        -   dequant\_scale\_key和dequant\_scale\_value要么都为空，要么都不为空；dequant\_offset\_key和dequant\_offset\_value要么都为空，要么都不为空。
        -   dequant\_scale\_key和dequant\_scale\_value都不为空时，除了key\_quant\_mode为0并且value\_quant\_mode为1的场景外，其shape需要保持一致；dequant\_offset\_key和dequant\_offset\_value都不为空时，除了key\_quant\_mode为0并且value\_quant\_mode为1的场景外，其shape需要保持一致。
        -   int4（int32）伪量化场景不支持后量化。
        -   管理scale/offset的量化模式如下：
    
            > [!NOTE]   
            > 注意scale、offset两个参数指dequant\_scale\_key、dequant\_scale\_key、dequant\_offset\_value、dequant\_offset\_value。
    
            <a name="zh-cn_topic_0000001832267082_table4401182238"></a>
            <table><thead align="left"><tr id="zh-cn_topic_0000001832267082_row124112817233"><th class="cellrowborder" valign="top" width="16.950000000000003%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000001832267082_p341780235"><a name="zh-cn_topic_0000001832267082_p341780235"></a><a name="zh-cn_topic_0000001832267082_p341780235"></a>量化模式</p>
            </th>
            <th class="cellrowborder" valign="top" width="23.09%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000001832267082_p144118852314"><a name="zh-cn_topic_0000001832267082_p144118852314"></a><a name="zh-cn_topic_0000001832267082_p144118852314"></a>该场景下scale和offset条件</p>
            </th>
            <th class="cellrowborder" valign="top" width="46.660000000000004%" id="mcps1.1.5.1.3"><p id="zh-cn_topic_0000001832267082_p123481541027"><a name="zh-cn_topic_0000001832267082_p123481541027"></a><a name="zh-cn_topic_0000001832267082_p123481541027"></a>该场景下key和value条件</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.3%" id="mcps1.1.5.1.4"><p id="zh-cn_topic_0000001832267082_p147001940151615"><a name="zh-cn_topic_0000001832267082_p147001940151615"></a><a name="zh-cn_topic_0000001832267082_p147001940151615"></a>支持的产品</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="zh-cn_topic_0000001832267082_row10411185232"><td class="cellrowborder" valign="top" width="16.950000000000003%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p154120882315"><a name="zh-cn_topic_0000001832267082_p154120882315"></a><a name="zh-cn_topic_0000001832267082_p154120882315"></a>perchannel模式</p>
            </td>
            <td class="cellrowborder" valign="top" width="23.09%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p21391147716"><a name="zh-cn_topic_0000001832267082_p21391147716"></a><a name="zh-cn_topic_0000001832267082_p21391147716"></a>两个参数shape支持(1, KV_N, 1, D)，(1, KV_N, D)，(1, H)，数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" width="46.660000000000004%" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul0858154962714"></a><a name="zh-cn_topic_0000001832267082_ul0858154962714"></a><ul id="zh-cn_topic_0000001832267082_ul0858154962714"><li><span id="zh-cn_topic_0000001832267082_ph1163117183317"><a name="zh-cn_topic_0000001832267082_ph1163117183317"></a><a name="zh-cn_topic_0000001832267082_ph1163117183317"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_26"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_26"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_26"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>：当key、value数据类型为int4（int32）或int8时支持。</li><li><span id="zh-cn_topic_0000001832267082_ph10252223193315"><a name="zh-cn_topic_0000001832267082_ph10252223193315"></a><a name="zh-cn_topic_0000001832267082_ph10252223193315"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_26"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_26"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_26"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int4（int32）或int8时支持。</li></ul>
            </td>
            <td class="cellrowborder" rowspan="9" valign="top" width="13.3%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000001832267082_ul15575120101720"></a><a name="zh-cn_topic_0000001832267082_ul15575120101720"></a><ul id="zh-cn_topic_0000001832267082_ul15575120101720"><li><span id="zh-cn_topic_0000001832267082_ph112662491714"><a name="zh-cn_topic_0000001832267082_ph112662491714"></a><a name="zh-cn_topic_0000001832267082_ph112662491714"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_27"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_27"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_27"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></li><li><span id="zh-cn_topic_0000001832267082_ph1290711610176"><a name="zh-cn_topic_0000001832267082_ph1290711610176"></a><a name="zh-cn_topic_0000001832267082_ph1290711610176"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_27"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_27"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_27"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></li></ul>
            <p id="zh-cn_topic_0000001832267082_p13700840111618"><a name="zh-cn_topic_0000001832267082_p13700840111618"></a><a name="zh-cn_topic_0000001832267082_p13700840111618"></a></p>
            <p id="zh-cn_topic_0000001832267082_p19700134031612"><a name="zh-cn_topic_0000001832267082_p19700134031612"></a><a name="zh-cn_topic_0000001832267082_p19700134031612"></a></p>
            <p id="zh-cn_topic_0000001832267082_p7700740201616"><a name="zh-cn_topic_0000001832267082_p7700740201616"></a><a name="zh-cn_topic_0000001832267082_p7700740201616"></a></p>
            <p id="zh-cn_topic_0000001832267082_p4700040131614"><a name="zh-cn_topic_0000001832267082_p4700040131614"></a><a name="zh-cn_topic_0000001832267082_p4700040131614"></a></p>
            <p id="zh-cn_topic_0000001832267082_p107002403160"><a name="zh-cn_topic_0000001832267082_p107002403160"></a><a name="zh-cn_topic_0000001832267082_p107002403160"></a></p>
            <p id="zh-cn_topic_0000001832267082_p1670174011616"><a name="zh-cn_topic_0000001832267082_p1670174011616"></a><a name="zh-cn_topic_0000001832267082_p1670174011616"></a></p>
            <p id="zh-cn_topic_0000001832267082_p67011840191613"><a name="zh-cn_topic_0000001832267082_p67011840191613"></a><a name="zh-cn_topic_0000001832267082_p67011840191613"></a></p>
            <p id="zh-cn_topic_0000001832267082_p1870113406160"><a name="zh-cn_topic_0000001832267082_p1870113406160"></a><a name="zh-cn_topic_0000001832267082_p1870113406160"></a></p>
            <p id="zh-cn_topic_0000001832267082_p107011540141611"><a name="zh-cn_topic_0000001832267082_p107011540141611"></a><a name="zh-cn_topic_0000001832267082_p107011540141611"></a></p>
            <p id="zh-cn_topic_0000001832267082_p070174061612"><a name="zh-cn_topic_0000001832267082_p070174061612"></a><a name="zh-cn_topic_0000001832267082_p070174061612"></a></p>
            <p id="zh-cn_topic_0000001832267082_p970174013162"><a name="zh-cn_topic_0000001832267082_p970174013162"></a><a name="zh-cn_topic_0000001832267082_p970174013162"></a></p>
            <p id="zh-cn_topic_0000001832267082_p18701134016166"><a name="zh-cn_topic_0000001832267082_p18701134016166"></a><a name="zh-cn_topic_0000001832267082_p18701134016166"></a></p>
            <p id="zh-cn_topic_0000001832267082_p107011040191616"><a name="zh-cn_topic_0000001832267082_p107011040191616"></a><a name="zh-cn_topic_0000001832267082_p107011040191616"></a></p>
            <p id="zh-cn_topic_0000001832267082_p107072401161"><a name="zh-cn_topic_0000001832267082_p107072401161"></a><a name="zh-cn_topic_0000001832267082_p107072401161"></a></p>
            <p id="zh-cn_topic_0000001832267082_p87072401163"><a name="zh-cn_topic_0000001832267082_p87072401163"></a><a name="zh-cn_topic_0000001832267082_p87072401163"></a></p>
            <p id="zh-cn_topic_0000001832267082_p8707640151615"><a name="zh-cn_topic_0000001832267082_p8707640151615"></a><a name="zh-cn_topic_0000001832267082_p8707640151615"></a></p>
            <p id="zh-cn_topic_0000001832267082_p1870774011617"><a name="zh-cn_topic_0000001832267082_p1870774011617"></a><a name="zh-cn_topic_0000001832267082_p1870774011617"></a></p>
            <p id="zh-cn_topic_0000001832267082_p12707174013166"><a name="zh-cn_topic_0000001832267082_p12707174013166"></a><a name="zh-cn_topic_0000001832267082_p12707174013166"></a></p>
            <p id="zh-cn_topic_0000001832267082_p14707184011619"><a name="zh-cn_topic_0000001832267082_p14707184011619"></a><a name="zh-cn_topic_0000001832267082_p14707184011619"></a></p>
            <p id="zh-cn_topic_0000001832267082_p15707540141620"><a name="zh-cn_topic_0000001832267082_p15707540141620"></a><a name="zh-cn_topic_0000001832267082_p15707540141620"></a></p>
            <p id="zh-cn_topic_0000001832267082_p2707164021613"><a name="zh-cn_topic_0000001832267082_p2707164021613"></a><a name="zh-cn_topic_0000001832267082_p2707164021613"></a></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row84115813237"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p10419882318"><a name="zh-cn_topic_0000001832267082_p10419882318"></a><a name="zh-cn_topic_0000001832267082_p10419882318"></a>pertensor模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p04120872317"><a name="zh-cn_topic_0000001832267082_p04120872317"></a><a name="zh-cn_topic_0000001832267082_p04120872317"></a>两个参数的shape均为(1,)，数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul19978121525716"></a><a name="zh-cn_topic_0000001832267082_ul19978121525716"></a><ul id="zh-cn_topic_0000001832267082_ul19978121525716"><li><span id="zh-cn_topic_0000001832267082_ph19978111585717"><a name="zh-cn_topic_0000001832267082_ph19978111585717"></a><a name="zh-cn_topic_0000001832267082_ph19978111585717"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_28"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_28"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_28"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>：当key、value数据类型为int8时支持。</li><li><span id="zh-cn_topic_0000001832267082_ph13978111545713"><a name="zh-cn_topic_0000001832267082_ph13978111545713"></a><a name="zh-cn_topic_0000001832267082_ph13978111545713"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_28"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_28"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_28"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int8时支持。</li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row1341138172312"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p134114862314"><a name="zh-cn_topic_0000001832267082_p134114862314"></a><a name="zh-cn_topic_0000001832267082_p134114862314"></a>pertoken模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p1037135185914"><a name="zh-cn_topic_0000001832267082_p1037135185914"></a><a name="zh-cn_topic_0000001832267082_p1037135185914"></a>两个参数的shape均为(1, B, KV_S)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p174417474211"><a name="zh-cn_topic_0000001832267082_p174417474211"></a><a name="zh-cn_topic_0000001832267082_p174417474211"></a>key、value数据类型为int4（int32）或int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row12620173672311"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p166201636132311"><a name="zh-cn_topic_0000001832267082_p166201636132311"></a><a name="zh-cn_topic_0000001832267082_p166201636132311"></a>pertensor叠加perhead模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p662110362235"><a name="zh-cn_topic_0000001832267082_p662110362235"></a><a name="zh-cn_topic_0000001832267082_p662110362235"></a>两个参数的shape均为(KV_N,)，数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul519618222020"></a><a name="zh-cn_topic_0000001832267082_ul519618222020"></a><ul id="zh-cn_topic_0000001832267082_ul519618222020"><li><span id="zh-cn_topic_0000001832267082_ph121961922601"><a name="zh-cn_topic_0000001832267082_ph121961922601"></a><a name="zh-cn_topic_0000001832267082_ph121961922601"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_29"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_29"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_29"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>：当key、value数据类型为int8时支持。</li><li><span id="zh-cn_topic_0000001832267082_ph11961022801"><a name="zh-cn_topic_0000001832267082_ph11961022801"></a><a name="zh-cn_topic_0000001832267082_ph11961022801"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_29"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_29"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_29"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int8时支持。</li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row136211336192318"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p22659468119"><a name="zh-cn_topic_0000001832267082_p22659468119"></a><a name="zh-cn_topic_0000001832267082_p22659468119"></a>pertoken叠加perhead模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p116212367239"><a name="zh-cn_topic_0000001832267082_p116212367239"></a><a name="zh-cn_topic_0000001832267082_p116212367239"></a>两个参数的shape均为(B, KV_N, KV_S)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p162285131891"><a name="zh-cn_topic_0000001832267082_p162285131891"></a><a name="zh-cn_topic_0000001832267082_p162285131891"></a>key、value数据类型为int4（int32）或int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row1037716581001"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p1757135414112"><a name="zh-cn_topic_0000001832267082_p1757135414112"></a><a name="zh-cn_topic_0000001832267082_p1757135414112"></a>pertoken叠加使用page attention模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p1025316567221"><a name="zh-cn_topic_0000001832267082_p1025316567221"></a><a name="zh-cn_topic_0000001832267082_p1025316567221"></a>两个参数的shape均为(blocknum, blocksize)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p04417476217"><a name="zh-cn_topic_0000001832267082_p04417476217"></a><a name="zh-cn_topic_0000001832267082_p04417476217"></a>key、value数据类型为int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row15621736192312"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p1986911315215"><a name="zh-cn_topic_0000001832267082_p1986911315215"></a><a name="zh-cn_topic_0000001832267082_p1986911315215"></a>pertoken叠加per head并使用page attention模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p2621173692313"><a name="zh-cn_topic_0000001832267082_p2621173692313"></a><a name="zh-cn_topic_0000001832267082_p2621173692313"></a>两个参数的shape均为(blocknum, KV_N, blocksize)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p551893313915"><a name="zh-cn_topic_0000001832267082_p551893313915"></a><a name="zh-cn_topic_0000001832267082_p551893313915"></a>key、value数据类型为int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row915113171020"><td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p01521217025"><a name="zh-cn_topic_0000001832267082_p01521217025"></a><a name="zh-cn_topic_0000001832267082_p01521217025"></a>key支持perchannel叠加value支持pertoken模式</p>
            <p id="zh-cn_topic_0000001832267082_p74743213101"><a name="zh-cn_topic_0000001832267082_p74743213101"></a><a name="zh-cn_topic_0000001832267082_p74743213101"></a></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p1867213119113"><a name="zh-cn_topic_0000001832267082_p1867213119113"></a><a name="zh-cn_topic_0000001832267082_p1867213119113"></a>对于key支持perchannel，两个参数的shape可支持(1, KV_N, 1, D)、(1, KV_N, D)、(1, H)，且参数数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul1037202951112"></a><a name="zh-cn_topic_0000001832267082_ul1037202951112"></a><ul id="zh-cn_topic_0000001832267082_ul1037202951112"><li><span id="zh-cn_topic_0000001832267082_ph10271547141214"><a name="zh-cn_topic_0000001832267082_ph10271547141214"></a><a name="zh-cn_topic_0000001832267082_ph10271547141214"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_30"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_30"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_30"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>：当key、value数据类型为int4（int32）或int8时支持；当key和value的数据类型为int8时，仅支持query和输出的dtype为float16。</li><li><span id="zh-cn_topic_0000001832267082_ph427116472125"><a name="zh-cn_topic_0000001832267082_ph427116472125"></a><a name="zh-cn_topic_0000001832267082_ph427116472125"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_30"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_30"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_30"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int4（int32）或int8时支持；当key和value的数据类型为int8时，仅支持query和输出的dtype为float16。</li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row194748261012"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p1154111491113"><a name="zh-cn_topic_0000001832267082_p1154111491113"></a><a name="zh-cn_topic_0000001832267082_p1154111491113"></a>对于value支持pertoken，两个参数的shape均为(1, B, KV_S)并且数据类型固定为float32。</p>
            </td>
            </tr>
        </tbody>
            </table>
    
    -   pse\_shift功能使用限制如下：
        -   pse\_shift数据类型需与query数据类型保持一致。
        -   仅支持D轴对齐，即D轴可以被16整除。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import math
    # 生成随机数据, 并发送到npu
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    softmax_scale = 1/math.sqrt(128.0)
    actseqlen = [164]
    actseqlenkv = [1024]

    # 调用FIA算子
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, 
    actual_seq_qlen = actseqlen, actual_seq_kvlen = actseqlenkv,
    num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

    # 执行上述代码的输出out类似如下
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16)
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import math
    import torchair as tng

    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # 数据生成
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    softmax_scale = 1/math.sqrt(128.0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
        print("single op output with mask:", single_op[0], single_op[0].shape)
        print("graph output with mask:", graph_output[0], graph_output[0].shape)
    if __name__ == "__main__":
        MetaInfershape()

    # 执行上述代码的输出类似如下
    single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
    ```


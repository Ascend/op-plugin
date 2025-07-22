# torch\_npu.npu\_fused\_infer\_attention\_score<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

-   算子功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当Query矩阵的S为1，进入IncreFlashAttention分支，其余场景进入PromptFlashAttention分支。
-   计算公式：

    ![](./figures/zh-cn_formulaimage_0000001849899320.png)

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
torch_npu.npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None,  SymInt[]? actual_shared_prefix_len=None,Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, bool softmax_lse_flag=False, int key_antiquant_mode=0, int value_antiquant_mode=0) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。

-   query：Tensor类型，attention结构的Query输入，不支持非连续的Tensor，数据格式支持ND。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16。

-   key：Tensor类型，attention结构的Key输入，不支持非连续的Tensor，数据格式支持ND。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16、int8、int4（int32）。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16、int8、int4（int32）。

-   value：Tensor类型，attention结构的Value输入，不支持非连续的Tensor，数据格式支持ND。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16、int8、int4（int32）。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16、int8、int4（int32）。

- \*：代表其之前的变量是位置相关，需要按照顺序输入，必选；之后的变量是键值对赋值的，位置无关，可选（不输入会使用默认值）。

-   pse\_shift：Tensor类型，在attention结构内部的位置编码参数，数据类型支持float16、bfloat16，数据类型与query的数据类型需满足数据类型推导规则。不支持非连续的Tensor，数据格式支持ND。如不使用该功能时可传入None。
    -   Q\_S不为1，要求在pse\_shift为float16类型时，此时的query为float16或int8类型；而在pse\_shift为bfloat16类型时，要求此时的query为bfloat16类型。输入shape类型需为\(B, N, Q\_S, KV\_S\)或\(1, N, Q\_S, KV\_S\)，其中Q\_S为query的shape中的S，KV\_S为key和value的shape中的S。对于pse\_shift的KV\_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
    -   Q\_S为1，要求在pse\_shift为float16类型时，此时的query为float16类型；而在pse\_shift为bfloat16类型时，要求此时的query为bfloat16类型。输入shape类型需为\(B, N, 1, KV\_S\)或\(1, N, 1, KV\_S\)，其中N为num\_heads，KV\_S为key和value的shape中的S。对于pse\_shift的KV\_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。

-   atten\_mask：Tensor类型，对QK的结果进行mask，用于指示是否计算Token间的相关性，数据类型支持bool、int8和uint8。不支持非连续的Tensor，数据格式支持ND。如果不使用该功能可传入None。

    -   Q\_S不为1时建议shape输入\(Q\_S, KV\_S\)、\(B, Q\_S, KV\_S\)、\(1, Q\_S, KV\_S\)、\(B, 1, Q\_S, KV\_S\)、\(1, 1, Q\_S, KV\_S\)。
    -   Q\_S为1时建议shape输入\(B, KV\_S\)、\(B, 1, KV\_S\)、\(B, 1, 1, KV\_S\)。

    其中Q\_S为query的shape中的S，KV\_S为key和value的shape中的S，但如果Q\_S、KV\_S非16或32对齐，可以向上取到对齐的S。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- actual\_seq\_lengths：int类型数组，代表不同Batch中query的有效seqlen，数据类型支持int64。如果不指定seqlen可以传入None，表示和query的shape的s长度相同。

  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  该入参中每个batch的有效seqlen应该不大于query中对应batch的seqlen。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。当query的input\_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。

- actual\_seq\_lengths\_kv：int类型数组，代表不同Batch中key/value的有效seqlenKv，数据类型支持int64。如果不指定None，表示和key/value的shape的S长度相同。不同O\_S值有不同的约束，具体参见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- dequant\_scale1：Tensor类型，数据类型支持uint64、float32。数据格式支持ND，表示BMM1后面的反量化因子，支持per-tensor。如不使用该功能时传入None。

- quant\_scale1：Tensor类型，数据类型支持float32。数据格式支持ND，表示BMM2前面的量化因子，支持per-tensor。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- dequant\_scale2：Tensor类型，数据类型支持uint64、float32。数据格式支持ND，表示BMM2后面的反量化因子，支持per-tensor。如不使用该功能时传入None。

- quant\_scale2：Tensor类型，数据类型支持float32、bfloat16。数据格式支持ND，表示输出的量化因子，支持per-tensor、per-channel。当输入为bfloat16时，同时支持float32和bfloat16，否则仅支持float32。per-channel格式，当输出layout为BSH时，要求quant\_scale2所有维度的乘积等于H；其他layout要求乘积等于N\*D。建议输出layout为BSH时，quant\_scale2 shape传入\(1, 1, H\)或\(H,\)；输出为BNSD时，建议传入\(1, N, 1, D\)或\(N, D\)；输出为BSND时，建议传入\(1, 1, N, D\)或\(N, D\)。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- quant\_offset2：Tensor类型，数据类型支持float32、bfloat16。数据格式支持ND，表示输出的量化偏移，支持per-tensor、per-channel。若传入quant\_offset2，需保证其类型和shape信息与quantScale2一致。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- antiquant\_scale：Tensor类型，数据类型支持float16、bfloat16。数据格式支持ND，表示伪量化因子，支持per-tensor、per-channel，Q\_S为1时只支持per-channel，Q\_S大于等于2时只支持float16，如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- antiquant\_offset：Tensor类型，数据类型支持float16、bfloat16。数据格式支持ND，表示伪量化偏移，支持per-tensor、per-channel，Q\_S为1时只支持per-channel，Q\_S大于等于2时只支持float16，如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- block\_table：Tensor类型，数据类型支持int32。数据格式支持ND。表示PageAttention中KV存储使用的block映射表，如不使用该功能可传入None。

- query\_padding\_size：Tensor类型，数据类型支持int64。数据格式支持ND。表示Query中每个batch的数据是否右对齐，且右对齐的个数是多少。仅支持Q\_S大于1，其余场景该参数无效。用户不特意指定时可传入默认值None。

- kv\_padding\_size：Tensor类型，数据类型支持int64。数据格式支持ND。表示key、value中每个batch的数据是否右对齐，且右对齐的个数是多少。用户不特意指定时可传入默认值None。

-   key\_antiquant\_scale：Tensor类型。数据格式支持ND，kv伪量化参数分离时表示key的反量化因子。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16、float32。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16、float32。

- key\_antiquant\_offset：Tensor类型，数据类型支持float16、bfloat16、float32。数据格式支持ND，kv伪量化参数分离时表示key的反量化偏移。支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

-   value\_antiquant\_scale：Tensor类型，数据类型支持float16、bfloat16、float32。数据格式支持ND，kv伪量化参数分离时表示value的反量化因子。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16、float32。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16、float32。

- value\_antiquant\_offset：Tensor类型，数据类型支持float16、bfloat16、float32。数据格式支持ND，kv伪量化参数分离时表示value的反量化偏移，支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset。如不使用该功能时可传入None，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- key\_shared\_prefix：Tensor类型，attention结构中Key的前缀部分参数，数据类型支持float16、bfloat16、int8，不支持非连续的Tensor，数据格式支持ND。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- value\_shared\_prefix：Tensor类型，attention结构中Value的前缀部分输入，数据类型支持float16、bfloat16、int8，不支持非连续的Tensor，数据格式支持ND。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

- actual\_shared\_prefix\_len：int型数组，代表key\_shared\_prefix/value\_shared\_prefix的有效Sequence Length。数据类型支持：int64。如果不指定seqlen可以传入None，表示和key\_shared\_prefix/value\_shared\_prefix的s长度相同。限制：该入参中的有效Sequence Length应该不大于key\_shared\_prefix/value\_shared\_prefix中的Sequence Length。

- query\_rope：Tensor类型，表示MLA（Multi-head Latent Attention）结构中的query的rope信息，数据类型支持float16、bfloat16，不支持非连续的Tensor，数据格式支持ND。仅支持Q\_S等于1-16，其余场景该参数无效。

- key\_rope：Tensor类型，表示MLA（Multi-head Latent Attention）结构中的key的rope信息，数据类型支持float16、bfloat16，不支持非连续的Tensor，数据格式支持ND。仅支持Q\_S等于1-16，其余场景该参数无效。

- key\_rope\_antiquant\_scale：Tensor类型，**预留参数，暂未使用，使用默认值即可。**

- num\_heads：整型，代表query的head个数，数据类型支持int64，在BNSD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。

- scale：浮点型，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持float。数据类型与query的数据类型需满足数据类型推导规则。用户不特意指定时可传入默认值1.0。

- pre\_tokens：整型，用于稀疏计算，表示attention需要和前几个Token计算关联，数据类型支持int64。用户不特意指定时可传入默认值2147483647，Q\_S为1时该参数无效。

- next\_tokens：整型，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持int64。用户不特意指定时可传入默认值2147483647，Q\_S为1时该参数无效。

- input\_layout：字符串类型，用于标识输入query、key、value的数据排布格式，用户不特意指定时可传入默认值"BSH"。

  >**说明：**<br> 
  >注意排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出output的格式，算子内部会进行layout转换。

  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持BSH、BSND、BNSD、BNSD\_BSND（输入为BNSD时，输出格式为BSND，仅支持Q\_S大于1）、BSH\_NBSD、BSND\_NBSD、BNSD\_NBSD（输出格式为NBSD时，仅支持Q\_S大于1且小于等于16）、TND、TND\_NTD、NTD\_TND（TND相关场景综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)）。

  其中BNSD\_BSND含义指当输入为BNSD，输出格式为BSND，仅支持Q\_S大于1。

- num\_key\_value\_heads：整型，代表key、value中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，数据类型支持int64。用户不特意指定时可传入默认值0，表示key/value和query的head个数相等，需要满足num\_heads整除num\_key\_value\_heads，num\_heads与num\_key\_value\_heads的比值不能大于64。在BSND、BNSD、BNSD\_BSND（仅支持Q\_S大于1）场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常。

-   sparse\_mode：整型，表示sparse的模式。数据类型支持int64。Q\_S为1且不带rope输入时该参数无效。input\_layout为TND、TND\_NTD、NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
    -   sparse\_mode为0时，代表defaultMask模式，如果atten\_mask未传入则不做mask操作，忽略pre\_tokens和next\_tokens（内部赋值为INT\_MAX）；如果传入，则需要传入完整的atten\_mask矩阵（S1\*S2），表示pre\_tokens和next\_tokens之间的部分需要计算。
    -   sparse\_mode为1时，代表allMask，必须传入完整的attenmask矩阵（S1\*S2）。
    -   sparse\_mode为2时，代表leftUpCausal模式的mask，需要传入优化后的atten\_mask矩阵（2048\*2048）。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的atten\_mask矩阵（2048\*2048）。
    -   sparse\_mode为4时，代表band模式的mask，需要传入优化后的atten\_mask矩阵（2048\*2048）。
    -   sparse\_mode为5、6、7、8时，分别代表prefix、global、dilated、block\_local，均暂不支持。用户不特意指定时可传入默认值0。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

-   inner\_precise：整型，一共4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。数据类型支持int64。Q\_S\>1时，sparse\_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效；Q\_S为1时该参数仅支持inner\_precise为0和1。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    -   inner\_precise为0时，代表开启高精度模式，且不做行无效修正。
    -   inner\_precise为1时，代表高性能模式，且不做行无效修正。
    -   inner\_precise为2时，代表开启高精度模式，且做行无效修正。
    -   inner\_precise为3时，代表高性能模式，且做行无效修正。

    >**说明：**<br> 
    >bfloat16和int8不区分高精度和高性能，行无效修正对float16、bfloat16和int8均生效。当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。

- block\_size：整型，PageAttention中KV存储每个block中最大的token个数，默认为0，数据类型支持int64。

- antiquant\_mode：整型，表示伪量化方式，传入0时表示为per-channel（per-channel包含per-tensor），传入1时表示per-token。用户不特意指定时可传入默认值0。

  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：Q\_S大于等于2时该参数无效；Q\_S等于1时传入0和1之外的其他值会执行异常。

- softmax\_lse\_flag：布尔型，表示是否输出softmax\_lse，支持S轴外切（增加输出）。true表示输出softmax\_lse，false表示不输出；用户不特意指定时可传入默认值false。

-   key\_antiquant\_mode：整型，表示key的伪量化方式。用户不特意指定时可传入默认值0，取值除了key\_antiquant\_mode为0并且value\_antiquant\_mode为1的场景外，需要与value\_antiquant\_mode一致。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

    <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：Q\_S大于等于2时仅支持传入值为0、1，Q\_S等于1时支持取值0、1、2、3、4、5。

    -   key\_antiquant\_mode为0时，代表per-channel模式（per-channel包含per-tensor）。
    -   key\_antiquant\_mode为1时，代表per-token模式。
    -   key\_antiquant\_mode为2时，代表per-tensor叠加per-head模式。
    -   key\_antiquant\_mode为3时，代表per-token叠加per-head模式。
    -   key\_antiquant\_mode为4时，代表per-token叠加使用page attention模式管理scale/offset模式。
    -   key\_antiquant\_mode为5时，代表per-token叠加per head并使用page attention模式管理scale/offset模式。

- value\_antiquant\_mode：整型，表示value的伪量化方式，模式编号与key\_antiquant\_mode一致。用户不特意指定时可传入默认值0，取值除了key\_antiquant\_mode为0并且value\_antiquant\_mode为1的场景外，需要与key\_antiquant\_mode一致。综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。

  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：Q\_S大于等于2时仅支持传入值为0、1；Q\_S等于1时支持取值0、1、2、3、4、5。

## 输出说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   attention\_out：Tensor类型，公式中的输出，数据类型支持float16、bfloat16、int8。数据格式支持ND。限制：该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
-   softmaxLse：Tensor类型，ring attention算法对query乘key的结果，先取max得到softmax\_max。query乘key的结果减去softmax\_max，再取exp，最后取sum，得到softmax\_sum，最后对softmax\_sum取log，再加上softmax\_max得到的结果。数据类型支持float32，softmax\_lse\_flag为True时，一般情况下，输出shape为\(B, N, Q\_S, 1\)的Tensor，当input\_layout为TND/NTD\_TND时，输出shape为\(T,N,1\)的Tensor；softmax\_lse\_flag为False时，则输出shape为\[1\]的值为0的Tensor。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则填充全零的对应shape的输出（填充attention\_out）。attention\_out为空Tensor时，框架会处理。
-   参数key、value中对应tensor的shape需要完全一致；非连续场景下key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。
-   int8量化相关入参数量与输入、输出数据格式的综合限制：
    -   输出为int8的场景：入参dequant\_scale1、quant\_scale1、dequant\_scale2、quant\_scale2需要同时存在，quant\_offset2可选，不传时默认为0。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：输入为int8。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：输入为int8。

    -   输出为float16的场景：入参dequant\_scale1、quant\_scale1、dequant\_scale2需要同时存在，若存在入参quant\_offset2或quant\_scale2（即不为None），则报错并返回。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：输入为int8。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：输入为int8。

    -   输入全为float16或bfloat16，输出为int8的场景：入参quant\_scale2需存在，quant\_offset2可选，不传时默认为0，若存在入参dequant\_scale1或quant\_scale1或dequant\_scale2（即不为None），则报错并返回。
    -   入参quant\_offset2和quant\_scale2支持per-tensor或per-channel格式，数据类型支持float32、bfloat16。

-   antiquant\_scale和antiquant\_offset参数约束：
    -   支持per-channel、per-tensor和per-token三种模式：

        -   per-channel模式：两个参数BNSD场景下shape为\(2, N, 1, D\)，BSND场景下shape为\(2, N, D\)，BSH场景下shape为\(2, H\)，N为num\_key\_value\_heads。参数数据类型和query数据类型相同，antiquant\_mode置0，当key、value数据类型为int8时支持。
        -   per-tensor模式：两个参数的shape均为\(2,\)，数据类型和query数据类型相同，antiquant\_mode置0，当key、value数据类型为int8时支持。

        -   per-token模式：两个参数的shape均为\(2, B, S\)，数据类型固定为float32，antiquant\_mode置1，当key、value数据类型为int8时支持。

        算子运行在何种模式根据参数的shape进行判断，dim为1时运行per-tensor模式，否则运行per-channel模式。

    -   支持对称量化和非对称量化：
        -   非对称量化模式下，antiquant\_scale和antiquant\_offset参数需同时存在。
        -   对称量化模式下，antiquant\_offset可以为空（即None）；当antiquant\_offset参数为空时，执行对称量化，否则执行非对称量化。

-   query\_rope和key\_rope参数约束：
    -   query\_rope的数据类型、数据格式与query一致。
    -   key\_rope的数据类型、数据格式与key一致。
    -   query\_rope和key\_rope要求同时配置或同时不配置，不支持只配置其中一个。
    -   当query\_rope和key\_rope非空时，支持如下特性：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：query的d只支持512、128；
        -   当query的d等于512时：
            -   sparse：Q\_S等于1时只支持sparse=0且不传mask，Q\_S大于1时只支持sparse=3且传入mask；
            -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>约束如下：
                - query\_rope配置时要求query的s为1-16、n为32、64、128，query\_rope的shape中d为64，其余维度与query一致；
                - key\_rope配置时要求key的n为1，d为512，keyRope的shape中d为64，其余维度与key一致；
                - 支持key、value、keyRope的input\_layout格式为ND或NZ。当input\_layout为NZ时，输入参数key和value的格式为\[blockNum, N, D/16, blockSize, 16\]；
                - input\_layout形状支持BSH、BSND、BNSD、BNSD\_NBSD、BSND\_NBSD、BSH\_NBSD、TND、TND\_NTD，当数据格式为NZ时input\_layout不支持BNSD、BNSD\_NBSD。
                - 该场景下，必须开启PageAttention，此时block\_size支持16、128，其中数据格式为NZ时block\_size不支持配置16。
                - 不支持开启SoftMaxLse、左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。

        -   当query的d等于128时：
            -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>约束如下：
                - inputLayout：TND、NTD\_TND。
                - query\_rope配置时要求query\_rope的shape中d为64，其余维度与query一致。
                - keyRope配置时要求keyRope的shape中d为64，其余维度与key一致。
                - 不支持左padding、tensorlist、pse、page attention、prefix、伪量化、全量化、后量化。
                - 其余约束同TND、NTD\_TND场景下的综合限制保持一致。
    
-   TND、TND\_NTD、NTD\_TND场景下query、key、value输入的综合限制：
    -   T小于等于1M;
    -   sparse模式仅支持sparse=0且不传mask，或sparse=3且传入mask；
    -   actualSeqLengths和actualSeqLengthsKv必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值；
    -   当query的d等于512时：
        -   支持TND、TND\_NTD;
        -   必须开启page attention，此时actualSeqLengthsKv长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV\_S；
        -   支持query每个batch的s为1-16；
        -   要求query的n为32/64/128，key、value的n为1；
        -   要求query\_rope和keyRope不等于空，query\_rope和keyRope的d为64；
        -   不支持开启SoftMaxLse、左padding、tensorlist、pse、prefix、伪量化、全量化、后量化。
    
    -   当query的d不等于512时：
        -   当query\_rope和keyRope为空时：要求Q\_D、K\_D等于192；TND场景，V\_D等于128/192；NTD\_TND场景，V\_D等于128。当query\_rope和keyRope不为空时，要求Q\_D、K\_D、V\_D等于128；
        -   N等于2/4/8/16/32/64/128，且Q\_N、K\_N、V\_N相等；
        -   支持TND、NTD\_TND；
        -   数据类型仅支持BFLOAT16；
        -   当sparse=3时，要求每个batch单独的actualSeqLengths<actualSeqLengthsKv；
        -   不支持左padding、tensorlist、pse、page attention、prefix、伪量化、全量化、后量化；
        -   **不支持图模式配置Tiling调度优化**（tiling\_schedule\_optimize=True）、**reduce-overhead执行模式**（config.mode="reduce-overhead"）。
        -   actual\_seq\_lengths和actual\_seq\_lengths\_kv的元素个数不大于4096。
    
-   **当Q\_S大于1时：**
    -   query、key、value输入，功能使用限制如下：
        -   支持B轴小于等于65536，D轴32byte不对齐时仅支持到128。
        -   支持N轴小于等于256，支持D轴小于等于512；input\_layout为BSH或者BSND时，建议N\*D小于65535。
        -   S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致PFA算子执行超时（aicore error类型报错，errorStr为timeout or trap error），此场景下建议做S切分处理（注：这里计算量会受B、S、N、D等的影响，值越大计算量越大），典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
            -   B=1，Q\_N=20，Q\_S=2097152，D=256，KV\_N=1，KV\_S=2097152。
            -   B=1，Q\_N=2，Q\_S=20971520，D=256，KV\_N=2，KV\_S=20971520。
            -   B=20，Q\_N=1，Q\_S=2097152，D=256，KV\_N=1，KV\_S=2097152。
            -   B=1，Q\_N=10，Q\_S=2097152，D=512，KV\_N=1，KV\_S=2097152。

        -   query、key、value输入类型包含int8时，D轴需要32对齐；输入类型全为float16、bfloat16时，D轴需16对齐。
        -   D轴限制：
            -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：query、key、value输入类型包含int8时，D轴需要32对齐；query、key、value或attentionOut类型包含INT4时，D轴需要64对齐；输入类型全为float16、bfloat16时，D轴需16对齐。

    -   actual\_seq\_lengths：
    
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于query中对应batch的Sequence Length。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。当query的inputLayout为TND/NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   actual\_seq\_lengths\_kv：
    
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的inputLayout为TND/NTD\_TND时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   参数sparse\_mode当前仅支持值为0、1、2、3、4的场景，取其它值时会报错。
        
        -   sparse\_mode=0时，atten\_mask如果为None，或者在左padding场景传入atten\_mask，则忽略入参pre\_tokens、next\_tokens（内部赋值为INT\_MAX）。
        -   sparse\_mode=2、3、4时，atten\_mask的shape需要为\(S, S\)或\(1, S, S\)或\(1, 1, S, S\)，其中S的值需要固定为2048，且需要用户保证传入的atten\_mask为下三角，不传入atten\_mask或者传入的shape不正确报错。   
        -   sparse\_mode=1、2、3的场景忽略入参pre\_tokens、next\_tokens并按照相关规则赋值。
        
    -   kvCache反量化的合成参数场景仅支持int8反量化到float16。入参key、value的data range与入参antiquant\_scale的data range乘积范围在（-1, 1）内，高性能模式可以保证精度，否则需要开启高精度模式来保证精度。
    -   page attention场景：
        -   page attention的使能必要条件是block\_table存在且有效，同时key、value是按照block\_table中的索引在一片连续内存中排布，支持key、value数据类型为float16、bfloat16。在该场景下key、value的input\_layout参数无效。block\_table中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
        -   block\_size是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，block\_size最小为128，最大为512，且要求是128的倍数。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
    
        -   page attention场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且KV\_N\*D超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小KV\_N）或调整kv cache排布格式为（blocknum, KV\_N, blocksize, D）解决。当query的input\_layout为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV\_N, blocksize, D）两种格式，当query的input\_layout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actual\_seq\_lengths\_kv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
        -   page attention不支持伪量化场景，不支持tensorlist场景，不支持左padding场景。
        -   page attention场景下，必须传入actual\_seq\_lengths\_kv。
        -   page attention场景下，block\_table必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actual\_seq\_lengths\_kv对应的block数量）。
        -   page atte两种格式和float32/bfloat1ntion场景下，不支持输入query为int8的场景。
        -   page attention使能场景下，以下场景输入需满足KV\_S\>=maxBlockNumPerSeq\*blockSize：
            -   传入attenMask时，如mask shape为\(B, 1, Q\_S, KV\_S\)。
            -   传入pseShift时，如pseShift shape为（B, N, Q\_S, KV\_S）。
    
    -   query左padding场景：
        -   query左padding场景query的搬运起点计算公式为：Q\_S-query\_padding\_size-actual\_seq\_lengths。query的搬运终点计算公式为：Q\_S-query\_padding\_size。其中query的搬运起点不能小于0，终点不能大于Q\_S，否则结果将不符合预期。
        -   query左padding场景kv\_padding\_size小于0时将被置为0。
        -   query左padding场景需要与actual\_seq\_lengths参数一起使能，否则默认为query右padding场景。
        -   query左padding场景不支持PageAttention，不能与block\_table参数一起使能。
        
    -   kv左padding场景：
        -   kv左padding场景key和value的搬运起点计算公式为：KV\_S-kv\_padding\_size-actual\_seq\_lengths\_kv。key和value的搬运终点计算公式为：KV\_S-kv\_padding\_size。其中key和value的搬运起点不能小于0，终点不能大于KV\_S，否则结果将不符合预期。
        -   kv左padding场景kv\_padding\_size小于0时将被置为0。
        -   kv左padding场景需要与actual\_seq\_lengths\_kv参数一起使能，否则默认为kv右padding场景。       
        -   kv左padding场景不支持PageAttention，不能与block\_table参数一起使能。
        
    -   入参quant\_scale2和quant\_offset2支持per-tensor、per-channel量化，支持float32、bfloat16类型。若传入quant\_offset2，需保证其类型和shape信息与quant\_scale2一致。当输入为bfloat16时，同时支持float32和bfloat16，否则仅支持float32。per-channel场景下，当输出layout为BSH时，要求quant\_scale2所有维度的乘积等于H；其他layout要求乘积等于N\*D。当输出layout为BSH时，quant\_scale2 shape建议传入\(1, 1, H\)或\(H,\)；当输出layout为BNSD时，建议传入\(1, N, 1, D\)或\(N, D\)；当输出为BSND时，建议传入\(1, 1, N, D\)或\(N, D)。
    -   输出为int8，quant\_scale2和quant\_offset2为per-channel时，暂不支持左padding、Ring Attention或者D非32Byte对齐的场景。
    -   输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。
    -   pse\_shift功能使用限制如下：
        
        -   支持query数据类型为float16或bfloat16或int8场景下使用该功能。
        -   query、key、value数据类型为float16且pse\_shift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
        -   Q\_S需大于等于query的S长度，KV\_S需大于等于key的S长度。prefix场景KV\_S需大于等于actual\_shared\_prefix\_len与key的S长度之和。
        
    -   输出为int8，入参quant\_offset2传入非None和非空tensor值，并且sparse\_mode、pre\_tokens和next\_tokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截：
        -   sparse\_mode=0，atten\_mask如果非None，每个batch actual\_seq\_lengths-actual\_seq\_lengths\_kv-pre\_tokens\>0或next\_tokens<0时，满足拦截条件。
        -   sparse\_mode=1或2，不会出现满足拦截条件的情况。
        -   sparse\_mode=3，每个batch actual\_seq\_lengths\_kv-actual\_seq\_lengths<0，满足拦截条件。      
        -   sparse\_mode=4，pre\_tokens<0或每个batch next\_tokens+actual\_seq\_lengths\_kv-actual\_seq\_lengths<0时，满足拦截条件。
        
    -   prefix相关参数约束：
        -   key\_shared\_prefix和value\_shared\_prefix要么都为空，要么都不为空。
        -   key\_shared\_prefix和value\_shared\_prefix都不为空时，key\_shared\_prefix、value\_shared\_prefix、key、value的维度相同、dtype保持一致。
        -   key\_shared\_prefix和value\_shared\_prefix都不为空时，key\_shared\_prefix的shape第一维batch必须为1，layout为BNSD和BSND情况下N、D轴要与key一致、BSH情况下H要与key一致，value\_shared\_prefix同理。key\_shared\_prefix和value\_shared\_prefix的S应相等。
        -   当actual\_shared\_prefix\_len存在时，actual\_shared\_prefix\_len的shape需要为\[1\]，值不能大于key\_shared\_prefix和value\_shared\_prefix的S。
        -   公共前缀的S加上key或value的S的结果，要满足原先key或value的S的限制。
        -   prefix不支持PageAttention场景、不支持左padding场景、不支持tensorlist场景。
        -   prefix场景不支持query、key、value数据类型同时为int8。
        -   prefix场景，sparse为0或1时，如果传入attenmask，则S2需大于等于actual\_shared\_prefix\_len与key的S长度之和。      
        -   prefix场景，不支持输入qkv全部为int8的场景。
    
-   kv伪量化参数分离：
    -   当伪量化参数和KV分离量化参数同时传入时，以KV分离量化参数为准。
    -   key\_antiquant\_mode和value\_antiquant\_mode取值需要保持一致。
    -   key\_antiquant\_scale和value\_antiquant\_scale要么都为空，要么都不为空；key\_antiquant\_offset和value\_antiquant\_offset要么都为空，要么都不为空。
    -   key\_antiquant\_scale和value\_antiquant\_scale都不为空时，其shape需要保持一致；key\_antiquant\_offset和value\_antiquant\_offset都不为空时，其shape需要保持一致。    
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        -   仅支持per-token和per-channel模式，per-token模式下要求两个参数的shape均为\(B, S\)，数据类型固定为float32；per-channel模式下要求两个参数的shape为（N，D），\(N, D\)，\(H\)，数据类型固定为bfloat16。
        -   key\_antiquant\_scale与value\_antiquant\_scale非空场景，要求query的s小于等于16；要求query的dtype为bfloat16，key、value的dtype为int8，输出的dtype为bfloat16；不支持tensorlist、左padding、page attention、prefix特性。

    -   管理scale/offset的量化模式如下：
    
        >**说明：**<br> 
        >注意scale、offset具体指key\_antiquant\_scale、key\_antiquant\_scale、value\_antiquant\_offset、value\_antiquant\_offset参数。

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
        <tbody><tr id="zh-cn_topic_0000001832267082_row172761159123213"><td class="cellrowborder" valign="top" width="16.950000000000003%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p14277165915327"><a name="zh-cn_topic_0000001832267082_p14277165915327"></a><a name="zh-cn_topic_0000001832267082_p14277165915327"></a>per-channel模式</p>
        </td>
        <td class="cellrowborder" valign="top" width="23.09%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p113847501392"><a name="zh-cn_topic_0000001832267082_p113847501392"></a><a name="zh-cn_topic_0000001832267082_p113847501392"></a>两个参数shape支持(N, 1, D)，(N, D)，(H)，(1, N, 1, D)，(1, N, D)，(1, H)数据类型和query数据类型相同。</p>
        </td>
        <td class="cellrowborder" valign="top" width="46.660000000000004%" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul2277759183216"></a><a name="zh-cn_topic_0000001832267082_ul2277759183216"></a><ul id="zh-cn_topic_0000001832267082_ul2277759183216"><li><span id="zh-cn_topic_0000001832267082_ph112776597327"><a name="zh-cn_topic_0000001832267082_ph112776597327"></a><a name="zh-cn_topic_0000001832267082_ph112776597327"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_19"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_19"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_19"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span>：仅支持key、value数据类型为int8。</li><li><span id="zh-cn_topic_0000001832267082_ph327717592325"><a name="zh-cn_topic_0000001832267082_ph327717592325"></a><a name="zh-cn_topic_0000001832267082_ph327717592325"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_19"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_19"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_19"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：仅支持key、value数据类型为int8。</li></ul>
        </td>
        <td class="cellrowborder" rowspan="2" valign="top" width="13.3%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000001832267082_ul7277159143219"></a><a name="zh-cn_topic_0000001832267082_ul7277159143219"></a><ul id="zh-cn_topic_0000001832267082_ul7277159143219"><li><span id="zh-cn_topic_0000001832267082_ph527775920323"><a name="zh-cn_topic_0000001832267082_ph527775920323"></a><a name="zh-cn_topic_0000001832267082_ph527775920323"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_20"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_20"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_20"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></li><li><span id="zh-cn_topic_0000001832267082_ph22772059103210"><a name="zh-cn_topic_0000001832267082_ph22772059103210"></a><a name="zh-cn_topic_0000001832267082_ph22772059103210"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_20"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_20"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_20"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></li></ul>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001832267082_row027816595321"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p19936325114616"><a name="zh-cn_topic_0000001832267082_p19936325114616"></a><a name="zh-cn_topic_0000001832267082_p19936325114616"></a>per-token模式</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p39361225144616"><a name="zh-cn_topic_0000001832267082_p39361225144616"></a><a name="zh-cn_topic_0000001832267082_p39361225144616"></a>两个参数的shape均为(B, S)，数据类型固定为float32。</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul7475108135214"></a><a name="zh-cn_topic_0000001832267082_ul7475108135214"></a><ul id="zh-cn_topic_0000001832267082_ul7475108135214"><li><span id="zh-cn_topic_0000001832267082_ph1947698135214"><a name="zh-cn_topic_0000001832267082_ph1947698135214"></a><a name="zh-cn_topic_0000001832267082_ph1947698135214"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_21"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_21"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_21"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span>：仅支持key、value数据类型为int8。</li><li><span id="zh-cn_topic_0000001832267082_ph12476287523"><a name="zh-cn_topic_0000001832267082_ph12476287523"></a><a name="zh-cn_topic_0000001832267082_ph12476287523"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_21"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_21"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_21"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：仅支持key、value数据类型为int8。</li></ul>
        </td>
        </tr>
        </tbody>
        </table>
    
-   **当Q\_S等于1时：**
    -   query、key、value输入，功能使用限制如下：
        -   支持B轴小于等于65536，支持N轴小于等于256，支持S轴小于等于262144，支持D轴小于等于512。
        -   query、key、value输入类型均为int8的场景暂不支持。
        -   在int4（int32）伪量化场景下，PyTorch入图调用仅支持KV int4拼接成int32输入（建议通过dynamicQuant生成int4格式的数据，因为dynamicQuant就是一个int32包括8个int4）。
        -   在int4（int32）伪量化场景下，若KV int4拼接成int32输入，那么KV的N、D或者H是实际值的八分之一（prefix同理）。并且，int4伪量化仅支持D 64对齐（int32支持D 8对齐）。

    -   actual\_seq\_lengths：
        
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当query的inputLayout不为TND时，Q\_S为1时该参数无效。当query的inputLayout为TND/TND\_NTD时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   actual\_seq\_lengths\_kv：    
        
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的inputLayout为TND/TND\_NTD时，综合约束请见[约束说明](#zh-cn_topic_0000001832267082_section12345537164214)。
        
    -   page attention场景：
        -   使能必要条件是block\_table存在且有效，同时key、value是按照block\_table中的索引在一片连续内存中排布，在该场景下key、value的input\_layout参数无效。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
            -   支持key、value数据类型为float16、bfloat16、int8。
            -   不支持Q为bfloat16、float16、key、value为int4（int32）的场景。
    
        -   该场景下，block\_size是用户自定义的参数，该参数的取值会影响page attention的性能。key、value输入类型为float16、bfloat16时需要16对齐，key、value输入类型为int8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
        -   参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。
        -   page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actual\_seq\_lengths\_kv对应的block数量）。
        -   page attention场景下，当query的input\_layout为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV\_N, blocksize, D）两种格式，当query的input\_layout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actual\_seq\_lengths\_kv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
        -   page attention场景下，kv cache排布为（blocknum, KV\_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV\_N, blocksize, D）格式。
        -   page attention使能场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且numKvHeads \* headDim超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小numKvHeads）或调整kv cache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
        -   page attention不支持左padding场景。
        -   page attention场景的参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。
    
    -   kv左padding场景：
        -   kvCache的搬运起点计算公式为：Smax-kv\_padding\_size-actual\_seq\_lengths。kvCache的搬运终点计算公式为：Smax-kv\_padding\_size。其中kvCache的搬运起点或终点小于0时，返回数据结果为全0。
        -   kv\_padding\_size小于0时将被置为0。
        -   使能需要同时存在actual\_seq\_lengths参数，否则默认为kv右padding场景。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：kv左padding场景不支持Q为bfloat16/float16、KV为int4（int32）的场景。
    
    -   kv伪量化参数分离：
        -   除了key\_antiquant\_mode为0并且value\_antiquant\_mode为1的场景外，key\_antiquant\_mode和value\_antiquant\_mode取值需要保持一致。
    
        -   key\_antiquant\_scale和value\_antiquant\_scale要么都为空，要么都不为空；key\_antiquant\_offset和value\_antiquant\_offset要么都为空，要么都不为空。
        -   key\_antiquant\_scale和value\_antiquant\_scale都不为空时，除了key\_antiquant\_mode为0并且value\_antiquant\_mode为1的场景外，其shape需要保持一致；key\_antiquant\_offset和value\_antiquant\_offset都不为空时，除了key\_antiquant\_mode为0并且value\_antiquant\_mode为1的场景外，其shape需要保持一致。
        -   int4（int32）伪量化场景不支持后量化。
        -   管理scale/offset的量化模式如下：
        
            >**说明：**<br> 
            >注意scale、offset两个参数指key\_antiquant\_scale、key\_antiquant\_scale、value\_antiquant\_offset、value\_antiquant\_offset。
        
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
            <tbody><tr id="zh-cn_topic_0000001832267082_row10411185232"><td class="cellrowborder" valign="top" width="16.950000000000003%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p154120882315"><a name="zh-cn_topic_0000001832267082_p154120882315"></a><a name="zh-cn_topic_0000001832267082_p154120882315"></a>per-channel模式</p>
            </td>
            <td class="cellrowborder" valign="top" width="23.09%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p21391147716"><a name="zh-cn_topic_0000001832267082_p21391147716"></a><a name="zh-cn_topic_0000001832267082_p21391147716"></a>两个参数shape支持(1, N, 1, D)，(1, N, D)，(1, H)，数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" width="46.660000000000004%" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul0858154962714"></a><a name="zh-cn_topic_0000001832267082_ul0858154962714"></a><ul id="zh-cn_topic_0000001832267082_ul0858154962714"><li><span id="zh-cn_topic_0000001832267082_ph1163117183317"><a name="zh-cn_topic_0000001832267082_ph1163117183317"></a><a name="zh-cn_topic_0000001832267082_ph1163117183317"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_26"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_26"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_26"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span>：当key、value数据类型为int4（int32）或int8时支持。</li><li><span id="zh-cn_topic_0000001832267082_ph10252223193315"><a name="zh-cn_topic_0000001832267082_ph10252223193315"></a><a name="zh-cn_topic_0000001832267082_ph10252223193315"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_26"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_26"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_26"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int4（int32）或int8时支持。</li></ul>
            </td>
            <td class="cellrowborder" rowspan="9" valign="top" width="13.3%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000001832267082_ul15575120101720"></a><a name="zh-cn_topic_0000001832267082_ul15575120101720"></a><ul id="zh-cn_topic_0000001832267082_ul15575120101720"><li><span id="zh-cn_topic_0000001832267082_ph112662491714"><a name="zh-cn_topic_0000001832267082_ph112662491714"></a><a name="zh-cn_topic_0000001832267082_ph112662491714"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_27"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_27"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_27"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></li><li><span id="zh-cn_topic_0000001832267082_ph1290711610176"><a name="zh-cn_topic_0000001832267082_ph1290711610176"></a><a name="zh-cn_topic_0000001832267082_ph1290711610176"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_27"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_27"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_27"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></li></ul>
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
            <tr id="zh-cn_topic_0000001832267082_row84115813237"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p10419882318"><a name="zh-cn_topic_0000001832267082_p10419882318"></a><a name="zh-cn_topic_0000001832267082_p10419882318"></a>per-tensor模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p04120872317"><a name="zh-cn_topic_0000001832267082_p04120872317"></a><a name="zh-cn_topic_0000001832267082_p04120872317"></a>两个参数的shape均为(1,)，数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul19978121525716"></a><a name="zh-cn_topic_0000001832267082_ul19978121525716"></a><ul id="zh-cn_topic_0000001832267082_ul19978121525716"><li><span id="zh-cn_topic_0000001832267082_ph19978111585717"><a name="zh-cn_topic_0000001832267082_ph19978111585717"></a><a name="zh-cn_topic_0000001832267082_ph19978111585717"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_28"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_28"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_28"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span>：当key、value数据类型为int8时支持。</li><li><span id="zh-cn_topic_0000001832267082_ph13978111545713"><a name="zh-cn_topic_0000001832267082_ph13978111545713"></a><a name="zh-cn_topic_0000001832267082_ph13978111545713"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_28"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_28"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_28"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int8时支持。</li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row1341138172312"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p134114862314"><a name="zh-cn_topic_0000001832267082_p134114862314"></a><a name="zh-cn_topic_0000001832267082_p134114862314"></a>per-token模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p1037135185914"><a name="zh-cn_topic_0000001832267082_p1037135185914"></a><a name="zh-cn_topic_0000001832267082_p1037135185914"></a>两个参数的shape均为(1, B, S)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p174417474211"><a name="zh-cn_topic_0000001832267082_p174417474211"></a><a name="zh-cn_topic_0000001832267082_p174417474211"></a>key、value数据类型为int4（int32）或int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row12620173672311"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p166201636132311"><a name="zh-cn_topic_0000001832267082_p166201636132311"></a><a name="zh-cn_topic_0000001832267082_p166201636132311"></a>per-tensor叠加per-head模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p662110362235"><a name="zh-cn_topic_0000001832267082_p662110362235"></a><a name="zh-cn_topic_0000001832267082_p662110362235"></a>两个参数的shape均为(N,)，数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul519618222020"></a><a name="zh-cn_topic_0000001832267082_ul519618222020"></a><ul id="zh-cn_topic_0000001832267082_ul519618222020"><li><span id="zh-cn_topic_0000001832267082_ph121961922601"><a name="zh-cn_topic_0000001832267082_ph121961922601"></a><a name="zh-cn_topic_0000001832267082_ph121961922601"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_29"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_29"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_29"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span>：当key、value数据类型为int8时支持。</li><li><span id="zh-cn_topic_0000001832267082_ph11961022801"><a name="zh-cn_topic_0000001832267082_ph11961022801"></a><a name="zh-cn_topic_0000001832267082_ph11961022801"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_29"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_29"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_29"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int8时支持。</li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row136211336192318"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p22659468119"><a name="zh-cn_topic_0000001832267082_p22659468119"></a><a name="zh-cn_topic_0000001832267082_p22659468119"></a>per-token叠加per-head模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p116212367239"><a name="zh-cn_topic_0000001832267082_p116212367239"></a><a name="zh-cn_topic_0000001832267082_p116212367239"></a>两个参数的shape均为(B, N, S)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p162285131891"><a name="zh-cn_topic_0000001832267082_p162285131891"></a><a name="zh-cn_topic_0000001832267082_p162285131891"></a>key、value数据类型为int4（int32）或int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row1037716581001"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p1757135414112"><a name="zh-cn_topic_0000001832267082_p1757135414112"></a><a name="zh-cn_topic_0000001832267082_p1757135414112"></a>per-token叠加使用page attention模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p1025316567221"><a name="zh-cn_topic_0000001832267082_p1025316567221"></a><a name="zh-cn_topic_0000001832267082_p1025316567221"></a>两个参数的shape均为(blocknum, blocksize)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p04417476217"><a name="zh-cn_topic_0000001832267082_p04417476217"></a><a name="zh-cn_topic_0000001832267082_p04417476217"></a>key、value数据类型为int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row15621736192312"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p1986911315215"><a name="zh-cn_topic_0000001832267082_p1986911315215"></a><a name="zh-cn_topic_0000001832267082_p1986911315215"></a>per-token叠加per head并使用page attention模式</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p2621173692313"><a name="zh-cn_topic_0000001832267082_p2621173692313"></a><a name="zh-cn_topic_0000001832267082_p2621173692313"></a>两个参数的shape均为(blocknum, N, blocksize)，数据类型固定为float32。</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000001832267082_p551893313915"><a name="zh-cn_topic_0000001832267082_p551893313915"></a><a name="zh-cn_topic_0000001832267082_p551893313915"></a>key、value数据类型为int8时支持。</p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row915113171020"><td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p01521217025"><a name="zh-cn_topic_0000001832267082_p01521217025"></a><a name="zh-cn_topic_0000001832267082_p01521217025"></a>key支持per-channel叠加value支持per-token模式</p>
            <p id="zh-cn_topic_0000001832267082_p74743213101"><a name="zh-cn_topic_0000001832267082_p74743213101"></a><a name="zh-cn_topic_0000001832267082_p74743213101"></a></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000001832267082_p1867213119113"><a name="zh-cn_topic_0000001832267082_p1867213119113"></a><a name="zh-cn_topic_0000001832267082_p1867213119113"></a>对于key支持per-channel，两个参数的shape可支持(1, N, 1, D)、(1, N, D)、(1, H)，且参数数据类型和query数据类型相同。</p>
            </td>
            <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.5.1.3 "><a name="zh-cn_topic_0000001832267082_ul1037202951112"></a><a name="zh-cn_topic_0000001832267082_ul1037202951112"></a><ul id="zh-cn_topic_0000001832267082_ul1037202951112"><li><span id="zh-cn_topic_0000001832267082_ph10271547141214"><a name="zh-cn_topic_0000001832267082_ph10271547141214"></a><a name="zh-cn_topic_0000001832267082_ph10271547141214"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_30"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_30"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term11962195213215_30"></a>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span>：当key、value数据类型为int4（int32）或int8时支持；当key和value的数据类型为int8时，仅支持query和输出的dtype为float16。</li><li><span id="zh-cn_topic_0000001832267082_ph427116472125"><a name="zh-cn_topic_0000001832267082_ph427116472125"></a><a name="zh-cn_topic_0000001832267082_ph427116472125"></a><term id="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_30"><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_30"></a><a name="zh-cn_topic_0000001832267082_zh-cn_topic_0000001312391781_term1253731311225_30"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>：当key、value数据类型为int4（int32）或int8时支持；当key和value的数据类型为int8时，仅支持query和输出的dtype为float16。</li></ul>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000001832267082_row194748261012"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000001832267082_p1154111491113"><a name="zh-cn_topic_0000001832267082_p1154111491113"></a><a name="zh-cn_topic_0000001832267082_p1154111491113"></a>对于value支持per-token，两个参数的shape均为(1, B, S)并且数据类型固定为float32。</p>
            </td>
            </tr>
            </tbody>
            </table>
    
    -   pse\_shift功能使用限制如下：
        -   pse\_shift数据类型需与query数据类型保持一致。
        -   仅支持D轴对齐，即D轴可以被16整除。

## 支持的型号<a name="zh-cn_topic_0000001832267082_section182030526511"></a>

<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

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
    scale = 1/math.sqrt(128.0)
    actseqlen = [164]
    actseqlenkv = [1024]
    
    # 调用FIA算子
    out, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, 
    actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
    num_heads = 8, input_layout = "BNSD", scale = scale, pre_tokens=65535, next_tokens=65535)
    
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
    # 入图方式
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
    scale = 1/math.sqrt(128.0)
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
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


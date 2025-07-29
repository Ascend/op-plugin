# torch\_npu.npu\_mla\_prolog\_v2<a name="ZH-CN_TOPIC_0000002350565320"></a>

## 产品支持情况<a name="zh-cn_topic_0000002313328922_section1756684883017"></a>

<a name="zh-cn_topic_0000002313328922_table12566134823012"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002313328922_row75661148173016"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002313328922_p14566448203011"><a name="zh-cn_topic_0000002313328922_p14566448203011"></a><a name="zh-cn_topic_0000002313328922_p14566448203011"></a><span id="zh-cn_topic_0000002313328922_ph1656604883010"><a name="zh-cn_topic_0000002313328922_ph1656604883010"></a><a name="zh-cn_topic_0000002313328922_ph1656604883010"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002313328922_p1356664883014"><a name="zh-cn_topic_0000002313328922_p1356664883014"></a><a name="zh-cn_topic_0000002313328922_p1356664883014"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002313328922_row456619487308"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002313328922_p456611482301"><a name="zh-cn_topic_0000002313328922_p456611482301"></a><a name="zh-cn_topic_0000002313328922_p456611482301"></a><span id="zh-cn_topic_0000002313328922_ph656734813016"><a name="zh-cn_topic_0000002313328922_ph656734813016"></a><a name="zh-cn_topic_0000002313328922_ph656734813016"></a><term id="zh-cn_topic_0000002313328922_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002313328922_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002313328922_zh-cn_topic_0000001312391781_term11962195213215"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002313328922_p14567184883016"><a name="zh-cn_topic_0000002313328922_p14567184883016"></a><a name="zh-cn_topic_0000002313328922_p14567184883016"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002313328922_row294304412306"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002313328922_p95671348183018"><a name="zh-cn_topic_0000002313328922_p95671348183018"></a><a name="zh-cn_topic_0000002313328922_p95671348183018"></a><span id="zh-cn_topic_0000002313328922_ph756720489303"><a name="zh-cn_topic_0000002313328922_ph756720489303"></a><a name="zh-cn_topic_0000002313328922_ph756720489303"></a><term id="zh-cn_topic_0000002313328922_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002313328922_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002313328922_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002313328922_p1456794883019"><a name="zh-cn_topic_0000002313328922_p1456794883019"></a><a name="zh-cn_topic_0000002313328922_p1456794883019"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002313328922_section163097481803"></a>

-   算子功能：推理场景下，Multi-Head Latent Attention（MLA）前处理的计算。主要计算过程分为五路；
    -   首先对输入x乘以W<sup>DQ</sup>进行下采样和RmsNorm后分成两路，第一路乘以W<sup>UQ</sup>和W<sup>UK</sup>经过两次上采样后得到q<sup>N</sup>；
    -   第二路乘以W<sup>QR</sup>后经过旋转位置编码（ROPE）得到q<sup>R</sup>。
    -   第三路是输入x乘以W<sup>DKV</sup>进行下采样和RmsNorm后传入Cache中得到k<sup>C</sup>。
    -   第四路是输入x乘以W<sup>KR</sup>后经过旋转位置编码后传入另一个Cache中得到k<sup>R</sup>。
    -   第五路是预留参数，当前版本不支持使用。

-   计算公式：
    -   RmsNorm公式

        ![](figures/zh-cn_formulaimage_0000002347254869.png)

    -   Query计算公式

        ![](figures/zh-cn_formulaimage_0000002313175960.png)

    -   Query ROPE旋转位置编码

        ![](figures/zh-cn_formulaimage_0000002347254877.png)

    -   Key计算公式

        ![](figures/zh-cn_formulaimage_0000002313175972.png)

    -   Key ROPE旋转位置编码

        ![](figures/zh-cn_formulaimage_0000002347254897.png)

## 函数原型<a name="zh-cn_topic_0000002313328922_section11217581501"></a>

```
torch_npu.npu_mla_prolog_v2(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002313328922_section18982182416164"></a>

-   **token\_x**（Tensor）：必选参数，对应公式中x。shape支持2维和3维，格式为\(T, He\)和\(B, S, He\)，dtype支持bfloat16，数据格式支持ND。
-   **weight\_dq**（Tensor）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为\(He, Hcq\)，dtype支持bfloat16，数据格式支持FRACTAL\_NZ（可通过torch\_npu.npu\_format\_cast将ND格式转为FRACTAL\_NZ格式）。
-   **weight\_uq\_qr**（Tensor）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为\(Hcq, N\*\(D+Dr\)\)，dtype支持bfloat16和int8，数据格式支持FRACTAL\_NZ。
    -   当weight\_uq\_qr为int8类型时，weight\_uq\_qr是一个per-tensor的量化后的输入，表示当前为部分量化场景。

        此时若kv\_cache、kr\_cache为bfloat16类型，对应kv\_cache\_out、kr\_cache\_out为非量化输出，此时dequant\_scale\_w\_uq\_qr字段必须传入，smooth\_scales\_cq字段可选传入。

        此时若kv\_cache、kr\_cache为int8类型，对应kv\_cache\_out、kr\_cache\_out为量化输出，此时dequant\_scale\_w\_uq\_qr、quant\_scale\_ckv、quant\_scale\_ckr字段必须传入，smooth\_scales\_cq字段可选传入。

    -   当weight\_uq\_qr为bfloat16类型时，表示当前为非量化场景。

        此时dequant\_scale\_w\_uq\_qr、quant\_scale\_ckv、quant\_scale\_ckr、smooth\_scales\_cq字段不能传入（即为none）。

-   **weight\_uk**（Tensor）：必选参数**，**表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为\(N, D, Hckv\)，dtype支持bfloat16，数据格式支持ND。
-   **weight\_dkv\_kr**（Tensor）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为\(He, Hckv+Dr\)，dtype支持bfloat16，数据格式支持FRACTAL\_NZ。
-   **rmsnorm\_gamma\_cq**（Tensor）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为\(Hcq,\)，dtype支持bfloat16，数据格式支持ND。
-   **rmsnorm\_gamma\_ckv**（Tensor）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为\(Hckv,\)，dtype支持bfloat16，数据格式支持ND。
-   **rope\_sin**（Tensor）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为\(T, Dr\)和\(B, S, Dr\)，dtype支持bfloat16，数据格式支持ND。
-   **rope\_cos**（Tensor）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为\(T, Dr\)和\(B, S, Dr\)，dtype支持bfloat16，数据格式支持ND。
-   **cache\_index**（Tensor）：必选参数，表示用于存储kv\_cache和kr\_cache的索引。shape支持1维和2维，格式为\(T\)和\(B, S\)，dtype支持int64，数据格式支持ND。
-   **kv\_cache**（Tensor）：必选参数，表示用于cache索引的aclTensor。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Hckv\)，dtype支持bfloat16和int8，数据格式支持ND。
-   **kr\_cache**（Tensor）：必选参数，表示用于key位置编码的cache。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Dr\)，dtype支持bfloat16和int8，数据格式支持ND。
-   **dequant\_scale\_x**（Tensor）：预留可选参数，暂未使用，不传或传入None即可。
-   **dequant\_scale\_w\_dq**（Tensor）：预留可选参数，暂未使用，不传或传入None即可。
-   **dequant\_scale\_w\_uq\_qr**（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化参数为perchannel。shape支持2维，格式为\(1, N\*\(D+Dr\)\)，dtype支持float，数据格式支持ND。
-   **dequant\_scale\_w\_dkv\_kr**（Tensor）：预留可选参数，暂未使用，不传或传入None即可。
-   **quant\_scale\_ckv**（Tensor）：可选参数，用于对输出到kv\_cache\_out中的数据做量化操作时的参数。shape支持2维，格式为\(1, Hckv\)，dtype支持float，数据格式支持ND。
-   **quant\_scale\_ckr**（Tensor）：可选参数，用于对输出到kr\_cache\_out中的数据做量化操作时的参数。shape支持2维，格式为\(1, Dr\)，dtype支持float，数据格式支持ND。
-   **smooth\_scales\_cq**（Tensor）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为\(1, Hcq\)，dtype支持float，数据格式支持ND。
-   **rmsnorm\_epsilon\_cq**（float）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   **rmsnorm\_epsilon\_ckv**（float）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   **cache\_mode**（str）：可选参数，表示kvCache的模式，支持"PA\_BSND"、"PA\_NZ"，用户不特意指定时可传入默认值“PA\_BSND”。

## 返回值说明<a name="zh-cn_topic_0000002313328922_section22231435517"></a>

-   **query**（Tensor）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为\(T, N, Hckv\)和\(B, S, N, Hckv\)，dtype支持bfloat16，数据格式支持ND。
-   **query\_rope**（Tensor）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为\(T, N, Dr\)和\(B, S, N, Dr\)，dtype支持bfloat16，数据格式支持ND。
-   **kv\_cache\_out**（Tensor）：表示Key输出到kv\_cache中的Tensor（本质in-place更新），即公式中k<sup>C</sup>。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Hckv\)，dtype支持bfloat16和int8，数据格式支持ND。
-   **kr\_cache\_out**（Tensor）：表示Key的位置编码输出到kr\_cache中的Tensor（本质in-place更新），即公式中k<sup>R</sup>。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Dr\)，dtype支持bfloat16和int8，数据格式支持ND。
-   **dequant\_scale\_q\_nope**（Tensor）：预留输出，默认生成shape为(1,)的零张量，dtype支持float，数据格式支持ND。

## 约束说明<a name="zh-cn_topic_0000002313328922_section13568144818111"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   接口参数中shape格式字段含义：
    -   B：Batch表示输入样本批量大小，取值范围为1\~65536。
    -   S：Seq-Length表示输入样本序列长度，取值范围为1\~16。
    -   He：Head-Size表示隐藏层的大小，取值为7168。

    -   Hcq：q低秩矩阵维度，取值为1536。
    -   N：Head-Num表示多头数，取值范围为8、16、32、64、128。

    -   Hckv：kv低秩矩阵维度，取值为512。
    -   D：qk不含位置编码维度，取值为128。
    -   Dr：qk位置编码维度，取值为64。
    -   Nkv：kv的head数，取值为1。
    -   BlockNum：PagedAttention场景下的块数，取值为计算B\*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度。
    -   BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    -   T：BS合轴后的大小，取值范围：1\~1048576。
-   shape约束：
    -   若token\_x的维度采用BS合轴，即\(T, He\)，则rope\_sin和rope\_cos的shape为\(T, Dr\)，cache\_index的shape为\(T,\)，dequant\_scale\_x的shape为\(T, 1\)，query的shape为\(T, N, Hckv\)，query\_rope的shape为\(T, N, Dr\)。
    -   若token\_x的维度不采用BS合轴，即\(B, S, He\)，则rope\_sin和rope\_cos的shape为\(B, S, Dr\)，cache\_index的shape为\(B, S\)，dequant\_scale\_x的shape为\(B\*S, 1\)，query的shape为\(B, S, N, Hckv\)，query\_rope的shape为\(B, S, N, Dr\)。

## 调用示例<a name="zh-cn_topic_0000002313328922_section983519211229"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import math
    # 生成随机数据, 并发送到npu
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    # 调用MlaProlog算子
    query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope_mla = torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    # 执行上述代码的输出out类似如下
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
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
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope_mla = torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
        print("single op output:", query_mla)
        print("graph output:", graph_output)
        
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
            device='npu:0', dtype=torch.bfloat16)
    
    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16) 
    ```


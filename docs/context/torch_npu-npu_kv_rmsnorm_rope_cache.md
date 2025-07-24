# torch\_npu.npu\_kv\_rmsnorm\_rope\_cache<a name="ZH-CN_TOPIC_0000002343094197"></a>

## 功能说明<a name="zh-cn_topic_0000002236535552_section1023311522369"></a>

-   **算子功能**：融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。
-   **计算公式**：
    -   **输入张量kv拆分**：拆分为两部分，其中B为批次大小，T为序列长度。

        ![](./figures/zh-cn_formulaimage_0000002239561238.png)

    -   **RMS归一化**：对rms\_in，应用RMS归一化。

        ![](./figures/zh-cn_formulaimage_0000002239721038.png)

        -   γ∈R^512是可学习的缩放参数。
        -   Ed\[·\]表示沿最后一个维度（维度d=512）的均值。
        -   ε为小常数（如0.00001），防止除以零。
        -   ⊙表示逐元素相乘。

    -   **旋转位置编码（RoPE）**
        1.  重塑与转置：将rope\_in重塑并转置以准备旋转

            ![](./figures/zh-cn_formulaimage_0000002239561242.png)

        2.  旋转操作：应用旋转位置编码

            ![](./figures/zh-cn_formulaimage_0000002239721042.png)

            -   cos⁡和sin⁡为预计算的旋转角度参数。
            -   RotateHalf\(k\)将k的后半部分元素移至前半部分并取反，后半部分用前半部分的值。具体来说，对于维度d=64：

            ![](./figures/zh-cn_formulaimage_0000002242091560.png)

## 函数原型<a name="zh-cn_topic_0000002236535552_section123412524369"></a>

```
torch_npu.npu_kv_rmsnorm_rope_cache(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index, Tensor k_cache, Tensor ckv_cache, *, Tensor? k_rope_scale=None, Tensor? c_kv_scale=None, Tensor? k_rope_offset=None, Tensor? c_kv_offset=None, float epsilon=1e-5, str cache_mode='Norm', bool is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002236535552_section1723416525369"></a>

>**说明：**<br> 
>Tensor中shape使用的变量说明：
>-   batch\_size：batch的大小。
>-   seq\_len：sequence的长度。
>-   hidden\_size：表示MLA输入的向量长度，取值仅支持576。
>-   rms\_size：表示RMSNorm分支的向量长度，取值仅支持512。
>-   rope\_size：表示RoPE分支的向量长度，取值仅支持64。
>-   cache\_length：Norm模式下有效，表示KVCache支持的最大长度。
>-   block\_num：PagedAttention模式下有效，表示Block的个数。
>-   block\_size：PagedAttention模式下有效，表示Block的大小。

-   kv：Tensor类型，表示输入的特征张量。数据类型支持bfloat16、float16，数据格式为BNSD，要求为4D的Tensor，形状为\[batch\_size, 1, seq\_len, hidden\_size\]，其中hidden\_size=rms\_size\(RMS\)+rope\_size\(RoPE\)。
-   gamma：Tensor类型，表示RMS归一化的缩放参数。数据类型支持bfloat16、float16，数据格式为ND，要求为1D的Tensor，形状为\[rms\_size\]。
-   cos：Tensor类型，表示RoPE旋转位置编码的余弦分量。数据类型支持bfloat16、float16，数据格式为ND，要求为4D的Tensor，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
-   sin：Tensor类型，表示RoPE旋转位置编码的正弦分量。数据类型支持bfloat16、float16，数据格式为ND，要求为4D的Tensor，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
-   index：Tensor类型，表示缓存索引张量，用于定位k\_cache和ckv\_cache的写入位置。数据类型支持int64，数据格式为ND。shape取决于cache\_mode。
-   k\_cache：Tensor类型，用于存储量化/非量化的键向量。数据类型支持bfloat16、float16、int8，数据格式为ND。shape取决于cache\_mode。
-   ckv\_cache：Tensor类型，用于存储量化/非量化的压缩后的kv向。数据类型支持bfloat16、float16、int8，数据格式为ND。shape取决于cache\_mode。

-   k\_rope\_scale：Tensor类型，可选，默认值None，表示k旋转位置编码的量化缩放因子。数据类型支持float32，数据格式为ND，要求为1D的Tensor，形状为\[rope\_size\]。量化模式下必填。
-   c\_kv\_scale：Tensor类型，可选，默认值None，表示压缩后kv的量化缩放因子。数据类型支持float32，数据格式为ND，要求为1D的Tensor，形状为\[rms\_size\]。量化模式下必填。
-   k\_rope\_offset：Tensor类型，可选，默认值None，表示k旋转位置编码量化偏移量。数据类型支持float32，数据格式为ND，要求为1D的Tensor，形状为\[rope\_size\]。量化模式下必填。
-   c\_kv\_offset：Tensor类型，可选，默认值None，表示压缩后kv的量化偏移量。数据类型支持float32，数据格式为ND，要求为1D的Tensor，形状为\[rms\_size\]。量化模式下必填。
-   epsilon：float类型，可选，默认值1e-5，表示RMS归一化中的极小值，防止除以零。
-   cache\_mode：string类型，可选，默认值'Norm'，表示缓存模式，支持的模式如下：

    <a name="zh-cn_topic_0000002236535552_table16997195773911"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000002236535552_row12998195743918"><th class="cellrowborder" valign="top" width="10.34%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002236535552_p1299819576394"><a name="zh-cn_topic_0000002236535552_p1299819576394"></a><a name="zh-cn_topic_0000002236535552_p1299819576394"></a>枚举值</p>
    </th>
    <th class="cellrowborder" valign="top" width="19.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002236535552_p93491748114015"><a name="zh-cn_topic_0000002236535552_p93491748114015"></a><a name="zh-cn_topic_0000002236535552_p93491748114015"></a>模式名</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.82000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002236535552_p099810576395"><a name="zh-cn_topic_0000002236535552_p099810576395"></a><a name="zh-cn_topic_0000002236535552_p099810576395"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000002236535552_row499835715398"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002236535552_p09984572391"><a name="zh-cn_topic_0000002236535552_p09984572391"></a><a name="zh-cn_topic_0000002236535552_p09984572391"></a>Norm</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002236535552_p23491348164018"><a name="zh-cn_topic_0000002236535552_p23491348164018"></a><a name="zh-cn_topic_0000002236535552_p23491348164018"></a>KV-Cache更新模式</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002236535552_p1499812579399"><a name="zh-cn_topic_0000002236535552_p1499812579399"></a><a name="zh-cn_topic_0000002236535552_p1499812579399"></a>k_cache形状为[batch_size, 1, cache_length, rope_size]，ckv_cache形状为[batch_size, 1, cache_length, rms_size]。index形状为[batch_size, seq_len]，index里的值表示每个Batch下的偏移。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002236535552_row1699810579398"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002236535552_p1699810576392"><a name="zh-cn_topic_0000002236535552_p1699810576392"></a><a name="zh-cn_topic_0000002236535552_p1699810576392"></a>PA/PA_BNSD</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002236535552_p1934914484406"><a name="zh-cn_topic_0000002236535552_p1934914484406"></a><a name="zh-cn_topic_0000002236535552_p1934914484406"></a>PagedAttention模式</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002236535552_p19457162584119"><a name="zh-cn_topic_0000002236535552_p19457162584119"></a><a name="zh-cn_topic_0000002236535552_p19457162584119"></a>k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*seq_len]，index里的值表示每个token的偏移。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002236535552_row179981757193914"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002236535552_p99981257113913"><a name="zh-cn_topic_0000002236535552_p99981257113913"></a><a name="zh-cn_topic_0000002236535552_p99981257113913"></a>PA_NZ</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002236535552_p6349174814019"><a name="zh-cn_topic_0000002236535552_p6349174814019"></a><a name="zh-cn_topic_0000002236535552_p6349174814019"></a>Cache数据格式为FRACTAL_NZ的PagedAttention模式</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002236535552_p15265174112368"><a name="zh-cn_topic_0000002236535552_p15265174112368"></a><a name="zh-cn_topic_0000002236535552_p15265174112368"></a>k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size * seq_len]，index里的值表示每个token的偏移。</p>
    <p id="zh-cn_topic_0000002236535552_p10572135917168"><a name="zh-cn_topic_0000002236535552_p10572135917168"></a><a name="zh-cn_topic_0000002236535552_p10572135917168"></a>不同量化模式下数据排布不同：</p>
    <a name="zh-cn_topic_0000002236535552_ul183421286166"></a><a name="zh-cn_topic_0000002236535552_ul183421286166"></a><ul id="zh-cn_topic_0000002236535552_ul183421286166"><li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li></ul>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002236535552_row428921319407"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002236535552_p18290111311409"><a name="zh-cn_topic_0000002236535552_p18290111311409"></a><a name="zh-cn_topic_0000002236535552_p18290111311409"></a>PA_BLK_BNSD</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002236535552_p12349144844010"><a name="zh-cn_topic_0000002236535552_p12349144844010"></a><a name="zh-cn_topic_0000002236535552_p12349144844010"></a>特殊的PagedAttention模式</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002236535552_p329011315406"><a name="zh-cn_topic_0000002236535552_p329011315406"></a><a name="zh-cn_topic_0000002236535552_p329011315406"></a>k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*Ceil(seq_len/block_size)]，index里的值表示每个block的起始偏移，不再和token一一对应。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002236535552_row182905133408"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002236535552_p13290313134017"><a name="zh-cn_topic_0000002236535552_p13290313134017"></a><a name="zh-cn_topic_0000002236535552_p13290313134017"></a>PA_BLK_NZ</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002236535552_p173491486403"><a name="zh-cn_topic_0000002236535552_p173491486403"></a><a name="zh-cn_topic_0000002236535552_p173491486403"></a>Cache数据格式为FRACTAL_NZ的特殊的PagedAttention模式</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002236535552_p637504423711"><a name="zh-cn_topic_0000002236535552_p637504423711"></a><a name="zh-cn_topic_0000002236535552_p637504423711"></a>k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size * Ceil(seq_len / block_size)]，index里的值表示每个block的起始偏移，不再和token一一对应。</p>
    <p id="zh-cn_topic_0000002236535552_p916702812303"><a name="zh-cn_topic_0000002236535552_p916702812303"></a><a name="zh-cn_topic_0000002236535552_p916702812303"></a>不同量化模式下数据排布不同：</p>
    <a name="zh-cn_topic_0000002236535552_ul20871102344210"></a><a name="zh-cn_topic_0000002236535552_ul20871102344210"></a><ul id="zh-cn_topic_0000002236535552_ul20871102344210"><li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

-   is\_output\_kv：bool类型，可选，表示是否输出处理后的k\_embed\_out和y\_out（未量化的原始值），默认值False不输出，仅cache\_mode在\(PA/PA\_BNSD/PA\_NZ/PA\_BLK\_BNSD/PA\_BLK\_NZ\)模式下有效。

## 输出说明<a name="zh-cn_topic_0000002236535552_section3234185215368"></a>

-   k\_cache：Tensor类型，和输入k\_cache的数据类型、维度、数据格式完全一致（本质in-place更新）。
-   ckv\_cache：Tensor类型，和输入ckv\_cache的数据类型、维度、数据格式完全一致（本质in-place更新）。
-   k\_embed\_out：Tensor类型，仅当is\_output\_kv=True时输出，表示RoPE处理后的值。要求为4D的Tensor，形状为\[batch\_size, 1, seq\_len, 64\]，数据类型和格式同输入kv一致。
-   y\_out：Tensor类型，仅当is\_output\_kv=True时输出，表示RMSNorm处理后的值。要求为4D的Tensor，形状为\[batch\_size, 1, seq\_len, 512\]，数据类型和格式同输入kv一致。

## 约束说明<a name="zh-cn_topic_0000002236535552_section1523425283618"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   量化模式：当k\_rope\_scale和c\_kv\_scale非空时，k\_cache和ckv\_cache的dtype为int8，缓存形状的最后一个维度需要为32（Cache数据格式为FRACTAL\_NZ模式），k\_rope\_scale和c\_kv\_scale必须同时非空，k\_rope\_offset和c\_kv\_offset必须同时为None为非空。
-   非量化模式：当k\_rope\_scale和c\_kv\_scale为空时，k\_cache和ckv\_cache的dtype为bfloat16或float16。
-   索引映射：所有cache\_mode缓存模式下，index的值不可以重复，如果传入的index值存在重复，算子的行为是未定义的且不可预知的。
    -   Norm：index的值表示每个Batch下的偏移。
    -   PA/PA\_BNSD/PA\_NZ：index的值表示全局的偏移。
    -   PA\_BLK\_BNSD/PA\_BLK\_NZ：index的值表示每个页的全局偏移；这个场景假设cache更新是连续的，不支持非连续更新的cache。

-   Shape关联规则：不同的cache\_mode缓存模式有不同的Shape规则。
    -   Norm：k\_cache形状为\[batch\_size, 1, cache\_length, rope\_size\]，ckv\_cache形状为\[batch\_size, 1, cache\_length, rms\_size\]，index形状为\[batch\_size, seq\_len\], cache\_length\>=seq\_len。
    -   非Norm模式\(PagedAttention相关模式\)：要求block\_num\>=Ceil\(seq\_len/block\_size\)\*batch\_size。

## 支持的型号<a name="zh-cn_topic_0000002236535552_section11235195219365"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>

-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例<a name="zh-cn_topic_0000002236535552_section3235105212365"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16
    
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16
    
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    ```


# torch_npu.npu_incre_flash_attention

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id1 -->
<!-- npu="310p" id11 -->
- <term>Atlas inference products</term>: Supported
<!-- end id11 -->

## Function

- Function: incremental FA implementation.
- Formulas:
    $$
    atten\_out = \text{softmax}(scale\_value * (query \mathbin{@} key^\top) + atten\_mask) \mathbin{@} value
    $$

## Prototype

```python
torch_npu.npu_incre_flash_attention(query, key, value, *, padding_mask=None, pse_shift=None, atten_mask=None, actual_seq_lengths=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None, kv_padding_size=None, num_heads=1, scale_value=1.0, input_layout="BSH", num_key_value_heads=0, block_size=0, inner_precise=1) -> Tensor
```

## Parameters

- **`query`** (`Tensor`): Required. Query input of the attention mechanism. The data layout can be ND, where ND stands for N-Dimensional Tensor (a tensor with any number of dimensions).
    <!-- npu="310p" id12 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id12 -->

    <!-- npu="910b" id2 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id2 -->

- **`key`** (`Tensor`): Required. Key input of the attention mechanism. The data layout can be ND.
    <!-- npu="310p" id13 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id13 -->

    <!-- npu="910b" id3 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id3 -->

- **`value`** (`Tensor`): Required. Value input of the attention mechanism. The data layout can be ND.
    <!-- npu="310p" id14 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16` or `torch.int8`.
    <!-- end id14 -->

    <!-- npu="910b" id4 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id4 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`padding_mask`** (`Tensor`): Reserved parameter, currently not used. The default value is `None`.
- **`pse_shift`** (`Tensor`): Optional. Position embedding parameter used within the attention mechanism. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used.
    <!-- npu="310p" id15 -->
    - <term>Atlas inference products</term>: Only `None` is supported.
    <!-- end id15 -->

    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id5 -->

- **`atten_mask`** (`Tensor`): Optional. A value of `1` indicates that the position is masked out and excluded from computation, and `0` indicates that the position is included in computation. The default value is `None` (all positions are included). The data type can be `torch.bool`, `torch.int8`, or `torch.uint8`. The data layout can be ND.
- **`actual_seq_lengths`** (`List[int]`): Optional. Valid sequence lengths $S$ for `key` and `value`. The shape of this parameter is `(B,)` or `(1,)`, such as `[1, 2, 3]`. The default value is `None` (all elements are valid). The data type can be `torch.int64`. The data layout can be ND.
- **`dequant_scale1`** (`Tensor`): Optional. Dequantization factor after BMM1. `pertensor` (scalar) mode is supported. The data type can be `torch.float32`. The data layout can be ND. This parameter is used for dequantization after BMM1 and is typically used when the BMM1 output is quantized and subsequent computations require it to be restored to a floating-point representation. This parameter can be omitted or set to `None` if this feature is not used. Currently, this parameter is not used for Atlas inference accelerator cards.
    <!-- npu="310p" id21 -->
    - <term>Atlas inference products</term>: This parameter is not used currently.
    <!-- end id21 -->
- **`quant_scale1`** (`Tensor`): Optional. Quantization factor before BMM2. `pertensor` (scalar) mode is supported. The data type can be `torch.float32`. The data layout can be ND. This parameter is used for quantization before BMM2 and is typically used when the input needs to be converted to a quantized representation before entering BMM2. This parameter can be omitted or set to `None` if this feature is not used. Currently, this parameter is not used for Atlas inference accelerator cards.
    <!-- npu="310p" id22 -->
    - <term>Atlas inference products</term>: This parameter is not used currently.
    <!-- end id22 -->
- **`dequant_scale2`** (`Tensor`): Optional. Dequantization factor after BMM2. `pertensor` (scalar) mode is supported. The data type can be `torch.float32`. The data layout can be ND. This parameter is used for dequantization after BMM2 and is typically used when the BMM2 output is quantized and subsequent computations require it to be restored to a floating-point representation. This parameter can be omitted or set to `None` if this feature is not used. Currently, this parameter is not used for Atlas inference accelerator cards.
    <!-- npu="310p" id23 -->
    - <term>Atlas inference products</term>: This parameter is not used currently.
    <!-- end id23 -->
- **`quant_scale2`** (`Tensor`): Optional. Quantization factor for output quantization. `pertensor` (scalar) and `perchannel` (list) modes are supported. The data layout can be ND. This parameter is used for final output quantization and is typically used together with `quant_offset2` when the output needs to be quantized. This parameter can be omitted or set to `None` if this feature is not used.
    <!-- npu="310p" id16 -->
    - <term>Atlas inference products</term>: Not supported in the current version.
    <!-- end id16 -->

    <!-- npu="910b" id6 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.bfloat16`.
    <!-- end id6 -->

- **`quant_offset2`** (`Tensor`): Optional. Quantization offset for output quantization. `pertensor` (scalar) and `perchannel` (list) modes are supported. The data layout can be ND. This parameter is used for final output quantization and is typically used together with `quant_scale2`. This parameter can be omitted or set to `None` if output quantization is not used.
    <!-- npu="310p" id17 -->
    - <term>Atlas inference products</term>: not supported in the current version.
    <!-- end id17 -->

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.bfloat16`.
    <!-- end id7 -->

- **`antiquant_scale`** (`Tensor`): Optional. Dequantization factor. `perchannel` (list) mode is supported, which is determined by the shape. In $BNSD$ layouts, the shape of this parameter is `(2, N, 1, D)`. In $BSH$ layouts, the shape is `(2, H)`. In $BSND$ layouts, the shape is `(2, N, D)`. This parameter is used for input/weight dequantization and is typically used to restore low-bit data to the format required for computation. It is generally used together with `antiquant_offset`. This parameter can be omitted or set to `None` if this feature is not used.
    <!-- npu="310p" id18 -->
    - <term>Atlas inference products</term>: the data type can be `torch.float16`.
    <!-- end id18 -->

    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id8 -->

- **`antiquant_offset`** (`Tensor`): Optional. Dequantization offset. `perchannel` (list) mode is supported, which is determined by the shape. In $BNSD$ layouts, the shape of this parameter is `(2, N, 1, D)`. In $BSH$ layouts, the shape is `(2, H)`. In $BSND$ layouts, the shape is `(2, N, D)`. This parameter is used for input/weight dequantization and is generally used together with `antiquant_scale`. This parameter can be omitted or set to `None` if this feature is not used.
    <!-- npu="310p" id19 -->
    - <term>Atlas inference products</term>: the data type can be `torch.float16`.
    <!-- end id19 -->

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id9 -->

- **`block_table`** (`Tensor`): Optional. The data type can be `torch.int32`. The data layout can be ND. This parameter must be a 2D tensor. It indicates the block mapping table used for KV storage in PagedAttention. For details about the constraints and usage, see [Constraints](#en-us_topic_0000001711274864_section12345537164214). This parameter can be omitted or set to `None` if this feature is not used.
- **`kv_padding_size`** (`Tensor`): Optional. Distance from the last valid token to $S$ when left padding of KV is enabled. The data type can be `torch.int64`. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
- **`num_heads`** (`int`): Optional. Head count of `query`, $N$ in the formula for `query`. The default value is `1`. Pass this value as a Python `int`.
- **`scale_value`** (`float`): Optional. Scaling factor used to restrict the gradient. Typical value: $\frac{1}{\sqrt{D}}$. The default value is `1.0`. The data type can be `float32`.
- **`input_layout`** (`str`): Optional. Layout of `query`, `key`, and `value`, which is determined by the shape of the input `query`, `key`, and `value`. For a 3D `Tensor`, the layout is $BSH$. For a 4D `Tensor`, the layout can be $BNSD$ or $BSND$. The data type is `str`. The default value is $BSH$. Other values are not supported.

    > [!NOTE]  
    > `query`, `key`, and `value` support multiple layout formats. $B$ (Batch Size) indicates the batch size of input samples, $S$ (Seq-Length) indicates the sequence length of input samples, $H$ (Hidden Size) indicates the total dimension of the hidden layer, $N$ (Head-Num) indicates the number of heads, and $D$ (Head-Dim) indicates the hidden dimension of each head.
    > When `input_layout` is $BSH$, the `shape` of `query`, `key`, and `value` is $(B, S, H)$; when it is $BNSD$, the `shape` is $(B, N, S, D)$; when it is $BSND$, the `shape` is $(B, S, N, D)$.

- **`num_key_value_heads`** (`int`): Optional. Specifies the number of heads used by `key` and `value` to support GQA (Grouped-Query Attention) scenarios. The default value is `0`, indicating that `key` and `value` have the same number of heads as `query`; otherwise, the number of heads for `key` and `value` is `num_key_value_heads`. In GQA, multiple query heads share the same group of key/value heads to reduce KV cache footprint and computational overhead. In this scenario, `query` has `num_heads` heads, while `key` and `value` have `num_key_value_heads` heads, where `num_heads` must be divisible by `num_key_value_heads`. In addition, the ratio of `num_heads` to `num_key_value_heads` must not exceed `64`. Pass this value as a Python `int`.
- **`block_size`** (`int`): Optional. Maximum head count of tokens inside each block for KV storage in PagedAttention. The data type can be `int64`. The default value is `0`, which is typically set to values such as `128` or `256`.
- **`inner_precise`** (`int`): Optional. Enables high-precision or high-performance mode. Valid values: `0` (high precision) or `1` (high performance) (default). The data type can be `int64`.

## Return Values

`Tensor`

Final computation result, corresponding to $atten\_out$ in the formulas. The `shape` is the same as that of `query`.

- In the non-quantization scenario, the output data type is the same as that of `query`.
- In the quantization scenario, if `quant_scale2` is passed, the output data type is `torch.int8`.

## Constraints<a name="en-us_topic_0000001711274864_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- The number of dimensions of `query`, `key`, and `value` must be the same, and the `shape` of `key` and `value` must be the same.
- The value of `num_heads` must be equal to $N$ of `query`.
- The value of `input_layout` is related to the `shape` of `query`: three dimensions indicate $BSH$, and four dimensions indicate $BNSD$ or $BSND$.
- The value of `num_key_value_heads` must be equal to $N$ of `key` and `value`, and `num_heads` must be divisible by `num_key_value_heads`.
- For the `query`, `key`, and `value` inputs, the usage restrictions are as follows:

    <!-- npu="910b" id10 -->
    - <term>Atlas A2 products</term> supports $B$ axis <= 65535, $N$ axis <= 256, $S$ axis <= 262144, and $D$ axis <= 512.
    <!-- end id10 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas inference products</term> supports $B$ axis <= 256, $N$ axis <= 256, $S$ axis <= 65536, and $D$ axis <= 512.
    <!-- end id20 -->
    - The scenario where the inputs `query`, `key`, and `value` are all `torch.int8` is not supported currently.

- Comprehensive restrictions on the number of `torch.int8` quantization-related input parameters and the input and output data formats:

    Scenario where the inputs `query`, `key`, and `value` are `torch.float16` and the output is `torch.int8`: the input parameter `quant_scale2` is required, `quant_offset2` is optional, and the `dequant_scale1`, `quant_scale1`, and `dequant_scale2` parameters must not be passed (that is, they must be `None`).

- Usage restrictions of the `pse_shift` function are as follows:
    - The data type of `pse_shift` must be the same as that of `query`.
    - Only $D$ axis alignment is supported, that is, the $D$ axis must be divisible by 16.

- PagedAttention usage restrictions:
    - PagedAttention is enabled only when `block_table` exists and is valid and the `actual_seq_lengths` corresponding to each batch is passed. In the PagedAttention scenario, `key` and `value` are mapped to the kv cache in a contiguous memory area according to the index in `block_table`. The data types of `key` and `value` can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    - In the PagedAttention scenario, the input kv cache layout format is $(blocknum, numKvHeads, blocksize, headDims)$ or $(blocknum, blocksize, H)$, where $blocknum$ indicates the total number of physical blocks in the kv cache and must not be less than the sum of the number of blocks required by each batch. Generally, the performance is better when the kv cache layout format is $(blocknum, numKvHeads, blocksize, headDims)$ than when it is $(blocknum, blocksize, H)$.
    - In the PagedAttention scenario, the kv cache layout format $(blocknum, numKvHeads, blocksize, headDims)$ is supported, but in this case the `query layout` supports only $BNSD$.
    - In the PagedAttention scenario, when the input kv cache layout format is $(blocknum, blocksize, H)$ and $H (H=numKvHeads * headDims)$ exceeds 64k, the request is intercepted and an error is reported due to hardware instruction constraints.
    - In the PagedAttention scenario, the input `actual_seq_lengths` must be passed. The `actual_seq_length` of each batch indicates the actual `sequence` length of each batch. Dividing this value by the attribute input `block_size` gives the number of blocks required by each batch.
    - In the PagedAttention scenario, `block_table` must be a 2D `Tensor` with the shape `(B, maxBlockNumPerSeq)`, where `B` indicates the number of batches and `maxBlockNumPerSeq` indicates the maximum number of blocks required in each batch. `block_table[i][j]` indicates the physical block index in the kv cache to which the j-th logical block of the i-th batch is mapped.
    - In the PagedAttention scenario, the length of the first dimension of `block_table` must be equal to the number of batches, and the length of the second dimension must not be less than `maxBlockNumPerSeq` (`maxBlockNumPerSeq` is the number of blocks corresponding to the maximum `actual_seq_lengths` in each batch). For example, if the number of batches is 2, the attribute `block_size=128`, and the `actual_seq_length` of each batch is 512, each batch requires at least 4 blocks, so the layout of `block_table` can be `(2, 4)`.
    - In the PagedAttention scenario, valid values in `block_table` indicate physical block numbers, and the value range must be within `[0, blocknum - 1]`.
    - In the PagedAttention scenario, `block_size` is a user-defined parameter. Its value affects the PagedAttention performance and is usually 128 or 256. When the input types of `key` and `value` are `torch.float16` or `torch.bfloat16`, `block_size` must be 16-aligned; when the input types of `key` and `value` are `torch.int8`, `block_size` must be 32-aligned. Generally, PagedAttention improves the throughput but may increase the decoding latency of a single token.

- `quant_scale2` and `quant_offset2` form a group of parameters, in which `quant_offset2` is optional. After this group of parameters is passed, the output data type of the operator is inferred as `torch.int8`. If `torch.int8` output is not expected, do not pass this group of parameters.
- Usage restrictions of the KV left `padding` scenario:
    - The formula for computing the transfer start point of `kvCache` is: `Smax-kv_padding_size-actual_seq_lengths`. The formula for computing the transfer end point of `kvCache` is: `Smax-kv_padding_size`. When the transfer start or end point of `kvCache` is less than 0, the returned data result is all 0s.
    - In the KV left `padding` scenario, when `kv_padding_size` is less than 0, it is set to 0.
    - In the KV left `padding` scenario, both the `kv_padding_size` and `actual_seq_lengths` parameters must exist; otherwise, the KV right `padding` scenario is used by default.

## Example

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import math
    >>>
    >>> # Generate random data and send it to the npu
    >>> q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> k = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> v = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> scale = 1/math.sqrt(128.0)
    >>>
    >>> # Call the IFA operator
    >>> out = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale)                                        
    >>> print(out)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[[-1.4863,  0.1667,  0.7256,  ...,  0.3052, -0.3630, -0.1936]],

            [[-1.5840, -2.2305, -0.3462,  ..., -2.1055,  0.4392, -1.2842]]],
        device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    # Graph mode approach
    import torch
    import torch_npu
    import math
    
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    import os
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    
    # Print macro that supports graph mode
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # Data generation
    q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    k = torch.randn(2, 2048, 40 * 128, dtype=torch.float16).npu()
    v = torch.randn(2, 2048, 40 * 128, dtype=torch.float16).npu()
    atten = torch.randn(2, 1, 1, 2048).bool().npu()
    scale_value = 1/math.sqrt(128.0)
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self):
            return torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale_value, atten_mask=atten)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
            
        single_op = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale_value, atten_mask=atten)
        print("single op output with mask:", single_op, single_op.shape)
        print("graph output with mask:", graph_output, graph_output.shape)

    if __name__ == "__main__":
        MetaInfershape()
    
    # Expected output of the preceding code sample:
    single op output with mask: tensor([[[ 0.2488, -0.6572,  1.0928,  ...,  0.1694,  0.1142, -2.2266]],
            [[-0.9595, -0.9609, -0.6602,  ...,  0.7959,  1.7920,  0.0783]]],
           device='npu:0', dtype=torch.float16) torch.Size([2, 1, 5120])
    graph output with mask: tensor([[[ 0.2488, -0.6572,  1.0928,  ...,  0.1694,  0.1142, -2.2266]],
            [[-0.9595, -0.9609, -0.6602,  ...,  0.7959,  1.7920,  0.0783]]],
           device='npu:0', dtype=torch.float16) torch.Size([2, 1, 5120])
    ```

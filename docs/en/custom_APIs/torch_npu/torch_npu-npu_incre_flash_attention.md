# torch_npu.npu_incre_flash_attention

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>           |    √     |
|<term>Atlas inference accelerator cards</term>  | √  |

## Function

- Description: Implements incremental FlashAttention (FA).
- Formula:
    $$
    atten\_out = \text{softmax}(scale\_value * (query * key) + atten\_mask) * value
    $$

## Prototype

```python
torch_npu.npu_incre_flash_attention(query, key, value, *, padding_mask=None, pse_shift=None, atten_mask=None, actual_seq_lengths=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None, kv_padding_size=None, num_heads=1, scale_value=1.0, input_layout="BSH", num_key_value_heads=0, block_size=0, inner_precise=1) -> Tensor
```

## Parameters

- **`query`** (`Tensor`): Required. Query input of the attention mechanism. The data layout can be ND.
    - Atlas inference accelerator cards: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16` or `bfloat16`.

- **`key`** (`Tensor`): Required. Key input of the attention mechanism. The data layout can be ND.
    - Atlas inference accelerator cards: The data type can be `float16`, `bfloat16`, or `int8`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `bfloat16`, or `int8`.

- **`value`** (`Tensor`): Required. Value input of the attention mechanism. The data layout can be ND.
    - Atlas inference accelerator cards: The data type can be `float16` or `int8`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `bfloat16`, or `int8`.

- **`*`**: Required. Positional argument separator. Parameters before this symbol are positional-only and must be provided in order. Parameters after this symbol are keyword-only and must be provided as key-value pairs. Default values are used if omitted.
- **`padding_mask`** (`Tensor`): Reserved parameter, currently not used. The default value is `None`.
- **`pse_shift`** (`Tensor`): Optional. Positional encoding parameter used within the attention mechanism. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used.
    - Atlas inference accelerator cards: Only `None` is supported.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16` or `bfloat16`.
- **`atten_mask`** (`Tensor`): Optional. A value of `1` indicates that the position is masked out and excluded from computation, and `0` indicates that the position is included in computation. The default value is `None` (all positions are included). The data type can be `bool`, `int8`, or `uint8`. The data layout can be ND.
- **`actual_seq_lengths`** (`List[int]`): Optional. Valid sequence lengths $S$ for `key` and `value`. The shape of this parameter is `(B,)` or `(1,)`, such as `[1, 2, 3]`. The default value is `None` (all elements are valid). The data type can be `int64`. The data layout can be ND.
- **`dequant_scale1`** (`Tensor`): Optional. Dequantization factor after BMM1. `pertensor` (scalar) mode is supported. The data type can be `float32`. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used. Currently, this parameter is not used for Atlas inference accelerator cards.
- **`quant_scale1`** (`Tensor`): Optional. Quantization factor before BMM2. `pertensor` (scalar) mode is supported. The data type can be `float32`. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used. Currently, this parameter is not used for Atlas inference accelerator cards.
- **`dequant_scale2`** (`Tensor`): Optional. Dequantization factor after BMM2. `pertensor` (scalar) mode is supported. The data type can be `float32`. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used. Currently, this parameter is not used for Atlas inference accelerator cards.
- **`quant_scale2`** (`Tensor`): Optional. Quantization factor for output quantization. `pertensor` (scalar) and `perchannel` (list) modes are supported. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used.
    - Atlas inference accelerator cards: This parameter is not supported in the current version.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32` or `bfloat16`.
- **`quant_offset2`** (`Tensor`): Optional. Quantization offset for output quantization. `pertensor` (scalar) and `perchannel` (list) modes are supported. The data layout can be ND. This parameter can be omitted or set to `None` if this feature is not used.
    - Atlas inference accelerator cards: This parameter is not supported in the current version.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32` or `bfloat16`.
- **`antiquant_scale`** (`Tensor`): Optional. Dequantization factor. `perchannel` (list) mode is supported, which is determined by the shape. In $BNSD$ layouts, the shape of this parameter is `(2, N, 1, D)`. In $BSH$ layouts, the shape is `(2, H)`. In $BSND$ layouts, the shape is `(2, N, D)`. This parameter can be omitted or set to `None` if this feature is not used.
    - Atlas inference accelerator cards: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16` or `bfloat16`.

- **`antiquant_offset`** (`Tensor`): Optional. Dequantization offset. `perchannel` (list) mode is supported, which is determined by the shape. In $BNSD$ layouts, the shape of this parameter is `(2, N, 1, D)`. In $BSH$ layouts, the shape is `(2, H)`. In $BSND$ layouts, the shape is `(2, N, D)`. This parameter can be omitted or set to `None` if this feature is not used.
    - Atlas inference accelerator cards: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16` or `bfloat16`.
- **`block_table`** (`Tensor`): Optional. The data type can be `int32`. The data layout can be ND. This parameter must be a 2D tensor. It indicates the block mapping table used for KV storage in page attention. For details about the constraints and usage, see [Constraints](#en-us_topic_0000001711274864_section12345537164214). This parameter can be omitted or set to `None` if this feature is not used.
- **`kv_padding_size`** (`Tensor`): Optional. Distance from the last valid token to $S$ when left padding of KV is enabled. The data type can be `int64`. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
- **`num_heads`** (`int`): Optional. Head count of `query`, $N$ in the formula for `query`. The default value is `1`. The data type can be `int64`.
- **`scale_value`** (`float`): Optional. Scaling factor used to restrict the gradient. Typical value: $\frac{1}{\sqrt{D}}$. The default value is `1.0`. The data type can be `float32`.
- **`input_layout`** (`str`): Optional. Layout of `query`, `key`, and `value`, which is determined by the shape of the input `query`, `key`, and `value`. For a 3D `Tensor`, the layout is $BSH$. For a 4D `Tensor`, the layout can be $BNSD$ or $BSND$. The data type is `str`. The default value is $BSH$. Other values are not supported.

    > [!NOTE]  
    > The data layouts of `query`, `key`, and `value` can be interpreted from multiple dimensions, where $B$ (Batch) represents the input batch size, $S$ (Seq-Length) represents the input sequence length, $H$ (Hidden-Size) represents the hidden size, $N$ (Head-Num) represents the number of attention heads, and $D$ (Head-Dim) represents the size of each attention head, satisfying $D = H / N$.

- **`num_key_value_heads`** (`int`): Optional. Number of `key` and `value` heads for Grouped-Query Attention (GQA). The default value is `0`, indicating that `key` and `value` use the same number of heads as `query`. Otherwise, it specifies the number of `key` and `value` heads. In this case, `num_heads` must be divisible by `num_key_value_heads`, and the ratio of `num_heads` to `num_key_value_heads` must not exceed 64. The data type is `int64`.
- **`block_size`** (`int`): Optional. Maximum head count of tokens inside each block for KV storage in page attention. The data type can be `int64`. The default value is `0`, which is typically set to values such as `128` or `256`.
- **`inner_precise`** (`int`): Optional. Enables high-precision or high-performance mode. Valid values: `0` (high precision) or `1` (high performance) (default). The data type can be `int64`.

## Return Values

`Tensor`

Final computation result, $atten\_out$ in the formula. The shape of this parameter is identical to that of `query`.

- In non-quantized scenarios, the output data type is identical to that of `query`.
- In quantized scenarios, if `quant_scale2` is provided, the output data type is `int8`.

## Constraints<a name="en-us_topic_0000001711274864_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- The dimensions of `query`, `key`, and `value` must be identical, and the shapes of `key` and `value` must be identical.
- The value of `num_heads` must be equal to $N$ of `query`.
- The value of `input_layout` is determined by the shape of `query`. For a 3D `Tensor`, the layout is $BSH$. For a 4D `Tensor`, the layout can be $BNSD$ or $BSND$.
- The value of `num_key_value_heads` must be equal to $N$ of `key` and `value`, and `num_heads` must be divisible by `num_key_value_heads`.
- Constraints on `query`, `key`, and `value`:
    - Atlas A2 training products/Atlas A2 inference products: The size of the $B$ axis can be less than or equal to `65535`, the size of the $N$ axis can be less than or equal to `256`, the size of the $S$ axis can be less than or equal to `262144`, and the size of the $D$ axis can be less than or equal to `512`.
    - Atlas inference accelerator cards: The size of the $B$ axis can be less than or equal to `256`, the size of the $N$ axis can be less than or equal to `256`, the size of the $S$ axis can be less than or equal to `65536`, and the size of the $D$ axis can be less than or equal to `512` 
    - Scenarios where `query`, `key`, and `value` inputs are all `int8` are currently not supported.

- Comprehensive constraints on the number of `int8` quantization-related parameters and the input or output data formats:

    In scenarios where `query`, `key`, and `value` inputs are `float16` and the output is `int8`, the parameter `quant_scale2` is required, `quant_offset2` is optional, and the parameters `dequant_scale1`, `quant_scale1`, and `dequant_scale2` must not be provided (must be set to `None`).

- Usage constraints for the `pse_shift` feature:
    - The data types of `pse_shift` and `query` must be identical.
    - Only $D$-axis alignment is supported. That is, the $D$ dimension must be divisible by 16.

- Constraints for page attention:
    - The prerequisite for enabling page attention is that `block_table` exists and is valid, and `actual_seq_lengths` corresponding to each batch must be provided. When page attention is enabled, `key` and `value` are arranged in contiguous memory according to the indices in `block_table`. The supported data types of `key` and `value` are `float16`, `bfloat16`, or `int8`.
    - When page attention is enabled, the input KV cache layout must be either `(blocknum, numKvHeads, blocksize, headDims)` or `(blocknum, blocksize, H)`. The value of `blocknum` must not be less than the total number of blocks required by all batches. Generally, the performance of the KV cache layout `(blocknum, numKvHeads, blocksize, headDims)` is superior to that of the KV cache layout `(blocknum, blocksize, H)`.
    - When the page attention feature is enabled, the KV cache layout `(blocknum, numKvHeads, blocksize, headDims)` is supported, but the `query` layout must be $BNSD$ only.
    - When the page attention feature is enabled, if the input KV cache layout is `(blocknum, blocksize, H)` and $H$ (where $H=numKvHeads * headDims$) exceeds 64k, an error is raised and execution is blocked due to hardware instruction constraints.
    - In page attention scenarios, `actual_seq_lengths` must be provided. Each element in `actual_seq_lengths` represents the actual sequence length of its corresponding batch. Dividing this value by the input attribute `block_size` yields the number of blocks required by that batch.
    - In page attention scenarios, `block_table` must be a 2D `Tensor`. The length of the first dimension must equal the batch size, and the length of the second dimension must not be less than `maxBlockNumPerSeq`, where `maxBlockNumPerSeq` is the number of blocks corresponding to the maximum value in `actual_seq_lengths` among all batches. For example, if the batch size is `2`, the attribute `block_size` is `128`, and the `actual_seq_lengths` of each batch is `512`, it indicates that each batch requires at least 4 blocks. Therefore, the layout of `block_table` can be `(2, 4)`.
    - When the page attention feature is enabled, `block_size` is a user-defined parameter whose value affects page attention performance, and is typically set to `128` or `256`. When the input type of `key` and `value` is `float16` or `bfloat16`, `block_size` must be aligned to 16. When the input type of `key` and `value` is `int8`, `block_size` must be aligned to 32. Generally, page attention improves throughput but reduces performance.

- `quant_scale2` and `quant_offset2` form a parameter group, where `quant_offset2` is optional. After this parameter group is provided, the operator output data type is inferred as `int8`. If an `int8` output is not desired, do not provide this parameter group.
- Constraints for KV left padding scenarios:
    - The computation formula for the transport start point of `kvCache` is `Smax - kv_padding_size - actual_seq_lengths`. The computation formula for the transport end point of `kvCache` is `Smax - kv_padding_size`. When the transport start point or end point of `kvCache` is less than 0, the returned data result is filled with all zeros.
    - In KV left padding, if `kv_padding_size` is less than 0, it is set to `0`.
    - To enable KV left padding, both the `kv_padding_size` and `actual_seq_lengths` parameters must be provided. Otherwise, KV right padding are used by default.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import math
    >>>
    >>> # Generate random data and send it to the NPU.
    >>> q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> k = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> v = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
    >>> scale = 1/math.sqrt(128.0)
    >>>
    >>> # Call the IFA operator.
    >>> out = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale)                                        
    >>> out
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[[-1.4863,  0.1667,  0.7256,  ...,  0.3052, -0.3630, -0.1936]],

            [[-1.5840, -2.2305, -0.3462,  ..., -2.1055,  0.4392, -1.2842]]],
        device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    # Configure graph capture
    import torch
    import torch_npu
    import math
    
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"
    
    # Configure logging and debug settings for graph capture
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # Generate data
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

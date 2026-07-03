# torch_npu.npu_grouped_matmul<a name="en-us_topic_0000002229788810"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |
|<term>Atlas inference products</term>   | √  |

## Function<a name="en-us_topic_0000002262888689_section1290611593405"></a>

- Description: Provides an efficient method to perform grouped computation of multiple matrix multiplication (MatMul) operations. This API implements batch processing of multiple MatMul operations by grouping MatMul operations with identical or similar shapes, reducing memory access overhead and saving computational resources to improve computational efficiency. For example:
$y_i[m_i, n_i] =  x_i[m_i, k_i] \times weight_i[k_i, n_i], i = 1 \ldots g$, where $g$ indicates the number of groups. Currently, m-axis and k-axis grouping are supported:
  - m-axis grouping: $k_i$ and $n_i$ are identical across groups. $m_i$ can vary.
  - k-axis grouping: $m_i$ and $n_i$ are identical across groups. $k_i$ can vary.

- Formulas:

     In the following formulas, the $@$ symbol indicates matrix multiplication, and the $\times$ symbol indicates the Hadamard product.

    - Non-quantization scenario (Formula 1):

        $y_i = x_i @ weight_i + bias_i$

    - `perchannel` or `pertensor` quantization scenario (Formula 2):

        $y_i = (x_i @ weight_i) \times scale_i + offset_i$

        - `x` with data type `int8` and `bias` with data type `int32` (Formula 2-1):

            $y_i = (x_i @ weight_i + bias_i) \times scale_i + offset_i$

        - `x` with data type `int8` and `bias` with data type `bfloat16`, `float16`, or `float32`, without `offset` (Formula 2-2):

            $y_i = (x_i @ weight_i) \times scale_i + bias_i$

    - `pertoken`, `pertensor`+`pertensor`, or `pertensor`+`perchannel` quantization scenario (Formula 3):

        $y_i = (x_i @ weight_i + bias_i) \times scale_i \times pertokenscale_i$

        - `x` with data type `int8` and `bias` with data type `int32` (Formula 3-1):

            $y_i = (x_i @ weight_i + bias_i) \times scale_i \times pertokenscale_i$

        - `x` with data type `int8` and `bias` with data type `bfloat16`, `float16`, or `float32` (Formula 3-2):

            $y_i = (x_i @ weight_i) \times scale_i \times pertokenscale_i + bias_i$
        - `x` with data type `int4`, `weight` with data type `int4`, and layout `NZ` (Formula 3-3):

            $y_i=x_i@ (weight_i \times scale_i) \times pertokenscale_i$
        
    - Fake-quantization scenario (Formula 4):

        $y_i = x_i @ ((weight_i + antiquant\_offset_i) \times antiquant\_scale_i) + bias_i$

## Prototype<a name="en-us_topic_0000002262888689_section87878612417"></a>

```python
npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=None, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None) -> List[Tensor]
```

## Parameters<a name="en-us_topic_0000002262888689_section135561610204110"></a>

- **`x`** (`List[Tensor]`): Required. Input matrix list representing the left matrices in MatMul.
    - Valid data types:
        - <term>Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products</term>: `float16`, `float32`, `bfloat16`, `int8`, or `int4`.
        - <term>Atlas inference products</term>: `float16`.

    - The maximum list length is `128`.
    - When `split_item` is set to `0`, each tensor supports 2D to 6D inputs. In other configurations, only 2D inputs are supported.

- **`weight`** (`List[Tensor]`): Required. Weight matrix list representing the right matrices in MatMul.
    - Valid data types:
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
            - When `group_list` is of type `List[int]`, the data type can be `float16`, `float32`, `bfloat16`, or `int8`.
            - When `group_list` is of type `Tensor`, the data type can be `float16`, `float32`, `bfloat16`, `int4`, or `int8`.

        - <term>Atlas inference products</term>: `float16`.

    - The maximum list length is `128`.
    - Each input tensor can be 2D or 3D.

- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`bias`** (`List[Tensor]`): Optional. Independent bias term for each grouped MatMul output.
    - Valid data types:
        - <term>Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products</term>: `float16`, `float32`, or `int32`.
        - <term>Atlas inference products</term>: `float16`.

    - The list length must be identical to that of `weight`.
    - Each input tensor must be 1D.

- **`scale`** (`List[Tensor]`): Optional. Scaling factor in quantization parameters, used to scale original values to match the quantized range, corresponding to Formula (2) and Formula (3).
    - Valid data types:
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
            - When `group_list` is of type `List[int]`, the data type can be `int64`.
            - When `group_list` is of type `Tensor`, the data type can be `float32`, `bfloat16`, or `int64`.

        - <term>Atlas inference products</term>: Only `None` is supported.

    - The list length must be identical to that of `weight`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Each input tensor must be 1D.

- **`offset`** (`List[Tensor]`): Optional. Offset in quantization parameters, used to adjust quantized numerical offsets to more accurately represent original floating-point values, corresponding to Formula (2). Currently, only `None` is supported.
- **`antiquant_scale`** (`List[Tensor]`): Optional. Scaling factor in fake-quantization parameters, used to scale original values to match the fake-quantized range, corresponding to Formula (4).
    - Valid data types:
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: `float16` or `bfloat16`.
        - <term>Atlas inference products</term>: Only `None` is supported.

    - The list length must be identical to that of `weight`.
    - Each tensor supports the following shapes, where $g$ indicates the number of MatMul groups, $G$ indicates the number of `pergroup` partitions, and $G_i$ indicates the number of `pergroup` partitions for the $i$-th tensor.
        - In `perchannel` fake-quantization mode, when `weight` is a single tensor, the shape must be `[g, n]`. When `weight` consists of multiple tensors, the shape must be `[n_i]`.
        - In `pergroup` fake-quantization mode, when `weight` is a single tensor, the shape must be `[g, G, n]`. When `weight` consists of multiple tensors, the shape must be `[G_i, n_i]`.

- **`antiquant_offset`** (`List[Tensor]`): Optional. Offset in fake-quantization parameters, used to adjust fake-quantized numerical offsets to more accurately represent original floating-point values, corresponding to Formula (4).
    - Valid data types:
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: `float16` or `bfloat16`.
        - <term>Atlas inference products</term>: Only `None` is supported.

    - The list length must be identical to that of `weight`.
    - Each tensor must have an identical input dimension to that of `antiquant_scale`.

- **`per_token_scale`** (`List[Tensor]`): Optional. Scaling factor in quantization parameters, used to scale original values to match the quantized range, representing the scaling factor introduced by `x` quantization in `pertoken` quantization parameters, corresponding to Formula (3).
    - When `group_list` is of type `List[int]`, only `None` can be provided.
    - When `group_list` is of type `Tensor`:
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float32`.
        - The list length must be identical to that of `x`.
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Each input tensor must be 1D.

- **`group_list`** (`List[int]`/`Tensor`): Optional. Grouping index indicating the index mapping of MatMul along dimension `0` of `x`. The data type can be `int64`.
    - <term>Atlas inference products</term>: Only the <code>**Tensor**</code> type is supported. Each input tensor must be 1D. The length must be identical to that of `weight`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Both **`List[int]`** and **`Tensor`** types are supported. If it is a **`Tensor`**, it must be 1D, and the length must be identical to that of `weight`.
    - Configuration requirements:
        - When `group_list` is of type `List[int]`, the configuration must be a non-negative strictly increasing sequence, and the length cannot be `1`.
        - When `group_list` is of type `Tensor`:
            - When `group_list_type` is set to `0`, `group_list` must be a non-negative monotonically non-decreasing sequence.
            - When `group_list_type` is set to `1`, `group_list` must be a non-negative sequence, and the length cannot be `1`.
            - When `group_list_type` is set to `2`, the shape of `group_list` must be `[E, 2]`, where `E` indicates the group size. The data layout must be `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`. `groupSize` indicates the size of each group along the grouping axis and must be non-negative.

- **`activation_input`** (`List[Tensor]`): Optional. Backward input of the activation function. Currently, only `None` is supported.
- **`activation_quant_scale`** (`List[Tensor]`): Optional. Reserved parameter. Currently, only `None` is supported.
- **`activation_quant_offset`** (`List[Tensor]`): Optional. Reserved parameter. Currently, only `None` is supported.
- **`split_item`** (`int`): Optional. Specifies the split mode. The data type can be `int32`.
    - `0` or `1`: The output contains multiple tensors, and the number matches that of `weight`.
    - `2` or `3`: The output is a single tensor.

- **`group_type`** (`int`): Optional. The grouping axis. The data type can be `int32`.
    - When `group_list` is of type `List[int]`, only `None` is supported.

    - When `group_list` is of type `Tensor`, and the matrix multiplication is $C[m,n]=A[m,k] \times B[k,n]$, the valid enum values are: `-1` (no grouping), `0` (grouping along the m-axis), and `2` (grouping along the k-axis).
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Currently, the value can be `-1`, `0`, or `2`.
        - <term>Atlas inference products</term>: Currently, only `0` is supported.

- `group_list_type` (`int`): Optional. Representation format of `group_list`. The data type can be `int32`.
    - When `group_list` is of type `List[int]`, only `None` is supported.

    - When `group_list` is of type `Tensor`, valid values are `0`, `1`, or `2`.
        - `0`: Default value. The values in `group_list` indicate the cumulative sum (`cumsum`) results of group sizes along the grouping axis.
        - `1`: The values in `group_list` indicate the size of each group along the grouping axis.
        - `2`: The shape of `group_list` must be `[E, 2]`, where `E` indicates the group size. The data layout must be `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`. `groupSize` indicates the size of each group along the grouping axis.
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value `2` is supported only when the data type of `x` and `weight` is `INT8`, and `group_type` is set to `0` (m-axis grouping).
        - <term>Atlas inference products</term>: The value `2` is not supported.
    
- **`act_type`** (`int`): Optional. Activation function type. The data type can be `int32`.
    - When `group_list` is of type `List[int]`, only `None` is supported.

    - When `group_list` is of type `Tensor`, valid enum values are: `0` (no activation), `1` (`RELU` activation), `2` (`GELU_TANH` activation), `3` (not supported), `4` (`FAST_GELU` activation), or `5` (`SILU` activation).
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value ranges from 0 to 5.
        - <term>Atlas inference products</term>: Currently, only `0` is supported.

- **`output_dtype`** (`torch.dtype`): Optional. Output data type. Valid values are:
    - `None`: Default value. The output data type is identical to that of `x`.
    - A type identical to that of output `y`. For details, see [Constraints](#en-us_topic_0000002262888689_section618392112366).

- **`tuning_config`** (`List[int]`): Optional. The first element in this list specifies the expected number of tokens processed by each expert. During operator tiling, optimization is performed based on this element for higher performance. For details about the usage scenarios, see [Constraints](#en-us_topic_0000002262888689_section618392112366). Elements starting from the second element are reserved for future extensions and can be omitted. This parameter can be omitted if it is not used.
    - <term>Atlas inference products</term>: This parameter is not supported currently.

## Return Values<a name="en-us_topic_0000002262888689_section1558311519405"></a>

`List[Tensor]`:

- When `split_item` is set to `0` or `1`, the number of returned tensors is identical to that of `weight`.
- When `split_item` is set to `2` or `3`, a single tensor is returned.

## Constraints<a name="en-us_topic_0000002262888689_section618392112366"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The inner dimension limit `InnerLimit` is fixed at `65536`.
- For each tensor group in `x` and `weight`, the size of the last dimension must be less than `InnerLimit`. The last dimension of <code>x<sub>i</sub></code> indicates the K-axis of <code>x<sub>i</sub></code> when `x` is not transposed, or the M-axis of <code>x<sub>i</sub></code> when `x` is transposed. The last dimension of <code>weight<sub>i</sub></code> indicates the $K$ axis of <code>weight<sub>i</sub></code> when `weight` is not transposed, or the $N$ axis of <code>weight<sub>i</sub></code> when `weight` is transposed.

- Usage scenario constraints for `tuning_config`:

    This parameter can be used only in quantization scenarios (where the inputs are `int8`, the outputs can be `int32`, `bfloat16`, `float16`, or `int8`) and only in single-tensor single-expert configurations.

    |x| weight|output_dtype|y|
    |---------|--------|--------|--------|
    |`int8`|`int8`|`int8`|`int8`|
    |`int8`|`int8`|`bfloat16`|`bfloat16`|
    |`int8`|`int8`|`float16`|`float16`|
    |`int8`|`int8`|`int32`|`int32`|

- Constraints on the input and output data types in different scenarios:
    - **When `group_list` is of type `List[int]`**: Data type constraints for <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term> apply.

        **Table 1** Data type constraints

        |Scenario|x|weight|bias|scale|antiquant_scale|antiquant_offset|output_dtype|y|
        |---------|--------|--------|--------|--------|--------|--------|--------|--------|
        |Non-quantization|`float16`|`float16`|`float16`|Not required|Not required|Not required|`float16`|`float16`|
        |Non-quantization|`bfloat16`|`bfloat16`|`float32`|Not required|Not required|Not required|`bfloat16`|`bfloat16`|
        |Non-quantization|`float32`|`float32`|`float32`|Not required|Not required|Not required|`float32`|`float32`|
        |`perchannel` full quantization|`int8`|`int8`|`int32`|`int64`|Not required|Not required|`int8`|`int8`|
        |Fake-quantization|`float16`|`int8`|`float16`|Not required|`float16`|`float16`|`float16`|`float16`|
        |Fake-quantization|`bfloat16`|`int8`|`float32`|Not required|`bfloat16`|`bfloat16`|`bfloat16`|`bfloat16`|

    - **When `group_list` is of type `Tensor`**: Data type constraints for the following products apply:
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>

            **Table 2** Data type constraints

            |Scenario|x|weight|bias|scale|antiquant_scale|antiquant_offset|per_token_scale|output_dtype|y|
            |---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
            |Non-quantization|`float16`|`float16`|`float16`|Not required|Not required|Not required|Not required|None/`float16`|`float16`|
            |Non-quantization|`bfloat16`|`bfloat16`|`float32`|Not required|Not required|Not required|Not required|None/`bfloat16`|`bfloat16`|
            |Non-quantization|`float32`|`float32`|`float32`|Not required|Not required|Not required|Not required|`None`/`float32` (only when `x`/`weight`/`y` is a single tensor)|`float32`|
            |`perchannel` full quantization|`int8`|`int8`|`int32`|`int64`|Not required|Not required|Not required|None/`int8`|`int8`|
            |`perchannel` full quantization|`int8`|`int8`|`int32`|`bfloat16`|Not required|Not required|Not required|`bfloat16`|`bfloat16`|
            |`perchannel` full quantization|`int8`|`int8`|`int32`|`float32`|Not required|Not required|Not required|`float16`|`float16`|
            |`pertoken` full quantization|`int8`|`int8`|`int32`|`bfloat16`|Not required|Not required|`float32`|`bfloat16`|`bfloat16`|
            |`pertoken` full quantization|`int8`|`int8`|`int32`|`float32`|Not required|Not required|`float32`|`float16`|`float16`|
            |`pertoken` full quantization|`int4`|`int4`|Not required|`uint64`|Not required|Not required|None/`float32`|`float16`|`float16`|
            |`pertoken` full quantization|`int4`|`int4`|Not required|`uint64`|Not required|Not required|None/`float32`|`bfloat16`|`bfloat16`|
            |Fake-quantization|`float16`|`int8`/`int4`|`float16`|Not required|`float16`|`float16`|Not required|None/`float16`|`float16`|
            |Fake-quantization|`bfloat16`|`int8`/`int4`|`float32`|Not required|`bfloat16`|`bfloat16`|Not required|None/`bfloat16`|`bfloat16`|

            > [!NOTE]   
            > - In fake-quantization scenarios, if the data type of `weight` is `int8`, only `perchannel` mode is supported. If the data type of `weight` is `int4`, both `perchannel` and `pergroup` modes are supported. In `pergroup` mode, the `pergroup` count $G$ or $G_i$ must exactly divide the corresponding $k_i$. If `weight` consists of multiple tensors, the `pergroup` length is defined as $s_i= k_i/G_i$, and all $s_i(i=1,2,...g)$ must be equal.
            > - In fake-quantization scenarios, if the data type of `weight` is `int4`, the size of the last dimension for each tensor group in `weight` must be an even number. The last dimension of <code>weight<sub>i</sub></code> indicates the $N$ axis of <code>weight<sub>i</sub></code> when `weight` is not transposed, or the $K$ axis of <code>weight<sub>i</sub></code> when `weight` is transposed. In `pergroup` mode, when `weight` is transposed, the `pergroup` length $s_i$ must be an even number. A transposed tensor refers to a configuration where a tensor with shape `[M, K]` has a stride of `[1, M]` and a data layout of `[K, M]`. That is, it indicates a non-contiguous tensor.
            > - PyTorch does not natively support the `int4` data type. To use `int4`, you can represent it using `int32` data through the [torch_npu.npu_quantize](torch_npu-npu_quantize.md) API.

        - <term>Atlas inference products</term>:

            **Table 3** Data type constraints

            |x|weight|bias|scale|antiquant_scale|antiquant_offset|per_token_scale|output_dtype|y|
            |--------|--------|--------|--------|--------|--------|--------|--------|--------|
            |`float16`|`float16`|`float16`|Not required|Not required|Not required|`float32`|`float16`|`float16`|
            
- The following scenarios are supported based on the variations in the tensor counts of input `x`, input `weight`, and output `y`. In these scenario descriptions, "single" indicates a single tensor, and "multiple" indicates multiple tensors. The naming sequence follows the order of `x`, `weight`, and `y`. For example, "single-multiple-single" indicates that `x` is a single tensor, `weight` is a list of tensors, and `y` is a single tensor.
    - **When `group_list` is of type `List[int]`**: constraints for <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term> apply.

        |Supported Scenario|Description|Constraints|
        |--------|--------|--------|
        |Multiple-Multiple-Multiple|`x`, `weight`, and `y` are lists of tensors. The tensors of each data group are independent.|1. `split_item` must be `0` or `1`.<br>2. The tensors in `x` must be identical in dimension and can be 2D to 6D. The tensors in `weight` must be 2D. The tensor dimensions in `y` must be identical to those of `x`.<br>3. When the tensors in `x` are greater than 2D, `group_list` must be omitted.<br>4. When `x` contains 2D tensors and `group_list` is provided, the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `x`.|
        |Single-Multiple-Single|`x` and `y` are single tensors, and `weight` is a list of tensors.|1. `split_item` must be `2` or `3`.<br>2. `group_list` must be provided, and its last value must be identical to the first dimension of the tensor in `x`.<br>3. The tensors in `x`, `weight`, and `y` must be 2D.<br>4. The N-axis size of each tensor in `weight` must be identical.|
        |Single–Multiple–Multiple|`x` is a single tensor, and `weight` and `y` are lists of tensors.|1. `split_item` must be `0` or `1`.<br>2. `group_list` must be provided, and the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `y`.<br>3. The tensors in `x`, `weight`, and `y` must be 2D.|
        |Multiple–Multiple–Single|`x` and `weight` are lists of tensors, and `y` is a single tensor. The results of each matrix multiplication group are stored sequentially within the same continuous tensor.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `x`, `weight`, and `y` must be 2D.<br>3. The N-axis size of each tensor in `weight` must be identical.<br>4. If `group_list` is provided, the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `x`.|
        
    - **When `group_list` is of type `Tensor`**: constraints for the following products apply.
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>

            > [!NOTE]   
            > - Quantization and fake-quantization are supported only when `group_type` is set to `-1` or `0`.
            > - Activation function computation is supported only in `pertoken` quantization scenarios.

            |group_type|Supported Scenario|Description|Constraints|
            |--------|--------|--------|--------|
            |-1|Multiple-Multiple-Multiple|`x`, `weight`, and `y` are lists of tensors. The tensors of each data group are independent.|1. `split_item` must be `0` or `1`.<br>2. The tensors in `x` must be identical in dimension and can be 2D to 6D. The tensors in `weight` must be 2D. The tensor dimensions in `y` must be identical to those of `x`.<br>3. `group_list` must be omitted.<br>4. Transposition of `weight` is supported, but the transpose configuration of each tensor in `weight` must be identical.<br>5. `x` does not support transposition.|
            |0|Single–Single–Single|`x`, `weight`, and `y` are all single tensors.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>3. `group_list` must be provided. When `group_list_type` is set to `0`, its last value must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `1`, the sum of its values must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `2`, the sum of the values in its second column must be identical to the first dimension of the tensor in `x`.<br>4. The maximum size of the first dimension of `group_list` is `1024`. That is, a maximum of `1024` groups are supported.<br>5. Transposition of `weight` is supported.<br>6. `x` does not support transposition.|
            |0|Single-Multiple-Single|`x` and `y` are single tensors, and `weight` is a list of tensors.|1. `split_item` must be `2` or `3`.<br>2. `group_list` must be provided. When `group_list_type` is set to `0`, its last value must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `1`, the sum of its values must be identical to the first dimension of the tensor in `x`, and the maximum length is `128`. When `group_list_type` is set to `2`, the sum of the values in its second column must be identical to the first dimension of the tensor in `x`, and the maximum length is `128`.<br>3. The tensors in `x`, `weight`, and `y` must be 2D.<br>4. The N-axis size of each tensor in `weight` must be identical.<br>5. Transposition of `weight` is supported, but the transpose configuration of each tensor in `weight` must be identical.<br>6. `x` does not support transposition.|
            |0|Multiple–Multiple–Single|`x` and `weight` are lists of tensors, and `y` is a single tensor. The results of each matrix multiplication group are stored sequentially within the same continuous tensor.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `x`, `weight`, and `y` must be 2D.<br>3. The N-axis size of each tensor in `weight` must be identical.<br>4. If `group_list` is provided, when `group_list_type` is set to `0`, the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `x`. When `group_list_type` is set to `1`, the values of `group_list` must correspond one-to-one to the first dimension of each tensor in `x`, and the maximum length is `128`. When `group_list_type` is set to `2`, the values in the second column of `group_list` must correspond one-to-one to the first dimension of each tensor in `x`, and the maximum length is `128`.<br>5. Transposition of `weight` is supported, but the transpose configuration of each tensor in `weight` must be identical.<br>6. `x` does not support transposition.|

        - <term>Atlas inference products</term>

            The inputs and outputs support only the `float16` data type. The N-axis size of the output `y` must be a multiple of `16`.

            |group_type|Supported Scenario|Description|Constraints|
            |--------|--------|--------|--------|
            |0|Single-Single-Single|`x`, `weight`, and `y` are all single tensors.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>3. `group_list` must be provided. When `group_list_type` is set to `0`, its last value must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `1`, the sum of its values must be identical to the first dimension of the tensor in `x`.<br>4. The maximum size of the first dimension of `group_list` is `1024`. That is, a maximum of `1024` groups are supported.<br>5. Transposition of `weight` is supported, but `x` does not support transposition.|

## Examples<a name="en-us_topic_0000002262888689_section1566973054111"></a>

- Single-operator call

    - General call

        ```python
        import torch
        import torch_npu
        
        x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
        x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
        x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
        x = [x1, x2, x3]
        
        weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
        weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
        weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
        weight = [weight1, weight2, weight3]
        
        bias1 = torch.randn(256, device='npu', dtype=torch.float16)
        bias2 = torch.randn(1024, device='npu', dtype=torch.float16)
        bias3 = torch.randn(128, device='npu', dtype=torch.float16)
        bias = [bias1, bias2, bias3]
        
        group_list = None
        split_item = 0
        npu_out = torch_npu.npu_grouped_matmul(x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=-1)
        ```

    - Example with x of type int4, weight of type int4, and layout NZ:

        ```python
        import numpy as np
        import torch
        import torch_npu

        E, K, N = 1, 16, 64
        x = torch.randint(10, (15, 16), dtype=torch.int8).npu()
        weight = torch.randint(10, (1, 16, 64), dtype=torch.int8).npu()

        x_quant = torch_npu.npu_quantize(x.to(torch.float32), torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
        weight_nz = torch_npu.npu_format_cast(weight.to(torch.float32), 29)
        weight_quant = torch_npu.npu_quantize(weight_nz, torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)

        scale = torch.rand((E, 1, N), dtype=torch.float32).npu()

        k_group = scale.shape[1]
        scale_np = scale.cpu().numpy()
        scale_uint32 = scale_np.astype(np.float32)
        scale_uint32.dtype = np.uint32
        scale_uint64 = np.zeros((E, k_group, N * 2), dtype=np.uint32)
        scale_uint64[...,::2] = scale_uint32
        scale_uint64.dtype = np.int64
        scale = torch.from_numpy(scale_uint64).npu()

        group_list = torch.Tensor([14]).to(torch.int64).npu()
        per_token_scale = torch.rand((15), dtype=torch.float32).npu()

        output = torch_npu.npu_grouped_matmul([x_quant], [weight_quant], scale=[scale], per_token_scale=[per_token_scale],
                                                group_list=group_list, group_list_type=0, group_type=0,
                                                split_item=3, output_dtype=torch.float16)
        ```

- Graph mode call
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>

        ```python
        import torch
        import torch.nn as nn
        import torch_npu
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        
        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)
        
        class GMMModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, weight):
                return torch_npu.npu_grouped_matmul(x, weight, group_type=-1)
        
        def main():
            x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
            x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
            x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
            x = [x1, x2, x3]
            
            weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
            weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
            weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
            weight = [weight1, weight2, weight3]
            
            model = GMMModel().npu()
            model = torch.compile(model, backend=npu_backend, dynamic=False)
            custom_output = model(x, weight)
        
        if __name__ == '__main__':
            main()
        ```

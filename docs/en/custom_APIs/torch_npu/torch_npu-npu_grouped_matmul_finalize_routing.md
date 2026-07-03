# torch_npu.npu_grouped_matmul_finalize_routing

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function<a name="en-us_topic_0000002259406069_section14441124184110"></a>

- **API function**: Fuses the `GroupedMatMul` and `MoeFinalizeRouting` operators. `GroupedMatMul` handles per-expert computation, and `MoeFinalizeRouting` restructures and aggregates the results into the original token order based on routing relationships. Together, they connect in series to form a complete MoE output pipeline.
  - **MoE**: a Mixture of Experts model. Each token is assigned to one or more experts for computation based on routing results.
  - **`MoeFinalizeRouting`**: Final stage of MoE routing. It maps the computation results of each expert back to the original token order based on routing indices and aggregates multi-expert results for each token to generate the final output. 

`GroupedMatMul` provides an optimized computation pattern for batched sparse matrix multiplications. Within an MoE architecture, each token is assigned to a specific expert, and different experts process varying numbers of tokens. `GroupedMatMul` aggregates all tokens assigned to the same expert into a single group and executes the matrix multiplication for that expert in a single operation. This eliminates the scheduling overhead of invoking individual matrix multiplications for each token, improving computational efficiency.

The inputs to `GroupedMatMul` typically consist of the input tensor `x`, expert weights `w`, and grouping information `group_list`. The shape of `x` is `(M, K)`, where $M$ indicates the total number of tokens and $K$ indicates the input feature dimension. The expert weights `w` can be represented as a tensor containing $E$ expert weight matrices. The `group_list` describes the number of tokens processed by each expert or the prefix sums of group sizes.

During computation, the operator partitions the rows of `x` into multiple continuous groups based on `group_list`. For each expert, the corresponding token group is extracted and multiplied by the expert weight matrix to generate the expert output. The outputs of all experts are concatenated in group order to form the final `GroupedMatMul` output.

Subsequently, `MoeFinalizeRouting` reorders and combines the `GroupedMatMul` output based on `row_index` to match target token positions. When `logit` is provided, each expert output is first multiplied by the corresponding token logit weight before combination. When `shared_input` is provided, the shared expert output is accumulated using `shared_input_weight` and `shared_input_offset`.

This fused operator is applicable to FFN layers of MoE models, high-performance inference, and dynamic batching scenarios. It reduces zero padding, lowers operator scheduling overhead, and improves NPU throughput.

## Prototype<a name="en-us_topic_0000002259406069_section45077510411"></a>

```python
torch_npu.npu_grouped_matmul_finalize_routing(x, w, group_list, *, scale=None, bias=None, offset=None, pertoken_scale=None, shared_input=None, logit=None, row_index=None, dtype=None, shared_input_weight=1.0, shared_input_offset=0, output_bs=0, group_list_type=1) -> Tensor
```

## Parameters<a name="en-us_topic_0000002259406069_section112637109429"></a>

- **`x`** (`Tensor`): Required. Left matrix for matrix computation. Non-contiguous tensors are not supported. The data type can be `int8`. The data layout can be ND. The shape of this parameter is `(m, k)`. The value range of `m` is [1, 16 \* 1024 \* 8].
- **`w`** (`Tensor`): Required. Right matrix for matrix computation. Non-contiguous tensors are not supported. The data type can be `int8` or `int4`.
    - In A8W8 quantization scenarios, the data layout can be NZ, and the shape of this parameter is `(e, n1, k1, k0, n0)`, where `k0` is fixed at `16` and `n0` is fixed at `32`. The `k` dimension in the shape of `x` and the `k1` dimension in the shape of `w` must satisfy the equation: $\mathrm{ceilDiv}(k, 16) = k1$. The value of `e` must be in the range [1, 256]. The value of `k` must be a multiple of `16`. The value of `n` must be a multiple of `32` and greater than or equal to `256`.
    - In A8W4 quantization scenarios, the data layout can be ND, and the shape of this parameter is `(e, k, n)`, where `k` is fixed at `2048` and `n` is fixed at `7168`.

- **`group_list`** (`Tensor`): Required. Group sizes for `GroupedMatMul`. Non-contiguous tensors are not supported. The data type can be `int64`. The data layout can be ND. The shape of this parameter is `(e,)`, where `e` is identical to that of `w`. The sum of all elements in `group_list` must be less than or equal to `m`.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`scale`** (`Tensor`): Optional. Dequantization parameter for the weight matrix. In A8W8 scenarios, per-channel quantization is supported. Non-contiguous tensors are not supported. The data type can be `float32`. The data layout can be ND. The shape of this parameter is `(e, n)`, where $n = n1 \times n0$. In A8W4 quantization scenarios, the data type can be `int64`, and the shape is `(e, 1, n)`.
- **`bias`** (`Tensor`): Optional. Bias parameter for matrix computation. Non-contiguous tensors are not supported. The data type can be `float32`. The data layout can be ND. The shape of this parameter is `(e, n)`. This parameter is supported only in A8W4 quantization scenarios.
- **`offset`** (`Tensor`): Optional. Offset for matrix quantization parameters. Non-contiguous tensors are not supported. The data type can be `float32`. The data layout can be ND. This parameter must be 3D. This parameter is supported only in A8W4 quantization scenarios.
- **`pertoken_scale`** (`Tensor`): Optional. Dequantization parameter for the `x` matrix under per-token quantization. Non-contiguous tensors are not supported. The shape of this parameter is `(m,)`, where `m` is identical to that of `x`. The data type can be `float32`. The data layout can be ND.
- **`shared_input`** (`Tensor`): Optional. Output of the shared expert in MoE computation, which must be combined with the MoE expert output. Non-contiguous tensors are not supported. The data type can be `bfloat16`. The data layout can be ND. This shape of this parameter is `(batch/dp, n)`, where `n` is identical to that of `scale`. The value range of `batch/dp` is [1, 2 \* 1024]. The value range of `batch` is [1, 16 \* 1024].
- **`logit`** (`Tensor`): Optional. Per-token logit values from MoE experts. The output of matrix multiplication is multiplied by these logit values and then combined based on indices. Non-contiguous tensors are not supported. The data type can be `float32`. The data layout can be ND. The shape of this parameter is `(m,)`, where `m` is identical to that of `x`.
- **`row_index`** (`Tensor`): Optional. Output of MoE experts is combined based on this row index, where the values serve as indices for scatter-add combination. Non-contiguous tensors are not supported. The data type can be `int32` or `int64`. The data layout can be ND. The shape of this parameter is `(m,)`, where `m` is identical to that of `x`.
- **`dtype`** (`ScalarType`): Optional. Output type of `GroupedMatMul` computation. The data type must be `float32` (default).
- **`shared_input_weight`** (`float`): Optional. Factor for combining shared expert output with MoE expert output. The `shared_input` is multiplied by this parameter before accumulating with the MoE expert results. The default value is `1.0`.
- **`shared_input_offset`** (`int`): Optional. Row offset for combining shared expert and MoE expert outputs. The default value is `0`, indicating no offset.
- **`output_bs`** (`int`): Optional. Maximum size of the output batch dimension. The default value is `0`.
- **`group_list_type`** (`int`): Optional. Grouping mode for `GroupedMatMul`. Valid values are: `1` (count mode) or `0` (cumsum mode, representing the prefix sum). The default value is `1`.

## Return Values<a name="en-us_topic_0000002259406069_section22231435517"></a>

`Tensor`

Return value. Non-contiguous tensors are not supported. The output data type is fixed at `float32`. The shape of this parameter is `(batch, n)`.

## Constraints<a name="en-us_topic_0000002259406069_section12345537164214"></a>

- This API can be used in inference and training scenarios.
- This API supports graph mode.
- The following table lists the data type combinations supported by the input and output tensors.

    |x|w|group_list|scale|bias|offset|pertoken_scale|shared_input|logit|row_index|y|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |`int8`|`int8`|`int64`|`float32`|None|None|`float32`|`bfloat16`|`float32`|`int64`|`float32`|
    |`int8`|`int8`|`int64`|`float32`|None|None|`float32`|None|`float32`|`int64`|`float32`|
    |`int8`|`int4`|`int64`|`int64`|`float32`|None|`float32`|`bfloat16`|`float32`|`int64`|`float32`|
    |`int8`|`int4`|`int64`|`int64`|`float32`|`float32`|`float32`|`bfloat16`|`float32`|`int64`|`float32`|

## Examples<a name="en-us_topic_0000002259406069_section14459801435"></a>

- Single-operator call

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

    torch_npu.npu.config.allow_internal_format = True

    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8
     
    x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
    weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
     
    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch
    y = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNz,
                group_list_clone, scale=scale_clone, pertoken_scale=pertoken_scale_clone,
                shared_input=shared_input_clone, logit=logit_clone, row_index=row_index_clone,
                shared_input_offset=shared_input_offset, output_bs=output_bs)
    ```

- Graph mode call

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
     
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
     
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs):
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,
                        scale=scale, pertoken_scale=pertoken_scale, shared_input=shared_input,
                        logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs)
            return output
     
    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8
     
    x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
    weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
     
    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weightNz, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs)
    ```
    
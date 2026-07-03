# torch_npu.npu_moe_gating_top_k

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |

## Function

- Description: Performs Sigmoid or Softmax computation on the input `x` during MoE computation, groups and sorts the computation results, and selects the top-K experts based on the group sorting results.
- Formulas:

    Performs Sigmoid computation on the input (the `bias` parameter is optional):
    - When `norm_type` is `1`:

      ![](../../figures/en-us_formulaimage_0000002258672873.png)

    - When `norm_type` is `0`:

      ![](../../figures/en-us_formulaimage_0000002313785750.png)

    Groups the computation results based on `group_count`:
    
    - When `group_select_mode` is `1`, the groups are sorted based on the sum of the top-k2 values within each group, and the top `k_group` groups are selected.

      ![](../../figures/en-us_formulaimage_0000002219010398.png)

    - When `group_select_mode` is `0`, the groups are sorted based on the maximum value within each group, and the top `k_group` groups are selected.

      ![](../../figures/en-us_formulaimage_0000002347834049.png)

    Obtains the corresponding elements from `normOut` based on the group IDs retrieved from the previous step, and executes a TopK operation on the data to generate the `expertIdxOut` result.

    ![](../../figures/en-us_formulaimage_0000002219172722.png)

    Computes `yOut` from `y` using the provided `routed_scaling_factor` and `eps` parameters to generate the final result.

    ![](../../figures/en-us_formulaimage_0000002219173660.png)

- Equivalent computation logic:

    ```python
    import torch
    import numpy

    def moe_gating_top_k_numpy(x: torch.tensor, k: int, *, bias: torch.tensor = None, k_group: int = 1, group_count: int = 1,
                                group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, out_flag: bool = False,
                                routed_scaling_factor: float = 1.0, eps: float = 1e-20) -> tuple:
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            bias = bias.to(dtype=torch.float32)

        x = x.numpy()
        bias = bias.numpy()
        if norm_type == 0:
            x = numpy.exp(x - numpy.expand_dims(numpy.log(numpy.sum(numpy.exp(x),
                            axis=-1, keepdims=True)), axis=-1))  # softmax
        else:
            x = 1 / (1 + numpy.exp(-x))  # sigmoid
        original_x = x
        if bias is not None:
            x = x + bias
            
        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = numpy.amax(x, axis=-1)
            else:
                group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
        indices = numpy.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]  # Indices of top-k_group

        mask = numpy.ones((x.shape[0], group_count), dtype=bool)  # Create a mask with all 1
        mask[numpy.arange(x.shape[0])[:, None], indices] = False  # Set to false at the indices
        x = numpy.where(mask[..., None], float('-inf'), x)  # Fill with -inf when mask value is true
        x = x.reshape(x.shape[0], -1)

        indices = numpy.argsort(-x, axis=-1, kind='stable')
        indices = indices[:, :k]
        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type == 1:
            y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor
        if out_flag:
            out = original_x
        else:
            out = None

        y = torch.tensor(y, dtype=dtype)
        return y, indices.astype(numpy.int32), out   

    k = 6
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
        

    x = numpy.random.uniform(-2, 2, (8, 256)).astype(numpy.float32)
    bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    y, expert_idx, out = moe_gating_top_k_numpy(x_tensor, k, bias = bias_tensor, k_group = k_group, group_count = group_count,
                                group_select_mode = group_select_mode, renorm = renorm, norm_type = norm_type, out_flag = out_flag,
                                routed_scaling_factor = routed_scaling_factor, eps = eps)

    print(f"yOut shape: {y.shape}")              
    print(f"expertIdxOut shape: {expert_idx.shape}")  
    print(f"Selected experts: {expert_idx[0]}")
    ```

## Prototype

```python
npu_moe_gating_top_k(x, k, *, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input to be processed. This parameter must be a 2D tensor. The data type can be `float16`, `bfloat16`, or `float32`. The data layout must be ND. Non-contiguous tensors are supported. The size of the last dimension (the expert count) must be less than or equal to `2048`.

- **`k`** (`int`): Required. Number of experts finally selected for each token. The data type is `int64`. The condition `1 <= k <= x_shape[-1] / group_count * k_group` must be satisfied.

- **`*`**: Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
 
- **`bias`** (`Tensor`): Optional. This parameter can be set to `None`. Bias values used together with the input `x` in the computation. This parameter must be a 1D tensor whose shape matches the size of the last dimension of `x`. The data type can be `float16`, `bfloat16`, or `float32`. The data type must be identical to that of `x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`k_group`** (`int`): Optional. Number of expert groups selected during the expert group selection process for each token. The data type is `int64`. The default value is `1`. The conditions `1 <= k_group <= group_count` and `k_group * x_shape[-1] / group_count >= k` must be satisfied.

- **`group_count`** (`int`): Optional. Number of groups into which all experts are divided. The data type is `int64`. The default value is `1`. `group_count` must be greater than `0`. `x_shape[-1]` must be divisible by `group_count`. The result of the division must be greater than `2`. After aligning the division result to a multiple of 32, the aligned value multiplied by `group_count` must be less than or equal to `2048`.

- **`group_select_mode`** (`int`): Optional. Score calculation method for each expert group. Valid values: `0` (uses the maximum value within the group as the group score) or `1` (uses the sum of the scores of the top two experts within the group as the group score). The default value is `0`. 

- **`renorm`** (`int`): Optional. Renormalisation flag. The default value is `0` (performs normalization before the TopK computation). Currently, only the value `0` is supported.
- **`norm_type`** (`int`): Optional. Type of the normalization function. Valid values: `1` (enables the Softmax function) or `0` (enables the Sigmoid function). The default value is `0`.

- **`out_flag`** (`bool`): Optional. Specifies whether to output the intermediate result of the normalization operation. The default value is `False`.
- **`routed_scaling_factor`** (`float`): Optional. `routed_scaling_factor` coefficient used to compute `yOut`. The default value is `1.0`.
- **`eps`** (`float`): Optional. `eps` coefficient used to compute `yOut`. The default value is `1e-20`.

## Return Values

- **`yOut`** (`Tensor`): Result obtained after performing normalization, group-based sorting, and TopK selection on `x`. This parameter must be a 2D tensor. The data type can be `float16`, `bfloat16`, or `float32`. The data type must be identical to that of `x`. The data layout must be ND. The size of the first dimension must match that of `x`. The size of the last dimension must match `k`. Non-contiguous tensors are not supported.
- **`expertIdxOut`** (`Tensor`): Indices of the experts selected after normalization, group-based sorting, and TopK selection on `x`. That is, the expert IDs. The shape must be identical to the shape of `yOut`. The data type is `int32`. The data layout must be ND. Non-contiguous tensors are not supported.
- **`normOut`** (`Tensor`): Output of the normalization computation. The shape must be identical to the shape of `x`. The data type is `float32`. The data layout must be ND. Non-contiguous tensors are not supported.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy
    
    k = 1
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
    
    # Generate random data and send it to the NPU.
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # Call the MoeGatingTopK operator.
    y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    ```

- Graph mode call

    ```python
    # Configure graph capture
    import torch
    import torch_npu
    import torchair
    import numpy
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_tensor, bias_tensor):
            return torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    # Instantiate the model.
    model = Model().npu()
    # Obtain the default backend provided by the NPU from TorchAir.
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the backend of TorchAir to call the compile API to compile the model.
    model = torch.compile(model, backend=npu_backend)
    
    k = 1
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
    
    # Generate random data and send it to the NPU.
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # Call the MoeGatingTopK operator.
    y_npu, expert_idx_npu, out_npu = model(x_tensor, bias_tensor)
    ```
    
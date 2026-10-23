# torch_npu.npu_moe_gating_top_k

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- Function: in the MoE computation, performs the Sigmoid or Softmax computation on the input x, sorts the computation results by group, and finally selects the top k experts according to the group sorting result.
- Formulas:

    Apply sigmoid to the input (`bias` optional):
    - When `norm_type` is 1:

      ![](../../figures/en-us_formulaimage_0000002258672873.png)

    - When `norm_type` is 0:

      ![](../../figures/en-us_formulaimage_0000002313785750.png)

    Group the computation results by `group_count`:
    
    - If `group_select_mode` is 1, the groups are sorted by the sum of topk2, and the first `k_group` groups are selected:

      ![](../../figures/en-us_formulaimage_0000002219010398.png)

    - If `group_select_mode` is 0, the groups are sorted by the maximum value in each group, and the first `k_group` groups are selected:

      ![](../../figures/en-us_formulaimage_0000002347834049.png)

    Obtain the corresponding elements in `normOut` according to the groupId obtained in the previous step, and perform TopK again on the data to obtain the result of `expertIdxOut`:

    ![](../../figures/en-us_formulaimage_0000002219172722.png)

    Compute y according to the input `routed_scaling_factor` and `eps` parameters to obtain the result of yOut:

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
npu_moe_gating_top_k(x, k, *, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input to be processed. This parameter must be a 2D tensor. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout must be ND. Non-contiguous tensors are supported. The size of the last dimension (the expert count) must be less than or equal to `2048`.

- **`k`** (`int`): Required. Number of experts finally selected for each token. The data type is `torch.int64`. The condition `1 <= k <= x_shape[-1] / group_count * k_group` must be satisfied.

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
 
- **`bias`** (`Tensor`): Optional. This parameter can be set to `None`. Bias values used together with the input `x` in the computation. This parameter must be a 1D tensor whose shape matches the size of the last dimension of `x`. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data type must be identical to that of `x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`input_ids`** (`Tensor`): Optional. Input index in hash mode, used to look up the expert index from `tid2eid`. It must be a 1D Tensor, and its shape value must be equal to the first dimension of `x`. The data type can be `torch.int32` or `torch.int64`.

- **`tid2eid`** (`Tensor`): Optional. Hash mapping table storing precomputed expert indices. It must be a 2D Tensor, and the last dimension of its shape must be equal to `k`. The data type can be `torch.int32` or `torch.int64`.

- **`k_group`** (`int`): Optional. Number of expert groups selected during the expert group selection process for each token. The data type is `torch.int64`. The default value is `1`. The conditions `1 <= k_group <= group_count` and `k_group * x_shape[-1] / group_count >= k` must be satisfied.

- **`group_count`** (`int`): Optional. Number of groups into which all experts are divided. The data type is `torch.int64`. The default value is `1`. `group_count` must be greater than `0`. `x_shape[-1]` must be divisible by `group_count`. The result of the division must be greater than `2`. After aligning the division result to a multiple of 32, the aligned value multiplied by `group_count` must be less than or equal to `2048`.

- **`group_select_mode`** (`int`): Optional. Score calculation method for each expert group. Valid values: `0` (uses the maximum value within the group as the group score) or `1` (uses the sum of the scores of the top two experts within the group as the group score). The default value is `0`. 

- **`renorm`** (`int`): Optional. Renormalization flag. The default value is `0`. The values `0` and `1` are supported. `0` indicates that renorm is not performed, and `1` indicates that renorm is performed. This parameter takes effect only when `norm_type` is `0`; it is invalid when `norm_type` is `1` or `2`.
- **`norm_type`** (`int`): Optional. Type of the normalization function. Valid values: `1` (enables the Softmax function) or `0` (enables the Sigmoid function). The default value is `0`.

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the supported values are `0` and `1`. `0` indicates that the Softmax function is used, and `1` indicates that the Sigmoid function is used.
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT products</term>: the supported values are `0`, `1`, and `2`. `0` indicates that the Softmax function is used, `1` indicates that the Sigmoid function is used, and `2` indicates that the SqrtSoftplus function is used.
    <!-- end id5 -->

- **`out_flag`** (`bool`): Optional. Specifies whether to output the intermediate result of the normalization operation. The default value is `False`.
- **`routed_scaling_factor`** (`float`): Optional. `routed_scaling_factor` coefficient used to compute `yOut`. The default value is `1.0`.
- **`eps`** (`float`): Optional. `eps` coefficient used to compute `yOut`. The default value is `1e-20`.

## Return Values

- **`yOut`** (`Tensor`): Result obtained after performing normalization, group-based sorting, and Top-K selection on `x`. This parameter must be a 2D tensor. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data type must be identical to that of `x`. The data layout must be ND. The size of the first dimension must match that of `x`. The size of the last dimension must match `k`. Non-contiguous tensors are not supported.
- **`expertIdxOut`** (`Tensor`): Indices of the experts selected after normalization, group-based sorting, and Top-K selection on `x`. That is, the expert IDs. The shape must be identical to the shape of `yOut`. The data type is `torch.int32`. The data layout must be ND. Non-contiguous tensors are not supported.
- **`normOut`** (`Tensor`): Output of the normalization computation. The shape must be identical to the shape of `x`. The data type is `torch.float32`. The data layout must be ND. Non-contiguous tensors are not supported.

## Constraints

- This API can be used only in inference scenarios.
- This API supports single-operator mode and graph mode calls.

## Example

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
    
    # Generate random data and send it to the npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # Call the MoeGatingTopK operator
    y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    ```

- Graph mode call

    ```python
    # Graph mode approach
    import torch
    import torch_npu
    import torchair
    import numpy
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_tensor, bias_tensor):
            return torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    # Instantiate the model
    model = Model().npu()
    # Obtain the default backend provided by the NPU from TorchAir
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to call the compile API to compile the model
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
    
    # Generate random data and send it to the npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # Call the MoeGatingTopK operator
    y_npu, expert_idx_npu, out_npu = model(x_tensor, bias_tensor)
    ```

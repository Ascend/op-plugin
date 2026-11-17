# torch_npu.npu_moe_gating_top_k

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

- API function: In MoE computation, applies Sigmoid, Softmax, or SqrtSoftplus to input `x`, sorts the computed results by group, and selects top-k experts based on the group sorting result. Two modes are supported:
  - **TopK mode**: Selects experts by performing TopK sorting on `normValue`.
  - **Hash mode**: Obtains precomputed expert indices from the `tid2eid` mapping table based on `input_ids`, skips the sorting step, and outputs the indices directly.
- Formulas:

  **TopK mode:**

    **Step 1: Normalization**

    Normalize input `x` based on `norm_type`:

    $$
    normOut =
    \begin{cases}
        \text{SoftMax}(x),      & norm\_type = 0 \\
        \text{Sigmoid}(x),      & norm\_type = 1\ or\ 3 \\
        \sqrt{\text{Softplus}(x)},     & norm\_type = 2
    \end{cases}
    $$

    **Step 2: Add bias**

    If `bias` is not empty, add the bias to obtain the values used for selection:

    $$
    normValue = normOut + bias
    $$

    Otherwise, $normValue = normOut$.

    **Step 3: Group filtering** (performed only when `group_count > 1`)

    Divide `normValue` into groups by `group_count`, and calculate the score of each group based on `group_select_mode`:

    $$
    groupedValue = Reshape(normValue,\ [batch,\ group\_count,\ -1])
    $$

    $$
    groupScore = \begin{cases}
        ReduceMax(groupedValue,\ dim=-1), & group\_select\_mode = 0 \\
        ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & group\_select\_mode = 1
    \end{cases}
    $$

    Select the `k_group` groups with the highest scores, and set the corresponding positions of unselected groups to $-\infty$:

    $$
    groupIdx = TopK(groupScore,\ k=k\_group).indices
    $$

    $$
    normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
    $$

    **Step 4: Top-K expert selection**

    Perform Top-K on `normValue` to obtain expert indices. Only `expertIdxOut` is needed here:

    $$
    y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
    $$

    **Step 5: Renorm and scaling**

    Obtain the scores of the corresponding k experts from `normOut` based on `expertIdxOut`:

    $$
    gathered = normOut[\text{expertIdxOut}]
    $$

    When `norm_type` is `1` or `2`, normalization is always performed. When `norm_type` is `0` or `3`, `renorm` controls whether normalization is performed:

    $$
    if\ (norm\_type = 1\ or\ norm\_type = 2)\ or\ ((norm\_type = 0\ or\ norm\_type = 3)\ and\ renorm = 1):
    $$

    $$
    \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
    $$

    Otherwise, $yOut = gathered$.

    Final output:

    $$
    yOut = yOut \times routed\_scaling\_factor
    $$

    **Step 6: Optional output**

    If `out_flag` is `True`, the third output is `normOut`; otherwise, it is empty.

  **Hash mode:**

    Hash mode is enabled when both `input_ids` and `tid2eid` are provided.

    **Step 1: Normalization**

    Normalize input `x` based on `norm_type` (the same as in TopK mode):

    $$
    normOut =
    \begin{cases}
        \text{SoftMax}(x),      & norm\_type = 0 \\
        \text{Sigmoid}(x),      & norm\_type = 1\ or\ 3 \\
        \sqrt{\text{Softplus}(x)},     & norm\_type = 2
    \end{cases}
    $$

    **Step 2: Hash index lookup**

    Obtain expert indices from the `tid2eid` mapping table based on `input_ids`:

    $$
    expertIdxOut = tid2eid[input\_ids, :]
    $$

    The shape of `tid2eid` is `[numKeys, k]`, and the shape of `input_ids` is `[batch]` (equal to the first dimension of `x`). Each value in `input_ids` corresponds to the k expert indices in one row of `tid2eid`.

    **Step 3: Gather and scaling**

    Obtain the scores of the corresponding k experts from `normOut` based on `expertIdxOut`:

    $$
    gathered = normOut[\text{expertIdxOut}]
    $$

    When `norm_type` is `1` or `2`, normalization is always performed. When `norm_type` is `0` or `3`, `renorm` controls whether normalization is performed:

    $$
    if\ (norm\_type = 1\ or\ norm\_type = 2)\ or\ ((norm\_type = 0\ or\ norm\_type = 3)\ and\ renorm = 1):
    $$

    $$
    \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
    $$

    Otherwise, $yOut = gathered$.

    Final output:

    $$
    yOut = yOut \times routed\_scaling\_factor
    $$

- Equivalent computation logic:

    ```python
    import torch
    import numpy

    def moe_gating_top_k_numpy(x: torch.tensor, k: int, *, bias: torch.tensor = None, input_ids: torch.tensor = None,
                                tid2eid: torch.tensor = None, k_group: int = 1, group_count: int = 1,
                                group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, out_flag: bool = False,
                                routed_scaling_factor: float = 1.0, eps: float = 1e-20) -> tuple:
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            if bias is not None:
                bias = bias.to(dtype=torch.float32)

        x = x.numpy()
        bias = bias.numpy() if bias is not None else None
        if norm_type == 0:
            x = x - numpy.max(x, axis=-1, keepdims=True)
            x = numpy.exp(x) / numpy.sum(numpy.exp(x), axis=-1, keepdims=True)  # softmax
        elif norm_type in (1, 3):
            x = 1 / (1 + numpy.exp(-x))  # sigmoid
        else:
            x = numpy.sqrt(numpy.maximum(x, 0) + numpy.log1p(numpy.exp(-numpy.abs(x))))  # sqrt softplus
        original_x = x
        if bias is not None:
            x = x + bias

        hash_flag = input_ids is not None and tid2eid is not None
        if hash_flag:
            # Hash mode: Obtain expert indices from the tid2eid mapping table based on input_ids.
            indices = tid2eid.numpy()[input_ids.numpy()].astype(numpy.int64)
        else:
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

        if norm_type in (1, 2) or (norm_type in (0, 3) and renorm != 0):
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

    # Hash mode
    k = 2
    norm_type = 1
    input_ids = numpy.random.randint(0, 16, (8,)).astype(numpy.int64)
    tid2eid = numpy.random.randint(0, 256, (16, 2)).astype(numpy.int64)

    y, expert_idx, out = moe_gating_top_k_numpy(x_tensor, k, input_ids = torch.tensor(input_ids),
                                tid2eid = torch.tensor(tid2eid), norm_type = norm_type,
                                routed_scaling_factor = routed_scaling_factor, eps = eps)

    print(f"yOut shape (hash mode): {y.shape}")
    print(f"Selected experts (hash mode): {expert_idx[0]}")
    ```

## Prototype

```python
npu_moe_gating_top_k(x, k, *, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input to be processed. It must be a 2D Tensor. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout must be ND. Non-contiguous tensors are supported. The size of the last dimension (the expert count) must be less than or equal to `2048`.

- **`k`** (`int`): Required. Number of experts finally selected for each token. The data type is `int64`. The condition `1 <= k <= x_shape[-1] / group_count * k_group` must be satisfied.

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
 
- **`bias`** (`Tensor`): Optional. Bias used together with the input `x` in the computation. It must be a 1D Tensor whose shape matches the last dimension of `x`. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`, and must be the same as that of `x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`input_ids`** (`Tensor`): Optional. Input indices in Hash mode, used to look up expert indices from `tid2eid`. It must be a 1D Tensor whose shape matches the first dimension of `x`. The data type can be `torch.int32` or `torch.int64`. The data layout must be ND. Non-contiguous tensors are supported.

- **`tid2eid`** (`Tensor`): Optional. Hash mapping table that stores precomputed expert indices. It must be a 2D Tensor, and its last dimension must be equal to `k`. The data type can be `torch.int32` or `torch.int64`. The data layout must be ND. Non-contiguous tensors are supported.

- **`k_group`** (`int`): Optional. Number of expert groups selected during group filtering for each token. The data type is `int64`, and the default value is `1`. The conditions `1 <= k_group <= group_count` and `k_group * x_shape[-1] / group_count >= k` must be satisfied.

- **`group_count`** (`int`): Optional. Number of groups into which all experts are divided. The data type is `int64`, and the default value is `1`. `group_count` must be greater than `0`. `x_shape[-1]` must be divisible by `group_count`, and the division result must be greater than `group_select_mode`. After aligning the division result to 32 elements, the product of the aligned result and `group_count` must be less than or equal to `2048`.

- **`group_select_mode`** (`int`): Optional. Method used to calculate the total score of an expert group. The default value is `0`. `0` indicates that the maximum value in the group is used as the group score. `1` indicates that the scores of the top two experts in the group are summed as the group score.

- **`renorm`** (`int`): Optional. Renormalization flag. The default value is `0`. The values `0` and `1` are supported. `0` indicates that renorm is not performed, and `1` indicates that renorm is performed. This parameter takes effect only when `norm_type` is `0` or `3`. When `norm_type` is `1` or `2`, this parameter is invalid, and renormalization is always performed.
- **`norm_type`** (`int`): Optional. Type of the normalization function. The default value is `0`.

    <!-- npu="A3,910b" id4 -->
    - Atlas A2 products and Atlas A3 products: The supported values are `0`, `1`, and `2`. `0` indicates that the Softmax function is used, `1` indicates that the Sigmoid function and renorm are used, and `2` indicates that the SqrtSoftplus function and renorm are used. When the value is `2`, `group_count` must be `1`.
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - Ascend 950PR&950DT products: The supported values are `0`, `1`, `2`, and `3`. `0` indicates that the Softmax function is used, `1` indicates that the Sigmoid function and renorm are used, `2` indicates that the SqrtSoftplus function and renorm are used, and `3` indicates that the Sigmoid function is used and `renorm` controls whether normalization is performed. When the value is `2`, there is no restriction on `group_count`.
    <!-- end id5 -->

- **`out_flag`** (`bool`): Optional. Specifies whether to output the intermediate result of the normalization function. The default value is `False`.
- **`routed_scaling_factor`** (`float`): Optional. `routed_scaling_factor` coefficient used to compute `yOut`. The default value is `1.0`.
- **`eps`** (`float`): Optional. `eps` coefficient used to compute `yOut`. The default value is `1e-20`.

## Return Values

- **`yOut`** (`Tensor`): Result obtained after normalization, group sorting, and Top-K selection on `x`. It must be a 2D Tensor. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`, and must be the same as that of `x`. The data layout must be ND. The first dimension must match that of `x`, and the last dimension must match `k`. Non-contiguous tensors are not supported.
- **`expertIdxOut`** (`Tensor`): Indices obtained after normalization, group sorting, and Top-K selection on `x`, that is, expert IDs. The shape must be the same as that of `yOut`. The data type is `torch.int32`, and the data layout must be ND. Non-contiguous tensors are not supported.
- **`normOut`** (`Tensor`): Output of the normalization computation. Its shape must be the same as that of `x`. The data type is `torch.float32`, and the data layout must be ND. Non-contiguous tensors are not supported.

## Constraints

- This API can be used only in inference scenarios.
- This API supports single-operator mode and graph mode calls.

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

- Hash mode single-operator call

    ```python
    import torch
    import torch_npu
    import numpy

    k = 2
    norm_type = 1
    routed_scaling_factor = 1.0
    eps = 1e-20

    # Generate random data and send it to the NPU.
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    # The shape of input_ids is [batch], with values in [0, numKeys-1]. The shape of tid2eid is [numKeys, k].
    input_ids = numpy.random.randint(0, 64, (16,)).astype(numpy.int64)
    tid2eid = numpy.random.randint(0, 256, (64, 2)).astype(numpy.int64)
    input_ids_tensor = torch.tensor(input_ids).npu()
    tid2eid_tensor = torch.tensor(tid2eid).npu()

    # Call the MoeGatingTopK operator (Hash mode is used when both input_ids and tid2eid are passed).
    y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, input_ids=input_ids_tensor, tid2eid=tid2eid_tensor, norm_type=norm_type, routed_scaling_factor=routed_scaling_factor, eps=eps)
    ```

- Graph mode call

    ```python
    # Graph capture method
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
    # Obtain the default NPU backend provided by TorchAir.
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to compile the model through the compile API.
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
    

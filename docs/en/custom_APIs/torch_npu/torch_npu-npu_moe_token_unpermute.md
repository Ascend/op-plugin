# torch_npu.npu_moe_token_unpermute

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

- API function: Performs the unpermute computation for MoE. It obtains the input data stored in `permuted_tokens` according to the indices stored in `sorted_indices`. If `probs` data exists, `permuted_tokens` is multiplied by `probs`. Finally, the results are accumulated and output.

- Formulas:
  - When `probs` is not None, the formulas are as follows, and `topK_num` is the size of the last dimension of `probs`:

    $$
    T[k] = T[S[k]]
    $$

    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$

    Here, $i \in {0,1,...,tokens-1}$; $j = k - i * topK$, $j \in {0,1,...,topK-1}$; $k \in {0,1,...,tokens*topK-1}$; T indicates permuted_tokens; S indicates sorted_indices; P indicates probs; O indicates out; topK indicates topK_num; and tokens indicates tokens_num.

  - When `probs` is None, `topK_num=1`, and the formulas are as follows:

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    Here, $i \in {0,1,...,tokens-1}$; T indicates permuted_tokens; S indicates sorted_indices; O indicates out; and tokens indicates tokens_num.

> [!NOTE]
>
> - This API uses deterministic computation.
> <!-- npu="950" id5 -->
> - When this API is called on <term>Ascend 950PR&950DT products</term>, the framework internally calls the aclnnMoeFinalizeRoutingV2 API. The parameter mapping is as follows: `permuted_tokens` is equivalent to the `expandedX` input, `sorted_indices` is equivalent to the `expandedRowIdx` input, `probs` is equivalent to the `scalesOptional` input, `padded_mode` is equivalent to the `dropPadMode` input, and the output `out` is equivalent to the `out` output of aclnnMoeFinalizeRoutingV2. If a parameter error occurs, refer to this mapping.
> <!-- end id5 -->

## Prototype

```python
torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None) -> Tensor
```

## Parameters

- **`permuted_tokens`** (`Tensor`): Required. Input tokens after the permute computation. A 2D Tensor with shape (tokens_num * topK_num, hidden_size), where tokens_num is the number of input tokens, topK_num is the number of experts processing each token, and hidden_size is the length of the vector representation of each token. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`sorted_indices`** (`Tensor`): Required. Positions of the data to be computed in `permuted_tokens`. A 1D Tensor with shape (tokens\_num \* topK\_num). The data type can be `torch.int32`. The data layout can be ND. Non-contiguous Tensors are supported. The value range is [0, tokens\_num * topK\_num - 1], and indices are not repeated.
- **`probs`** (`Tensor`): Optional. Probability values multiplied by `permuted_tokens`. The default value is `None`. A 2D Tensor with shape (tokens_num, topK_num). The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported. When `probs` is passed, topK_num is the size of the last dimension of `probs`. When `probs` is not passed, topK_num is 1.
- **`padded_mode`** (`bool`): Optional. Whether to enable paddedMode. The default value is `False`. When padded_mode is `True`, `restore_shape` takes effect, and the shape of the output result is the same as `restore_shape`. Currently, only `False` is supported.
- **`restore_shape`** (`int[]`): Optional. Shape of the output result when padded_mode is `True`. The default value is `None`. Currently, only `None` is supported.

## Return Values

`Tensor`

Output result after the unpermute computation. A 2D Tensor. When padded_mode is `False`, the shape is (tokens_num, hidden_size). When padded_mode is `True`, the shape is the same as `restore_shape` (currently padded_mode supports only `False`). The data type is the same as that of `permuted_tokens`. The data layout can be ND. Non-contiguous Tensors are not supported.

## Constraints

- This API supports inference and training scenarios.
- This API supports graph mode only.
- This API uses deterministic computation.

<!-- npu="A3,910b" id4 -->
- For <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, topK_num must be less than or equal to 512.
<!-- end id4 -->

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> permuted_tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1], [2, 2], [3, 3], [4, 4]],
    ...                                dtype=torch.float16, device='npu')
    >>> sorted_indices = torch.tensor([0, 4, 5, 1, 2, 6, 7, 3], dtype=torch.int32, device='npu')
    >>> probs = torch.ones((4, 2), dtype=torch.float16, device='npu') / 2
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
    >>> unpermuted_tokens
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    ```

    Calling with probs=None (topK_num is 1 in this case):

    ```python
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute(
    ...     permuted_tokens[:4], torch.tensor([0, 1, 2, 3], dtype=torch.int32, device='npu'))
    >>> unpermuted_tokens
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, permuted_tokens, sorted_indices, probs=None):
            return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
    # Instantiate the model.
    model = Model().npu()
    # Obtain the default NPU backend from TorchAir.
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to compile the model.
    model = torch.compile(model, backend=npu_backend)

    # Generate random data and send it to the NPU.
    permuted_tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1], [2, 2], [3, 3], [4, 4]],
                                   dtype=torch.float16).npu()
    sorted_indices = torch.tensor([0, 4, 5, 1, 2, 6, 7, 3], dtype=torch.int32).npu()
    probs = torch.ones((4, 2), dtype=torch.float16).npu() / 2

    # Call the MoeTokenUnpermute operator.
    unpermuted_tokens = model(permuted_tokens, sorted_indices, probs=probs)
    ```

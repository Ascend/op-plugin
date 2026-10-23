# torch_npu.npu_moe_token_permute

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

- API function: MoE (Mixture of Experts) permute computation. It broadcasts and sorts tokens according to the indices.

- Formulas: When `padded_mode` is `False` (currently only `False` is supported), the formulas are as follows, where `topK` is the number of experts selected by one token. When `indices` has 2 dimensions, `topK` is the size of the last dimension of `indices`. When `indices` has 1 dimension, `topK` is 1:

  $$
  sortedIndicesFirst=argSort(flatten(Indices))
  $$

  $$
  sortedIndicesOut=argSort(sortedIndicesFirst)
  $$

  $$
  permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

  When `padded_mode` is `True` (currently not supported):

  $$
  permuteTokensOut[i]=tokens[indices[i]]
  $$

  $$
  sortedIndicesOut=indices
  $$

> [!NOTE]
>
> - This API uses deterministic computation.
> - This API is usually used together with `torch_npu.npu_moe_token_unpermute`: tokens are first permuted and sorted by expert, and after expert computation, the data is restored to the original order through unpermute.

## Prototype

```python
torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False) -> (Tensor, Tensor)
```

## Parameters

- **`tokens`** (`Tensor`): Required. Input token features. A 2D Tensor with shape (num_tokens, hidden_size), where the size of the first dimension is num_tokens. Empty Tensors are supported. The data layout can be ND. Non-contiguous Tensors are supported.

  <!-- npu="950" id4 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.float32`, or `torch.int8`. `torch.int8` is processed in a non-quantized manner.
  <!-- end id4 -->
  <!-- npu="A3" id5 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id5 -->
  <!-- npu="910b" id6 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id6 -->

- **`indices`** (`Tensor`): Required. Input indices. A 1D or 2D Tensor. Empty Tensors are supported. The data type can be `torch.int32` or `torch.int64`. The data layout can be ND. Non-contiguous Tensors are supported.
  - When `padded_mode` is `False`, it indicates the topK processing expert indices corresponding to each input token. When the number of dimensions is 2, the shape is (num_tokens, topK). When the number of dimensions is 1, the shape is (num_tokens), and topK is considered to be 1.
  - When `padded_mode` is `True`, it indicates the token indices selected by each expert (currently not supported).
  - The number of elements must be less than 16777215, and element values must be greater than or equal to 0 and less than 16777215.

  <!-- npu="950" id7 -->
  - When this API is called on <term>Ascend 950PR&950DT products</term> and the data type of `tokens` is `torch.int8`, the elements represent expert IDs. The value range is [0, 10240), the maximum value is 10239, and 10240 is not supported.
  <!-- end id7 -->

- **`num_out_tokens`** (`int`): Optional. Number of valid output tokens. The default value is `None`, and the data type is `int64`.
  - When the value is `None` or `0`, no token is deleted.
  - When the value is greater than `0`, the tokens sorted by expert are sliced according to num_out_tokens, and the first num_out_tokens tokens are retained.
  - When the value is less than `0`, it is processed as a negative slice index.

- **`padded_mode`** (`bool`): Optional. Whether to use padded mode. The default value is `False`.
  - `False`: Non-padded mode. The indices are sorted.
  - `True`: Padded mode. The indices have been padded to represent the token indices selected by each expert, and the indices are not sorted (currently not supported).

## Return Values

- **`permuted_tokens`** (`Tensor`): Tokens expanded and sorted according to the indices. A 2D Tensor. The data type is the same as that of `tokens`. The data layout can be ND. Non-contiguous Tensors are not supported. The size of the first dimension is min(num_tokens \* topK, num_out_tokens). When `num_out_tokens` is `None` or `0`, the size of the first dimension is num_tokens \* topK. When `num_out_tokens` is less than `0`, it is processed as a negative slice index. The sizes of the other dimensions are the same as those of `tokens`.
- **`sorted_indices`** (`Tensor`): Mapping between `permuted_tokens` and `tokens`. A 1D Tensor with shape (num_tokens * topK), that is, the number of elements in `indices`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous Tensors are not supported. sorted_indices[i] indicates the position of row i // topK of `tokens` in `permuted_tokens`, that is, permuted_tokens[sorted_indices[i]] = tokens[i // topK]. It can be used with `torch_npu.npu_moe_token_unpermute` to restore the data to the original order.

## Constraints

- This API supports inference and training scenarios.
- This API supports graph mode only.
- This API uses deterministic computation.
- The data types of `tokens` and `permuted_tokens` must be the same.
<!-- npu="950" id8 -->
- `tokens` and `permuted_tokens` of the `torch.int8` type are supported only on <term>Ascend 950PR&950DT products</term>.
<!-- end id8 -->

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float16, device='npu')
    >>> indices = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=torch.int32, device='npu')
    >>> permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices)
    >>> permuted_tokens
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.],
            [1., 1.],
            [2., 2.],
            [3., 3.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    >>> sorted_indices
    tensor([0, 4, 5, 1, 2, 6, 7, 3], device='npu:0', dtype=torch.int32)
    >>> permuted_tokens[sorted_indices]
    tensor([[1., 1.],
            [1., 1.],
            [2., 2.],
            [2., 2.],
            [3., 3.],
            [3., 3.],
            [4., 4.],
            [4., 4.]], device='npu:0', dtype=torch.float16)
    ```

    Specify `num_out_tokens` to slice the sorted tokens:

    ```python
    >>> permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=4)
    >>> permuted_tokens
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
        def forward(self, tokens, indices):
            return torch_npu.npu_moe_token_permute(tokens, indices)
    # Instantiate the model.
    model = Model().npu()
    # Obtain the default NPU backend from TorchAir.
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to compile the model.
    model = torch.compile(model, backend=npu_backend)

    # Generate random data and send it to the NPU.
    tokens = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float16).npu()
    indices = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=torch.int32).npu()

    # Call the MoeTokenPermute operator.
    permuted_tokens, sorted_indices = model(tokens, indices)
    ```

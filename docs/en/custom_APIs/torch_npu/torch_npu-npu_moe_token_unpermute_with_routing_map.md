# torch_npu.npu_moe_token_unpermute_with_routing_map

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

- API function: For permuted_tokens processed by the permute computation, this API accumulates them back to the original unpermuted_tokens according to sorted_indices and routing_map. The drop_and_pad mode is supported.

- Formulas:

  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  Here, `topK_num` is the maximum number of expert slots reserved for each token. When `drop_and_pad` is `False`, the actual number of experts selected by each token can be less than or equal to `topK_num`. Unused slots are represented by `-1` in `sorted_indices` and are skipped during computation.

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  (1) When probs is not None and drop_and_pad is `True`:

  $$
  permuteProbs[i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens * permuteProbs
  $$

  $$
  unpermutedTokens = zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (2) When probs is not None and drop_and_pad is `False` (T is the transpose operation):

  $$
  permuteProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens * permuteProbs
  $$

  $$
  unpermutedTokens = zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  if \space sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]] * permuteProbs[i]
  $$

  (3) When probs is None and drop_and_pad is `True`:

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (4) When probs is None and drop_and_pad is `False`:

  $$
  if \space sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

> [!NOTE]
>
> - This API uses a non-deterministic implementation by default. Deterministic computation can be enabled through alcrtCtxSetSysParamOpt.

## Prototype

```python
torch_npu.npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape, *, probs=None, routing_map=None, drop_and_pad=False) -> Tensor
```

## Parameters

- **`permuted_tokens`** (`Tensor`): Required. Input tokens after the permute computation. A 2D Tensor: when drop_and_pad is `False`, the shape is (tokens_num \* topK_num, hidden_size); when drop_and_pad is `True`, the shape is (experts_num \* capacity, hidden_size), where capacity is the number of tokens that each expert can process. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`sorted_indices`** (`Tensor`): Required. Mapping between the input and output gradients. A 1D Tensor: when drop_and_pad is `False`, the shape is (tokens_num \* topK_num), the index range is [0, tokens_num \* topK_num - 1], and `-1` can be used to indicate an invalid slot, which is skipped during computation; when drop_and_pad is `True`, the shape is (experts_num \* capacity), and the index range is [0, tokens_num - 1]. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`restore_shape`** (`int[]`): Required. Shape of the output unpermuted_tokens. The size is 2, that is, (tokens_num, hidden_size).

- **`*`**: Syntax separator used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.

- **`probs`** (`Tensor`): Optional. Weight of the result obtained after the token at the corresponding position is processed by the corresponding expert in the final result. The default value is `None`. A 2D Tensor with the same shape as routing_map, that is, (tokens_num, experts_num). The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`: the data type is the same as that of permuted_tokens, and when the data type of permuted_tokens is `torch.bfloat16`, probs also supports `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported. When probs is `None`, routing_map does not need to be passed.
- **`routing_map`** (`Tensor`): Optional. Whether the token at the corresponding position is processed by the corresponding expert. The default value is `None`. A 2D Tensor with shape (tokens_num, experts_num). The data type can be `torch.int8` or `torch.bool`: when the data type is `torch.int8`, the values can be 0 or 1; when the data type is `torch.bool`, the values can be true or false. The data layout can be ND. Non-contiguous Tensors are supported. When probs is `None`, this parameter does not need to be passed.
- **`drop_and_pad`** (`bool`): Optional. Whether to enable padded mode. The default value is `False`.
  - `False`: Padded mode is disabled.
  - `True`: Padded mode is enabled.

## Return Values

 `Tensor`

 Forward output result after the unpermute computation. A 2D Tensor with shape restore_shape, that is, (tokens_num, hidden_size). The data type is the same as that of permuted_tokens. The data layout can be ND.

## Constraints

- This API supports inference and training scenarios.
- This API supports graph mode only.
- This API uses a non-deterministic implementation by default. Deterministic computation can be enabled through alcrtCtxSetSysParamOpt.
- topK_num must be less than or equal to 512. When drop_and_pad is `False`, a maximum of topK_num expert slots are reserved for each token, the number of 1 or true values in each row of routing_map must be less than or equal to topK_num, and `-1` can be used in sorted_indices to indicate an invalid slot.
- The following scenarios will be intercepted in later versions. If a warning is displayed, rectification is recommended:
  - drop_and_pad is `True` and topK_num is greater than experts_num.
  - drop_and_pad is `True` and capacity is greater than tokens_num.
  - The data type or shape of routing_map does not meet the requirements.
  - The data layout of the input Tensor is not ND.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> permuted_tokens = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float16, device='npu')
    >>> sorted_indices = torch.tensor([0, 1], dtype=torch.int32, device='npu')
    >>> routing_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8, device='npu')
    >>> probs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float16, device='npu')
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute_with_routing_map(
    ...     permuted_tokens, sorted_indices, [2, 4], probs=probs, routing_map=routing_map)
    >>> unpermuted_tokens
    tensor([[1., 1., 1., 1.],
            [2., 2., 2., 2.]], device='npu:0', dtype=torch.float16)
    ```

    drop_and_pad mode:

    ```python
    >>> permuted_tokens = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float16, device='npu')
    >>> sorted_indices = torch.tensor([0, 1], dtype=torch.int32, device='npu')
    >>> probs = torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.float16, device='npu')
    >>> unpermuted_tokens = torch_npu.npu_moe_token_unpermute_with_routing_map(
    ...     permuted_tokens, sorted_indices, [2, 4], probs=probs, routing_map=routing_map, drop_and_pad=True)
    >>> unpermuted_tokens
    tensor([[0.5000, 0.5000, 0.5000, 0.5000],
            [1.0000, 1.0000, 1.0000, 1.0000]], device='npu:0', dtype=torch.float16)
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
        def forward(self, permuted_tokens, sorted_indices, restore_shape, probs=None, routing_map=None, drop_and_pad=False):
            return torch_npu.npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape,
                                                                      probs=probs, routing_map=routing_map,
                                                                      drop_and_pad=drop_and_pad)
    # Instantiate the model.
    model = Model().npu()
    # Obtain the default NPU backend from TorchAir.
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to compile the model.
    model = torch.compile(model, backend=npu_backend)

    # Generate random data and send it to the NPU.
    permuted_tokens = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float16).npu()
    sorted_indices = torch.tensor([0, 1], dtype=torch.int32).npu()
    routing_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8).npu()
    probs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float16).npu()

    # Call the MoeTokenUnpermuteWithRoutingMap operator.
    unpermuted_tokens = model(permuted_tokens, sorted_indices, [2, 4], probs=probs, routing_map=routing_map)
    ```

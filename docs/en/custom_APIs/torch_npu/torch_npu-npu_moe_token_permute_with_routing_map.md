# torch_npu.npu_moe_token_permute_with_routing_map

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

- API function: MoE permute computation. The token and expert labels are passed as routing_map, and tokens and optional probs are broadcast and sorted according to routing_map.

- Formulas: tokens_num is the size of dimension 0 of routing_map, and expert_num is the size of dimension 1 of routing_map.

  When drop_and_pad is `False`:

  $$
  expertIndex=arange(tokens\_num).expand(expert\_num,-1)
  $$

  $$
  sortedIndicesFirst=expertIndex.masked\_select(routingMap.T)
  $$

  $$
  sortedIndicesOut=argsort(sortedIndicesFirst)
  $$

  $$
  topK = numOutTokens // tokens\_num
  $$

  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

  If probs is not None:

  $$
  permuteProbsOutOptional=probsOptional.T.masked\_select(routingMap.T)
  $$

  When drop_and_pad is `True`:

  $$
  capacity = numOutTokens // expert\_num
  $$

  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort[routingMap.T,dim=-1](:, :capacity)
  $$

  $$
  permutedTokensOut = tokens.index\_select(0, sortedIndicesOut)
  $$

  If probs is not None:

  $$
  probs\_T\_1D = probsOptional.T.view(-1)
  $$

  $$
  indices\_dim0 = arange(expert\_num).view(expert\_num, 1)
  $$

  $$
  indices\_dim1 = sortedIndicesOut.view(expert\_num, capacity)
  $$

  $$
  indices\_1D = (indices\_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$

  $$
  permuteProbsOutOptional = probs\_T\_1D.index\_select(0, indices\_1D)
  $$

> [!NOTE]
>
> - This API uses deterministic computation.
> - In non-drop_and_pad mode, the number of 1 or true values in each row of routing_map must be fixed (recorded as topK) and must be consistent with num_out_tokens // tokens_num.

## Prototype

```python
torch_npu.npu_moe_token_permute_with_routing_map(tokens, routing_map, *, probs=None, num_out_tokens=None, drop_and_pad=False) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`tokens`** (`Tensor`): Required. Input token features. A 2D Tensor with shape (tokens_num, hidden_size). Empty Tensors are supported. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`routing_map`** (`Tensor`): Required. Mapping between tokens and experts. A 2D Tensor with shape (tokens_num, experts_num). Empty Tensors are supported. The data type can be `torch.int8` or `torch.bool`: when the data type is `torch.int8`, the values can be 0 or 1; when the data type is `torch.bool`, the values can be true or false. The data layout can be ND. Non-contiguous Tensors are supported.
  - In non-drop_and_pad mode, the number of 1 or true values in each row must be fixed. This number is recorded as topK.

- **`*`**: Syntax separator used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.

- **`probs`** (`Tensor`): Optional. Probability values corresponding to routing_map. The default value is `None`. Empty Tensors are supported. The number of elements must be the same as that of routing_map (shape (tokens_num, experts_num)). The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`: only when the data type of probs is `torch.float32` and the data type of tokens is `torch.bfloat16` can the data type of probs differ from that of tokens; in other scenarios, the data type of probs must be the same as that of tokens. The data layout can be ND. Non-contiguous Tensors are supported. When probs is `None`, the output permuted_probs is empty.

- **`num_out_tokens`** (`int`): Optional. Number of valid output tokens, used to calculate topK and capacity. The default value is `None`, which is equivalent to tokens_num. The value range is [0, tokens_num * experts_num].
  - In non-drop_and_pad mode, topK = num_out_tokens // tokens_num, and the actual number of output tokens outToken = topK * tokens_num.
  - In drop_and_pad mode, capacity = num_out_tokens // experts_num, and the actual number of output tokens outToken = capacity * experts_num.

- **`drop_and_pad`** (`bool`): Optional. Whether to enable dropAndPad mode. The default value is `False`.
  - `False`: Non-dropAndPad mode.
  - `True`: dropAndPad mode. Each expert selects tokens according to a fixed capacity, and the insufficient part is padded.

## Return Values

- **`permuted_tokens`** (`Tensor`): Tokens expanded, sorted, and filtered according to routing_map. A 2D Tensor with shape (outToken, hidden_size). The data type is the same as that of tokens. The data layout can be ND.
- **`permuted_probs`** (`Tensor`): Probs sorted and filtered according to routing_map. The shape is (outToken), and the data type is the same as that of probs. When probs is `None`, this output is empty.
- **`sorted_indices`** (`Tensor`): Mapping between permuted_tokens and tokens. A 1D Tensor with shape (outToken). The data type can be `torch.int32`. The data layout can be ND. In non-drop_and_pad mode, sorted_indices[i] indicates the position of row i // topK of tokens in permuted_tokens, that is, permuted_tokens[sorted_indices[i]] = tokens[i // topK].

## Constraints

- This API supports inference and training scenarios.
- This API supports graph mode only.
- This API uses deterministic computation.

- Because lossless float-to-int conversion is required, tokens_num and experts_num must be less than 16777215.
- When drop_and_pad is `False`, the number of 1 or true values in each row of routing_map must be fixed and less than 512, topK (num_out_tokens // tokens_num) must be less than 512, and topK * tokens_num must be less than 16777215.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> tokens = torch.tensor([[0.1, 0.1, 0.1, 0.1],
    ...                        [0.2, 0.2, 0.2, 0.2],
    ...                        [0.3, 0.3, 0.3, 0.3]], dtype=torch.float16, device='npu')
    >>> routing_map = torch.ones((3, 2), dtype=torch.int8, device='npu')
    >>> permuted_tokens, _, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(tokens, routing_map, num_out_tokens=6)
    >>> permuted_tokens
    tensor([[0.1000, 0.1000, 0.1000, 0.1000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.3000, 0.3000, 0.3000, 0.3000],
            [0.1000, 0.1000, 0.1000, 0.1000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.3000, 0.3000, 0.3000, 0.3000]], device='npu:0', dtype=torch.float16)
    >>> sorted_indices
    tensor([0, 3, 1, 4, 2, 5], device='npu:0', dtype=torch.int32)
    >>> permuted_tokens[sorted_indices]
    tensor([[0.1000, 0.1000, 0.1000, 0.1000],
            [0.1000, 0.1000, 0.1000, 0.1000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000, 0.2000],
            [0.3000, 0.3000, 0.3000, 0.3000],
            [0.3000, 0.3000, 0.3000, 0.3000]], device='npu:0', dtype=torch.float16)
    ```

    Specify probs for the permute computation:

    ```python
    >>> probs = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float16, device='npu')
    >>> permuted_tokens, permuted_probs, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
    ...     tokens, routing_map, probs=probs, num_out_tokens=6)
    >>> permuted_probs
    tensor([0.5000, 0.4000, 0.2000, 0.3000, 0.6000, 0.8000], device='npu:0',
           dtype=torch.float16)
    ```

    drop_and_pad mode call

    ```python
    >>> permuted_tokens, _, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
    ...     tokens, routing_map, num_out_tokens=4, drop_and_pad=True)
    >>> permuted_tokens.shape, sorted_indices.shape
    (torch.Size([4, 4]), torch.Size([4]))
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
        def forward(self, tokens, routing_map, probs=None, num_out_tokens=None, drop_and_pad=False):
            return torch_npu.npu_moe_token_permute_with_routing_map(tokens, routing_map, probs=probs,
                                                                    num_out_tokens=num_out_tokens,
                                                                    drop_and_pad=drop_and_pad)
    # Instantiate the model.
    model = Model().npu()
    # Obtain the default NPU backend from TorchAir.
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to compile the model.
    model = torch.compile(model, backend=npu_backend)

    # Generate random data and send it to the NPU.
    tokens = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
                          dtype=torch.float16).npu()
    routing_map = torch.ones((3, 2), dtype=torch.int8).npu()

    # Call the MoeTokenPermuteWithRoutingMap operator.
    permuted_tokens, _, sorted_indices = model(tokens, routing_map, num_out_tokens=6)
    ```

# torch\_npu.npu\_moe\_re\_routing

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Rearranges tokens by expert order in the Mixture of Experts (MoE) network after AlltoAll communication across ranks.
- Formulas:

    ![](../../figures/en-us_formulaimage_0000002277237821.png)

    - `SrcOffset`: source offset of tokens to be moved, computed from the input `expert_token_num_per_rank`.
    - `DstOffset`: destination offset of tokens to be moved.
    - `cur_rank`: row index of `expert_token_num_per_rank`, representing the source rank of tokens.
    - `cur_expert`: column index of `expert_token_num_per_rank`, representing the expert on the rank that processes the tokens.

## Prototype

```python
torch_npu.npu_moe_re_routing(tokens, expert_token_num_per_rank, *, per_token_scales=None, expert_token_num_type=1, idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

> [!NOTE]  
> Variables used in tensor shapes:
>
> - `A`: indicates the number of tokens. The value is calculated as `Sum(expert_token_num_per_rank)`.
> - `H`: indicates the token length. The value range is: 0< `H` <16384.
> - `N`: indicates the number of ranks. The value is not limited.
> - `E`: indicates the number of experts on a rank. The value is not limited.

- **`tokens`** (`Tensor`): Required. Tokens to be rearranged. This parameter must be 2D with shape `(A, H)`. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND.
- **`expert_token_num_per_rank`** (`Tensor`): Required. Two-dimensional matrix, where `[i, j]` represents the token count received from rank `i` that is processed by expert `j` on the current rank. This parameter must be 2D with shape `(N, E)`. The data type can be `int32` or `int64`. The data layout must be ND. All values must be greater than 0.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`per_token_scales`** (`Tensor`): Optional. Scale corresponding to each token, which must be rearranged along with the tokens. This parameter must be 1D with shape `(A,)`. The data type can be `float32`. The data layout must be ND.
- **`expert_token_num_type`** (`int`): Optional. Output mode of `expert_token_num`. The value `0` enables cumsum mode, and the value `1` (default)enables count mode. Currently, only the value `1` is supported.
- **`idx_type`** (`int`): Optional. Index type of the output `permute_token_idx`. The value `0` (default) enables gather indices, and the value `1` enables scatter indices. Currently, only the value `0` is supported.

## Return Values

- **`permute_tokens`** (`Tensor`): Rearranged tokens. This parameter must be 2D with shape `(A, H)`. The data type can be `float16`, `bfloat16`, or `int8`. The data layout must be ND.
- **`permute_per_token_scales`** (`Tensor`): Rearranged `per_token_scales`. In scenarios where `per_token_scales` is not provided as input, this output is invalid. This parameter must be 1D with shape `(A,)`. The data type can be `float32`. The data layout must be ND.
- **`permute_token_idx`** (`Tensor`): Index of each token in the original layout. This parameter must be 1D with shape `(A,)`. The data type is `int32`. The data layout must be ND.
- **`expert_token_num`** (`Tensor`): Token count processed by each expert. This parameter must be 1D with shape `(E,)`. The data type can be `int32` or `int64`. The data layout must be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import random
    import copy
    import math
    
    tokens_num = 16384
    tokens_length = 7168
    rank_num = 16
    expert_num = 16
    tokens = torch.randint(low=-10, high = 20, size=(tokens_num, tokens_length), dtype=torch.int8)
    expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int32)
    tokens_sum = 0
    for i in range(rank_num):
        for j in range(expert_num):
            if i == rank_num - 1 and j == expert_num - 1:
                expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                break
            if tokens_num >= rank_num * expert_num :
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
            elif tokens_sum >= tokens_num:
                rand_num = 0
            else:
                rand_int = tokens_num - tokens_sum
                rand_num = random.randint(0, rand_int)
            rand_num = 1
            expert_token_num_per_rank[i][j] = rand_num
            tokens_sum += rand_num
    per_token_scales = torch.randn(tokens_num, dtype = torch.float32)
    expert_token_num_type = 1
    idx_type = 0
    tokens_npu = copy.deepcopy(tokens).npu()
    per_token_scales_npu = copy.deepcopy(per_token_scales).npu()
    expert_token_num_per_rank_npu = copy.deepcopy(expert_token_num_per_rank).npu()
    permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = torch_npu.npu_moe_re_routing(tokens_npu, expert_token_num_per_rank_npu, per_token_scales=per_token_scales_npu, expert_token_num_type=expert_token_num_type, idx_type=idx_type)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import random
    import copy
    import math
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    tokens_num = 16384
    tokens_length = 7168
    rank_num = 16
    expert_num = 16
    tokens = torch.randint(low=-10, high = 20, size=(tokens_num, tokens_length), dtype=torch.int8)
    expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype = torch.int32)
    tokens_sum = 0
    for i in range(rank_num):
        for j in range(expert_num):
            if i == rank_num - 1 and j == expert_num - 1:
                expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                break
            if tokens_num >= rank_num * expert_num :
                floor = math.floor(tokens_num / (rank_num * expert_num))
                rand_num = random.randint(1, floor)
            elif tokens_sum >= tokens_num:
                rand_num = 0
            else:
                rand_int = tokens_num - tokens_sum
                rand_num = random.randint(0, rand_int)
            rand_num = 1
            expert_token_num_per_rank[i][j] = rand_num
            tokens_sum += rand_num
    per_token_scales = torch.randn(tokens_num, dtype = torch.float32)
    expert_token_num_type = 1
    idx_type = 0
    tokens_npu = copy.deepcopy(tokens).npu()
    per_token_scales_npu = copy.deepcopy(per_token_scales).npu()
    expert_token_num_per_rank_npu = copy.deepcopy(expert_token_num_per_rank).npu()
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, tokens, expert_token_num_per_rank, per_token_scales=None, expert_token_num_type=1, idx_type=0):
            permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = torch_npu.npu_moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales=per_token_scales, expert_token_num_type=expert_token_num_type, idx_type=idx_type)
            return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num
    
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    permute_tokens_npu, permute_per_token_scales_npu, permute_token_idx_npu, expert_token_num_npu = model(tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu, expert_token_num_type, idx_type)
    ```

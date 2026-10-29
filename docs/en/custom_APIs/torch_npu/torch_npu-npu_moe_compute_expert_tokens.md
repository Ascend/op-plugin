# torch_npu.npu_moe_compute_expert_tokens

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- Uses binary search to locate the position of the last row processed by each expert in Mixture-of-Experts (MOE) computation.
- Formula:

    $$
    expertTokens_i = BinarySearch(sortedExpertForSourceRow,i), \quad i \in [0,numExpert)
    $$

## Prototype

```python
torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert) -> Tensor
```

## Parameters

- **`sorted_expert_for_source_row`** (`Tensor`): Required. Expert index corresponding to each source row after sorting, $\mathrm{sortedExpertForSourceRow}$ in the formula. This parameter must be a 1D tensor. The data type must be `torch.int32`. The data layout must be `ND`. The number of elements must be less than $2^{24}$.

- **`num_expert`** (`int`): Required. Total number of experts, $numExpert$ in the formula. The value range is (0, 2048].

## Return Values

`Tensor`

 $expertTokens$ in the formula. This tensor must be a 1D tensor. The data type must be identical to that of `sorted_expert_for_source_row`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    sorted_experts = torch.tensor([0,0,0,1,1,2], dtype=torch.int32)
    num_experts = 3
    output = torch_npu.npu_moe_compute_expert_tokens(sorted_experts.npu(), num_experts)
    ```

- Graph mode call

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
        
        def forward(self, sorted_experts, num_experts):
            return torch_npu.npu_moe_compute_expert_tokens(sorted_experts, num_experts)
    def main():
        sorted_experts = torch.tensor([0,0,0,1,1,2], dtype=torch.int32)
        num_experts = 3
        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        custom_output = model(sorted_experts, num_experts)
    if __name__ == '__main__':
        main()
    ```

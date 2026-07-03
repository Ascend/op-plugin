# torch_npu.npu_moe_init_routing

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Performs MoE routing based on the computation results of [torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md).
- Formulas:
    $$
    expanded\_expert\_idx, sorted\_row\_idx = keyValueSort(expert\_idx, row\_idx)\\
    expanded\_row\_idx[sorted\_row\_idx[i]] = i\\
    expanded\_x[i] = x[sorted\_row\_idx[i] \% num\_rows]
    $$

- Equivalent computation logic:

    ```python
    import numpy as np

    def cpu_op_exec( x, row_idx, expert_idx, active_num):
        num_rows = x.shape[0]
        hidden_size = x.shape[-1]
        k = expert_idx.shape[-1]
        sort_expert_for_source_row = np.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable")
        expanded_expert_idx = np.sort(
            expert_idx.reshape((-1,)), axis=-1)

        expanded_dst_to_src_row = np.take_along_axis(
            row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1)
        expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
        expanded_row_idx[expanded_dst_to_src_row] = np.arange(
            expanded_dst_to_src_row.shape[-1])
        active_num = min(active_num, num_rows) * k
        expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
        return expanded_x, expanded_row_idx, expanded_expert_idx


    if __name__ == "__main__":
        n = 10
        col = 200
        k = 2
        dtype = np.float32 
        x = np.random.uniform(-1, 1, size=(n, col)).astype(dtype)
        row_idx = np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32)
        expert_idx = np.random.randint(0, 100, size=(n, k)).astype(np.int32)
        active_num = n
        expanded_x, expanded_row_idx, expanded_expert_idx = cpu_op_exec( x, row_idx, expert_idx, active_num)
        print(f"expanded_x:{expanded_x}")
        print(f"expanded_row_idx:{expanded_row_idx}")
        print(f"expanded_expert_idx:{expanded_expert_idx}")
    ``` 
    
## Prototype

```python
torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input token features for MoE. This parameter must be 2D with shape `(NUM_ROWS, H)`. The data type can be `float16`, `bfloat16`, or `float32`. The data layout must be ND. The total shape size must be less than `2^24`.
- **`row_idx`** (`Tensor`): Required. Original row position corresponding to each location. The shape must be identical to the shape of `expert_idx`. The data type is `int32`. The data layout must be ND.
- **`expert_idx`** (`Tensor`): Required. Selected K processing experts corresponding to each row feature in the output of [torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md). This parameter must be 2D with shape `(NUM_ROWS, K)`. The data type can be `int32`. The data layout is ND.
- **`active_num`** (`int`): Required. Maximum number of rows to be processed. The number of rows in `expanded_x` is limited to the value specified by this parameter.

## Return Values

- **`expanded_x`** (`Tensor`): Expanded features generated according to `expert_idx`. This parameter must be 2D with shape `(min(NUM_ROWS, active_num) * K, H)`. The data type must be identical to that of `x`. The data layout must be ND.
- **`expanded_row_idx`** (`Tensor`): Mapping between `expanded_x` and `x`. This parameter must be 1D with shape `(NUM_ROWS * K,)`. The data type is `int32`. The data layout must be ND.
- **`expanded_expert_idx`** (`Tensor`): Sorted result of `expert_idx`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
    row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
    expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
    active_num = 3
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    device=torch.device(f'npu:0')
    
    torch_npu.npu.set_device(device)
    
    class MoeInitRoutingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, row_idx, expert_idx, active_num):
            expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num=active_num)
            return expanded_x, expanded_row_idx, expanded_expert_idx
    
    x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
    row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
    expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
    active_num = 3
    
    moe_init_routing_model = MoeInitRoutingModel().npu()
    moe_init_routing_model = torch.compile(moe_init_routing_model, backend=npu_backend, dynamic=True)
    expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing_model(x, row_idx, expert_idx, active_num=active_num)
    print(expanded_x)
    print(expanded_row_idx)
    print(expanded_expert_idx)
    ```
    
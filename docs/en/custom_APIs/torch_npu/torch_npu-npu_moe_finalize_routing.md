# torch_npu.npu_moe_finalize_routing

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Combines the output results of the MoE feedforward neural network (FFN) at the end of MoE computation.

- Formulas:

    $$
    expertid = expertForSourceRow[i,k] \\
    out(i,j) = skip1_{i,j} + skip2Optional_{i,j} + \sum_{k=0}^{K}(scales_{i,k} * (expandPermutedRows_{expandedSrcToDstRow_{i+k*num\_rows},j} + bias_{expertid,j}))
    $$

- Equivalent computation logic:

    ```python
    import numpy as np
    from copy import deepcopy   
        
    def generate_input_data(expert_num=16, token_len=10, top_k=4, num_rows=50):
        expanded_permuted_rows = np.random.randn(num_rows * top_k, token_len).astype(np.float32)
        skip1 = np.random.randn(num_rows, token_len).astype(np.float32)
        skip2_optional = np.random.randn(num_rows, token_len).astype(np.float32)
        bias = np.random.randn(expert_num, token_len).astype(np.float32)
        scales = np.random.randn(num_rows, top_k).astype(np.float32)
        expanded_src_to_dst_row = np.arange(num_rows * top_k).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row)
        expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(num_rows, top_k)).astype(np.int32)
        return expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row
        
    def generate_input_data_drop_pad(expert_num=16, token_len=10, c=20, top_k=4, num_rows=50):
        expanded_permuted_rows = np.random.randn(expert_num, c, token_len).astype(np.float32)
        skip1 = np.random.randn(num_rows, token_len).astype(np.float32)
        skip2_optional = np.random.randn(num_rows, token_len).astype(np.float32)
        bias = np.random.randn(expert_num, token_len).astype(np.float32)
        scales = None
        expanded_src_to_dst_row = np.arange(num_rows * top_k).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row)
        expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(num_rows, top_k)).astype(np.int32) 
        return expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row
        
            
    def moe_finalize_routing_np(expanded_permuted_rows: np.array,skip1: np.array,skip2_optional: np.array ,bias: np.array ,scales: np.array ,expanded_src_to_dst_row: np.array ,expert_for_source_row: np.array):
        NK = expanded_src_to_dst_row.shape[0]
        K = 1
        if scales is not None:
            K = scales.shape[1] 
        num_rows = NK // K
        H = expanded_permuted_rows.shape[-1]
        expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)
        if (skip1 is not None) and (skip2_optional is not None):
            out = skip1 + skip2_optional
        elif (skip2_optional is not None) and (skip1 is None):
            out = deepcopy(skip2_optional)
        elif (skip2_optional is None) and (skip1 is not None):
            out = deepcopy(skip1)
        else:
            out = np.zeros([num_rows, H])
        for i in range(num_rows):
            for k in range(K):
                value = expanded_src_to_dst_row[k * num_rows + i]
                if value == -1:
                    dst_row = 0
                else:
                    dst_row = expanded_permuted_rows[value, :]
                expert_id = expert_for_source_row[i, k]
                    
                scalesV = 1.0
                if scales is not None:
                    scalesV = scales[i, k]
                if bias is not None:
                    out[i, :] += scalesV * (dst_row + bias[expert_id, :])
                else:
                    out[i, :] += scalesV * dst_row
        return out

    if __name__ == "__main__":
        #test_moe_finalize_routing
        expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row = generate_input_data(expert_num=16, token_len=5, top_k=4, num_rows=5)
        out1 = moe_finalize_routing_np(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row)
        print(f"\nout with moe finalize routing:{out1}")
        #moe_finalize_routing_np_drop_pad
        expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row = generate_input_data_drop_pad(expert_num=16, token_len=5, top_k=1, num_rows=20)
        out2 = moe_finalize_routing_np(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row)
        print(f"\nout with moe finalize routing drop pad:{out2}")  
    ```

## Prototype

```python
torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode=0) -> Tensor
```

## Parameters
>
> [!NOTE]  
> Definitions of variables in the shape:
>
> - $NUM\_ROWS$: number of rows.
> - $K$: number of experts selected from the total of $E$ experts.
> - $H$: hidden layer size, indicating the number of columns.
> - $E$: number of experts. The condition $E \ge K$ must be met.
> - $C$: capacity threshold, indicating the number of tokens an expert can process.

- **`expanded_permuted_rows`** (`Tensor`): Required. Result after expert processing, $expandPermutedRows$ in the formula. This parameter must be a 2D tensor. The data type can be `float16`, `bfloat16`, or `float32`. The data layout must be ND. When `drop_pad_mode` is `0` or `2`, the shape is `(NUM_ROWS * K, H)`. When `drop_pad_mode` is `1` or `3`, the shape is `(E, C, H)`.
- **`skip1`** (`Tensor`): Required. This parameter can be set to `None`. The first input parameter for summation, $skip1$ in the formula. This parameter must be a 2D tensor. The data type must be identical to that of `expanded_permuted_rows`. The shape must be identical to the shape of the output `out`.
- **`skip2`** (`Tensor`): Required. This parameter can be set to `None`. The second input parameter for summation, $skip2$ in the formula. This parameter must be a 2D tensor. The data type must be identical to that of `expanded_permuted_rows`. The shape must be identical to the shape of the output `out`. When `skip1` is `None`, `skip2` must also be `None`.
- **`bias`** (`Tensor`): Required. This parameter can be set to `None`. Expert bias, $bias$ in the formula. This parameter must be a 2D tensor. The data type must be identical to that of `expanded_permuted_rows`. The shape can be `(E, H)`.
- **`scales`** (`Tensor`): Required. This parameter can be set to `None`. Expert weights, $scales$ in the formula. This parameter must be a 2D tensor. The data type must be identical to that of `expanded_permuted_rows`. The shape can be `(NUM_ROWS, K)`.
- **`expanded_src_to_dst_row`** (`Tensor`): Required. Index of each expert processing result, $expandedSrcToDstRow$ in the formula. This parameter must be a 1D tensor. The data type is `int32`. The shape can be `(NUM_ROWS * K,)`. When `drop_pad_mode` is `0` or `2`, the value range of the tensor is `[0, NUM_ROWS * K - 1]`. When `drop_pad_mode` is `1` or `3`, the value range of the tensor is `[-1, E * C - 1]`.
- **`expert_for_source_row`** (`Tensor`): Required. This parameter can be set to `None`. Expert ID for each row, $expertForSourceRow$ in the formula. This parameter must be a 2D tensor. The data type is `int32`. The shape can be `(NUM_ROWS, K)`. The value range of the tensor is `[0, E - 1]`.
- **`drop_pad_mode`** (`int`): Optional. Specifies whether to enable drop-pad mode and the layout of `expanded_src_to_dst_row`. Valid values are `0`, `1`, `2`, or `3`. The default value is `0`.
     - `0`: enables dropless mode and `expanded_src_to_dst_row` is arranged in column-major order.
     - `1`: enables drop-pad mode and `expanded_src_to_dst_row` is arranged in column-major order.
     - `2`: enables dropless mode and `expanded_src_to_dst_row` is arranged in row-major order.
     - `3`: enables drop-pad mode and `expanded_src_to_dst_row` is arranged in row-major order.

## Return Values

`Tensor`

Output parameter `out`, representing the final merged output of the MoE FFN. This parameter must be 2D with shape `(NUM_ROWS, H)`. The data type must be identical to that of `expanded_permuted_rows`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call
    - Example in dropless mode

        ```python
        >>> import torch
        >>> import torch_npu
        >>> expert_num = 16
        >>> token_len = 10
        >>> top_k = 4
        >>> num_rows = 50
        >>> device = torch.device('npu')
        >>> dtype = torch.float32
        >>> expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
        >>> skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        >>> skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        >>> bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        >>> scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        >>> expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        >>> expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device,dtype=torch.int32)
        >>> drop_pad_mode = 0
        >>> output = torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        >>> print(output.shape)
        torch.Size([50, 10])
        >>> print(output.dtype)
        torch.float32
        ```
      
    - Example in drop-pad mode

      ```python
        >>> import torch
        >>> import torch_npu
        >>> expert_num = 16
        >>> token_len = 10
        >>> top_k = 4
        >>> num_rows = 50
        >>> expert_capacity = 8
        >>> device = torch.device('npu')
        >>> dtype = torch.float32
        >>> expanded_permuted_rows = torch.randn((expert_num, expert_capacity, token_len), device=device, dtype=dtype)
        >>> skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        >>> skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        >>> bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        >>> scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        >>> expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        >>> expanded_src_to_dst_row = torch.randint(low=-1, high=expert_num * expert_capacity - 1, size=(num_rows * top_k,), device=device,dtype=torch.int32)
        >>> drop_pad_mode = 1
        >>> output = torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        >>> print(output.shape)
      torch.Size([50, 10])
        >>> print(output.dtype)
      torch.float32
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
        
        def forward(self, expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode):
            return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)

    def main():
        expert_num = 16
        token_len = 10
        top_k = 4
        num_rows = 50
        device =torch.device('npu')
        dtype = torch.float32

        expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
        skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device, dtype=torch.int32)
        drop_pad_mode = 0

        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        custom_output = model(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        print(custom_output.shape, custom_output.dtype)

    if __name__ == '__main__':
        main()
    
    # Expected output of the preceding code sample:
    torch.Size([50, 10]) torch.float32
    ```

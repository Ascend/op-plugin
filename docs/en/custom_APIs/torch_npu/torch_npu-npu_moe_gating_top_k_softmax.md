# torch_npu.npu_moe_gating_top_k_softmax

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Uses the gating mechanism of the Mixture of Experts (MoE) architecture to execute expert routing computations. This API performs Softmax computation on the input `x` followed by TopK computation to select the top-K experts with the highest weights.

- Formulas:

$$
softmaxOut = softmax(x, axis = -1) \\
yOut, expertIdxOut = topK(softmaxOut, k = k) \\
rowIdxRange = arange(expertIdxOut.shape[0] * expertIdxOut.shape[1])\\
rowIdxOut = rowIdxRange.reshape([expertIdxOut.shape[1], expertIdxOut.shape[0]]).transpose(1, 0)
$$

## Prototype

```python
torch_npu.npu_moe_gating_top_k_softmax(x, finished=None, k=1) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input to be processed, $x$ in the formulas. This parameter must be a 2D or 3D tensor. The data type can be `float16`, `bfloat16`, or `float32`. The data layout must be ND.
- **`finished`** (`Tensor`): Optional. Rows in the input that are excluded from the computation. This parameter must be a 2D or 3D tensor. The data type is `bool`. The shape is `gating_shape[:-1]`. The data layout must be ND. This parameter can be set to `None`. Valid values:`True` (the corresponding row does not participate in the computation) or `False` (the corresponding row participates in the computation).
- **`k`** (`int`): Optional. TopK selection count, $k$ in the formulas. The conditions `0 < k <= x.shape[-1]` and `k <= 1024` must be satisfied. The default value is `1`.

## Return Values

- **`y`** (`Tensor`): TopK values obtained after performing Softmax computation on `x`, $yOut$ in the formulas. This parameter must be a 2D or 3D tensor. The data type must be identical to that of `x`. The sizes of its non-last dimensions must match the corresponding dimension sizes of `x`. The size of its last dimension must match `k`. The data layout must be ND.
- **`expert_idx`** (`Tensor`): Indices of the TopK values obtained after performing Softmax computation on `x` (expert IDs), $expertIdxOut$ in the formula. The shape must be identical to the shape of `y`. The data type is `int32`. The data layout must be ND.
- **`row_idx`** (`Tensor`): Positional mapping between output row locations and input row locations, $rowIdxOut$ in the formula. The shape must be identical to the shape of `y`. The data type is `int32`. The data layout must be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    x = torch.rand((3, 3), dtype=torch.float32).to("npu")
    finished = torch.randint(2, size=(3,), dtype=torch.bool).to("npu")
    y, expert_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x, finished, k=2)
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
    class MoeGatingTopkSoftmaxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, finish, k):
            res = torch_npu.npu_moe_gating_top_k_softmax(x, finish, k)
            return res
    x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
    moe_gating_topk_softmax_model = MoeGatingTopkSoftmaxModel().npu()
    moe_gating_topk_softmax_model = torch.compile(moe_gating_topk_softmax_model, backend=npu_backend, dynamic=True)
    res = moe_gating_topk_softmax_model(x, None, 2)
    print(res)
    ```

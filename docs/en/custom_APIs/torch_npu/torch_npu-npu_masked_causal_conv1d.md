# torch_npu.npu_masked_causal_conv1d

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>           |    √     |

## Function

- API function: Performs masked causal 1D grouped convolution between tokens at the hidden layer.

- Formula:

    For the given input tensor input with shape [S, B, H] and the convolution weight `weight` with shape [W, H] (W = 3), perform the following computation:

    1. Perform causal zero padding on `input` along the sequence dimension (W – 1 zeros are padded at the beginning of the sequence), and then perform depthwise 1D convolution.

        $$
        \text{output}[s, b, h] = \sum_{k=0}^{W-1} \text{weight}[k, h] \cdot \text{input}[s - (W-1-k), b, h]
        $$

        The out-of-bounds `input` index is considered as 0 (causal zero padding).

    2. If `mask` (shape: [B, S], where `true` indicates a valid position) is provided, masking is performed on the output.

        $$
        \text{output}[s, b, :] = 0, \quad \text{if } \text{mask}[b, s] = \text{false}
        $$

## Prototype

```python
torch_npu.npu_masked_causal_conv1d(input, weight, *, mask=None) -> Tensor
```

## Parameters

> **Note**:
>
> - The dimensions of the `input` and `weight` parameters are as follows: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, and W (Window Size) indicates the size of the convolution window.

- **`input`** (`Tensor`): Required. Convolution input tensor. Non-contiguous tensors are supported. The data format can be $ND$, the data type can be `float16` or `bfloat16`, and the shape is [S, B, H].
- **`weight`** (`Tensor`): Required. Convolution weight tensor. Non-contiguous tensors are supported. The data format can be $ND$, the data type must match that of `input`, and the shape is [W, H], where W can only be 3 currently.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`mask`** (`Tensor`): Optional. Mask of the convolution output. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bool`. The shape is `[B, S]`, where `true` indicates a valid position and `false` indicates a position to be zeroed out. The default value is `None`, indicating that no mask operation is performed.

## Return Values

`Tensor`

Output result of causal convolution, `output` in the formulas. The shape and data type must match those of `input`, and the data layout is ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports both single-operator mode and graph mode.

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu

    S, B, H, W = 2048, 4, 768, 3
    input  = torch.randn(S, B, H, dtype=torch.bfloat16).npu()
    weight = torch.randn(W, H, dtype=torch.bfloat16).npu()
    mask   = torch.rand(B, S).npu() > 0.3  # bool [B, S]

    output = torch_npu.npu_masked_causal_conv1d(input, weight, mask=mask)
    # output shape: [S, B, H]
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

    class MaskedCausalConv1dModel(torch.nn.Module):
        def forward(self, input, weight, mask):
            return torch_npu.npu_masked_causal_conv1d(input, weight, mask=mask)

    S, B, H, W = 2048, 4, 768, 3
    input  = torch.randn(S, B, H, dtype=torch.bfloat16).npu()
    weight = torch.randn(W, H, dtype=torch.bfloat16).npu()
    mask   = torch.rand(B, S).npu() > 0.3

    model = MaskedCausalConv1dModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(input, weight, mask)
    ```

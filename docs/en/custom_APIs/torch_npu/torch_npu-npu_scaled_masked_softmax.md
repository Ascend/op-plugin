# torch_npu.npu_scaled_masked_softmax

> [!NOTICE]  
> This API is updated in this version. For details about the specific changes, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950DT</term>                        |    √    |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Computes the Softmax result after scaling the input tensor `x` and applying the `mask`.

## Prototype

```python
torch_npu.npu_scaled_masked_softmax(x, mask, scale=1.0, fixed_triu_mask=False) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input logits. The data type can be `float16`, `float32`, or `bfloat16`. The data layout can be ND or FRACTAL_NZ.
- **`mask`** (`Tensor`): Required. Input mask. The data type must be `bool`. The data layout can be ND or FRACTAL_NZ.
- **`scale`** (`float`): Optional. Scaling factor of `x`. The default value is `1.0`.
- **`fixed_triu_mask`** (`bool`): Reserved parameter. The corresponding feature is not implemented. Currently, only the default value `False` is supported. This parameter will support automatic generation of an upper-triangular `bool` mask after the feature is implemented.

## Return Values

`Tensor`

Output tensor, which is the Softmax result of the masked `x` on the last dimension. The output shape must be identical to that of `x`. The data type can be `float16`, `float32`, or `bfloat16`. The data layout can be ND or FRACTAL_NZ.

## Constraints

- Currently, when the shape of `x` is converted to the `NCHW` layout, the lengths of the H axis and W axis must be within the range [32, 4096] and must be divisible by 32.
- The shape of `mask` must be broadcastable to that of `x`.

## Examples

```python
>>> import torch
>>> import torch_npu
>>> shape = [4, 4, 2048, 2048]
>>> x = torch.rand(shape).npu()
>>> mask = torch.zeros_like(x).bool()
>>> scale = 1.0
>>> fixed_triu_mask = False
>>> output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
>>> print(output.shape)
torch.Size([4, 4, 2048, 2048])
```

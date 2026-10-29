# torch_npu.npu_scaled_masked_softmax

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

Computes the Softmax result after scaling the input tensor `x` and applying the `mask`.

- Formulas:

  $$
  \mathrm{out}=\operatorname{Softmax}_{\mathrm{dim}=-1}\left(\operatorname{Mask}(\mathrm{scale}\cdot x,\mathrm{mask})\right)
  $$

Here, `x` is the input logits, `scale` is the scaling coefficient, `mask` is a boolean mask, and `out` is the computation result.

## Prototype

```python
torch_npu.npu_scaled_masked_softmax(x, mask, scale=1.0, fixed_triu_mask=False) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input logits. The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`. The data layout can be ND or FRACTAL_NZ.
- **`mask`** (`Tensor`): Required. Input mask. The data type must be `torch.bool`. The data layout can be ND or FRACTAL_NZ.
- **`scale`** (`float`): Optional. Scaling factor of `x`. The default value is `1.0`. The supported data type is `torch.float32`.
- **`fixed_triu_mask`** (`bool`): Reserved parameter. The corresponding feature is not implemented. Currently, only the default value `False` is supported. This parameter will support automatic generation of an upper-triangular `bool` mask after the feature is implemented.

## Return Values

`Tensor`

Output tensor, which is the Softmax result of the masked `x` on the last dimension. The output shape must be identical to that of `x`. The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`. The data layout can be ND or FRACTAL_NZ.

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

# (beta) torch_npu.contrib.module.Focus

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Replaces the native `Focus` module in YOLOv5 with an NPU-optimized implementation.

## Prototype

```python
torch_npu.contrib.module.Focus(c1, c2, k=1, s=1, p=None, g=1, act=True)
```

## Parameters

**Computation Parameters**

- **`c1`** (`int`): Number of channels in the input image.
- **`c2`** (`int`): Number of output channels produced by the convolution.
- **`k`** (`int`): Optional. Convolution kernel size. The default value is `1`.
- **`s`** (`int`): Optional. Convolution stride. The default value is `1`.
- **`p`** (`int`): Optional. Padding size. The default value is `None`.
- **`g`** (`int`): Number of groups from the input channels to the output channels. The default value is `1`.
- **`act`** (`bool`): Specifies whether to use an activation function. The default value is `True`.

**Computation Input**

- **`x`** (`Tensor`): Input tensor.

## Return Values

`Tensor`

Focus computation result.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import Focus
>>> input = torch.randn(4, 8, 300, 40).npu()
>>> input.requires_grad_(True)
>>> fast_focus = Focus(8, 13).npu()
>>> output = fast_focus(input)
>>> output.sum().backward()
>>> print(output.shape)
torch.Size([4, 13, 150, 20])
```

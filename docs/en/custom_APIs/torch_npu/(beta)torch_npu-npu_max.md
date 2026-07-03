# (beta) torch_npu.npu_max

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Computes the maximum values along `dim`. This API is similar to `torch.max` and is optimized for NPUs.

## Prototype

```python
torch_npu.npu_max(self, dim, keepdim=False) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input tensor.
- **`dim`** (`int`): Required. Dimension to be reduced.
- **`keepdim`** (`bool`): Optional. Specifies whether to retain `dim` in the output tensor. The default value is `False`.

## Return Values

- **`values`** (`Tensor`): Maximum values in the input tensor.
- **`indices`** (`Tensor`): Indices of the maximum values in the input tensor.

## Example

```python
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> print(input)
tensor([[[[-1.8135,  0.2078],
          [-0.6678,  0.7846]],

        [[ 0.6458, -0.0923],
          [-0.2124, -1.9112]]],

        [[[-0.5800, -0.4979], 
         [ 0.2580,  1.1335]],

          [[ 0.6669,  0.1876],
          [ 0.1160, -0.1061]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_max(input, 2)
>>> print(outputs)
tensor([[[-0.6678,  0.7846],
        [ 0.6458, -0.0923]],

        [[ 0.2580,  1.1335],
        [ 0.6669,  0.1876]]], device='npu:0')
>>> print(indices)
tensor([[[1, 1],
        [0, 0]],

        [[1, 1],
        [0, 0]]], device='npu:0', dtype=torch.int32)
```

# (beta) torch_npu._npu_dropout

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Counts dropout results without using a random seed.

## Prototype

```python
torch_npu._npu_dropout(self, p) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`p`** (`float`): Dropout discard probability.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([1.,2.,3.,4.]).npu()
>>> print(input)
tensor([1., 2., 3., 4.], device='npu:0')
>>> prob = 0.3
>>> output, mask = torch_npu._npu_dropout(input, prob)
>>> print(output)
tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
>>> print(mask)
tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
      253, 255], device='npu:0', dtype=torch.uint8)
```

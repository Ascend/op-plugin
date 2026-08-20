# (beta) torch_npu.npu_one_hot

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Returns a one-hot tensor. The positions indicated by indices in `input` take the `on_value`, whereas all other positions take the `off_value`.

## Prototype

```python
torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Class values of any shape.
- **`num_classes`** (`int`): Axis to be filled. The default value is `-1`.
- **`depth`** (`int`): Depth of the one-hot dimension. The default value is `1`.
- **`on_value`** (`Scalar`): Value filled in the output when `indices[j] == i`. The default value is `1`.
- **`off_value`** (`Scalar`): Value filled in the output when `indices[j] != i`. The default value is `0`.

## Example

```python
>>> a=torch.IntTensor([5, 3, 2, 1]).npu()
>>> b=torch_npu.npu_one_hot(a, depth=5)
>>> print(b)
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]], device='npu:0')
```

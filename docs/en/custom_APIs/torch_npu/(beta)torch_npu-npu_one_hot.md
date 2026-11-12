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

Returns a one-hot tensor. Each value `v` in `input` is used as a class index: the output is filled with `on_value` at index `v` in the one-hot dimension (with depth `depth`), and all other positions are filled with `off_value`. `input` must be an integer index tensor. Element values are recommended to be in the range `[0, depth)`. For out-of-range elements, all corresponding output positions are filled with `off_value`.

## Prototype

```python
torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Class index tensor of any shape. Each element value is used as an index in the one-hot dimension. An integer tensor is recommended.
- **`num_classes`** (`int`): Axis position at which the one-hot dimension is inserted. The default value is `-1`, indicating that the one-hot dimension is appended after the last dimension of the input.
- **`depth`** (`int`): Depth of the one-hot dimension (number of classes). The default value is `1`. If set to `-1`, the depth is inferred as the maximum value of `input` plus 1.
- **`on_value`** (`Scalar`): Value filled at the output position where the index `i` in the one-hot dimension equals the corresponding value in `input`. The default value is `1`.
- **`off_value`** (`Scalar`): Value filled at the output position where the index `i` in the one-hot dimension does not equal the corresponding value in `input`. The default value is `0`.

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

# (beta) torch_npu.npu_one_hot

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Returns a one-hot tensor. The value v of each element in `input` is used as a class index: the output is filled with `on_value` at index v of the one_hot dimension (whose depth is `depth`), and all other positions are filled with `off_value`. `input` must be an integer index tensor. Element values are recommended to be in the range $[0, depth)$. The output corresponding to out-of-range elements is filled entirely with `off_value`.

## Prototype

```python
torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Class index tensor. The shape is unlimited. The value of each element is used as the index of the one_hot dimension. Passing an integer tensor is recommended.
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

# torch_npu.empty_with_swapped_memory

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Allocates a tensor whose device information is NPU while its actual memory resides on the host.

## Prototype

```python
torch_npu.empty_with_swapped_memory(size, dtype=None, device=None) -> Tensor
```

## Parameters

- **`size`** (`List[int]`): Required. Integer sequence defining the shape of the output tensor. It can be a variable number of arguments or a collection such as a list or tuple.
- **`dtype`** (`torch.dtype`): Optional. Data type of the generated `Tensor`. The default value is `None`, indicating that the global default `dtype` type is used.
- **`device`** (`torch.device`): Optional. Device information of the generated `Tensor`. The default value is `None`, indicating that the current default device is used.

## Return Values

`Tensor`

Special `Tensor` generated.

## Constraints

- This API does not support graph mode.

- Currently, the special `Tensor` allocated by this API supports only the following operators:<br>
`torch.fill_`<br>
`torch.zero_`<br>
`torch.mul_`<br>
`torch_npu.npu_apply_adam_w`<br>
`torch_npu.npu_hans_encode`<br>
`torch_npu.npu_hans_decode`<br>

- When the installed CANN version is 8.5.0 or later and the Ascend HDK version is 26.0.rc1 or later, the special tensor allocated by this API supports direct printing.
- When the installed CANN version is earlier than 8.5.0 or the Ascend HDK version is earlier than 26.0.rc1, the special tensor allocated by this API does not support direct printing. In this case, a warning log is printed. To view the value, convert it to a regular tensor through `mul_` before printing.

## Example

Single-operator call

```python
>>> import torch
>>> import torch_npu
>>> swapped_tensor = torch_npu.empty_with_swapped_memory([2, 2], dtype=torch.float32, device=torch.device("npu:0"))
>>> tmp_tensor = swapped_tensor.fill_(3.14)
>>> out = torch.empty_like(swapped_tensor).fill_(1).mul_(tmp_tensor)
>>> print(out)
tensor([[3.1400, 3.1400],
        [3.1400, 3.1400]], device='npu:0')
```

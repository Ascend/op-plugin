# at_npu::native::empty_with_swapped_memory

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Allocates a special tensor whose device configuration is NPU but whose actual memory resides on the host.

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor empty_with_swapped_memory(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<c10::Device> device_opt)
```

## Parameters

- **`size`** (`c10::IntArrayRef`): Required. Shape of the tensor to be created.
- **`dtype_opt`** (`c10::optional<at::ScalarType>`): Optional. Data type of the tensor to be created. If the value is `c10::nullopt`, the global default data type is used.
- `device_opt` (`c10::optional<c10::Device>`): Optional. Device configuration of the tensor to be created. If the value is `c10::nullopt`, the current default device is used.

## Return Values

`at::Tensor`

Special `Tensor` generated.

## Constraints

- This API does not support graph mode.

- Currently, the special `Tensor` allocated by this API supports only the following operators:<br>
`aten::fill_`<br>
`aten::zero_`<br>
`aten::mul_`<br>
`npu_apply_adam_w`<br>
`npu_hans_encode`<br>
`npu_hans_decode`<br>

- When the installed CANN version is 8.5.0 or later and the Ascend HDK version is 26.0.rc1 or later, the special tensor allocated by this API supports direct printing.
- When the installed CANN version is earlier than 8.5.0 or the Ascend HDK version is earlier than 26.0.rc1, the special tensor allocated by this API does not support direct printing. In this case, a warning log is printed. To view the value, convert it to a regular tensor through `mul_` before printing.

# at_npu::native::empty_with_swapped_memory

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

Allocates a special Tensor whose device information is NPU and whose actual memory is on the host.

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor empty_with_swapped_memory(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<c10::Device> device_opt)
```

## Parameters

- **`size`** (`c10::IntArrayRef`): Required. Shape of the Tensor to be generated.
- **`dtype_opt`** (`c10::optional<at::ScalarType>`): Optional. Data type of the Tensor to be generated. If the value is `c10::nullopt`, the global default data type is used.
- **`device_opt`** (`c10::optional<c10::Device>`): Optional. Device information of the Tensor to be generated. If the value is `c10::nullopt`, the current default device is used.

## Return Values

`at::Tensor`

The special Tensor generated.

## Constraints

- This API does not support graph mode.

- The special Tensor allocated by this API currently supports only the following operators:<br>
`aten::fill_`<br>
`aten::zero_`<br>
`aten::mul_`<br>
`npu_apply_adam_w`<br>
`npu_hans_encode`<br>
`npu_hans_decode`<br>

- When the installed CANN version is 8.5.0 or later and the Ascend HDK version is 26.0.rc1 or later, the special Tensor allocated by this API can be printed directly.
- When the installed CANN version is earlier than 8.5.0 or the Ascend HDK version is earlier than 26.0.rc1, the special Tensor allocated by this API cannot be printed directly. In this case, a warning is logged. To view the values, convert it to a normal Tensor by using `mul_` and then print it.

## Examples

```cpp
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

// Create a special Tensor with the specified shape and dtype.
at::Tensor tensor = at_npu::native::empty_with_swapped_memory({2, 3}, at::kFloat, c10::nullopt);

// The special Tensor allocated by this API must be converted to a normal Tensor by using the mul_ operator before its values can be viewed.
at::mul_(tensor, 1.0);
```

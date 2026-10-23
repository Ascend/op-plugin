# (beta) at_npu::native::npu_format_cast

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->

## Function

Converts the format of an NPU tensor. The return value type is `Tensor`, which represents the converted Tensor.

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format, c10::optional<int64_t> customize_dtype = c10::nullopt)
```

## Parameters

**`self`** (`Tensor`): Tensor whose format is to be converted.

**`acl_format`** (`int64_t`): Destination format for conversion.

**`customize_dtype`** (`int64_t`): Optional. Specifies the destination data type for the format conversion. This parameter can control the C0 value. The default value is `c10::nullopt`.

- If no value is passed, the default value is `c10::nullopt`; the default C0 value is 16 for the `torch.float32` and `torch.int32` data types, and 32 for the `torch.int8` data type;
- If `3` (corresponding to int(`torch.int32`)) is passed, the C0 value in the FRACTAL_NZ format is 8.

## Constraints

The `customize_dtype` parameter is supported only for Atlas A2 training products/Atlas A3 training products with CANN 9.1.0 or later. For other products or CANN versions earlier than 9.1.0, passing this parameter causes an exception.

<!-- npu="950" id6 -->
For <term>Ascend 950DT products</term>, the behavior of converting a tensor to a private format (such as FRACTAL_NZ) by this API is not restricted by `allow_internal_format = False`.
<!-- end id6 -->

<!-- npu="950" id7 -->
For <term>Ascend 950DT products</term>, the following special scenarios are currently not supported when converting a tensor to the FRACTAL_NZ format:

- If the `dtype` of `self` is the same as that of `customize_dtype` and the data type is `torch.float16` or `torch.bfloat16`, and the dimensions of `self` are expressed as [k, n], the case where k is 1 is not supported.
- After this API is called to convert a tensor to the FRACTAL_NZ format, operations that modify the Tensor are not supported, including `contiguous`, `pad`, `view`, `slice`, and so on.
- If either of the last two dimensions of the shape of `self` is 1, `transpose` is not supported after the conversion to the FRACTAL_NZ format.
<!-- end id7 -->

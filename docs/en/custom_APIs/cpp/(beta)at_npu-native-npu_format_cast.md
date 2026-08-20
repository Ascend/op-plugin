# (beta) at_npu::native::npu_format_cast

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format)
```

## Function

Converts the format of an NPU tensor. The return value type is `Tensor`, which represents the converted tensor.

## Parameters

**`self`** (`Tensor`): Tensor whose format is to be converted.

**`acl_format`** (`int64_t`): Destination format for conversion.

## Constraints

In <term>Ascend 950DT</term> scenarios, the following special cases are currently not supported when converting a tensor to the `FRACTAL_NZ` format:

- If the `dtype` of `self` is `float16` or `bfloat16`, and the dimensions of `self` are represented as `[k, n]`, the case where `k` is 1 is not supported.
- After calling this API to convert a tensor to the `FRACTAL_NZ` format, operations that modify the tensor, including `contiguous`, `pad`, `view`, and `slice`, are not supported.
- If either of the last two dimensions of the shape of `self` is 1, `transpose` is not supported after converting to the `FRACTAL_NZ` format.

## Supported Products

- <term>Ascend 950DT</term>
- <term>Atlas A3 training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas training products</term>
- <term>Atlas inference products</term>

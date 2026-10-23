# torch_npu.npu_dual_level_quant_matmul

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Two-level quantization for QuantMatmul to reduce precision loss.

- **Formula**:

  $$
  out = \sum_{i}^{level0GroupSize} x1Level0Scale @ x2Level0Scale \cdot \\
  \sum_{ij}^{level1GroupSize} ((x1Level1Scale @ x1_{ij}) @ (x2Level1Scale @ x2_{ij})) + bias
  $$

  - $x1_{ij}$ and $x2_{ij}$ represent the left and right input matrices, and $i$ and $j$ represent the i-th and j-th groups.
  - $x1Level1Scale$ is the first-level dequantization parameter of $x1$ and uses MX quantization, with $level1GroupSize$ being 32.
  - $x2Level1Scale$ is the first-level dequantization parameter of $x2$ and uses MX quantization, with $level1GroupSize$ being 32.
  - $x1Level0Scale$ is the zeroth-level dequantization parameter of $x1$ and uses per-group quantization, with $level0GroupSize$ being 512.
  - $x2Level0Scale$ is the zeroth-level dequantization parameter of $x2$ and uses per-group quantization, with $level0GroupSize$ being 512.
  - $bias$ is the bias matrix.

## Prototype

```python
torch_npu.npu_dual_level_quant_matmul(x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias=None, output_dtype=torch.bfloat16) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Left input matrix, $x1_{ij}$ in the formula. A 2D Tensor, not transposed, with shape $[M, K]$. The data layout can be ND only. The data type can be `torch_npu.float4_e2m1fn_x2` only.
- **`x2`** (`Tensor`): Required. Right input matrix, $x2_{ij}$ in the formula. A 2D Tensor, transposed, with shape $[N, K]$. The data layout can be `FRACTAL_NZ` only. The data type can be `torch_npu.float4_e2m1fn_x2` only.
- **`x1_level0_scale`** (`Tensor`): Required. Zeroth-level dequantization parameter of the left matrix, $x1Level0Scale$ in the formula. A 2D Tensor, not transposed, with shape $[M, ceil(K/level0GroupSize)]$. The data layout can be ND only. The data type can be `torch.float32` only.
- **`x2_level0_scale`** (`Tensor`): Required. Zeroth-level dequantization parameter of the right matrix, $x2Level0Scale$ in the formula. A 2D Tensor, not transposed, with shape $[ceil(K/level0GroupSize), N]$. The data layout can be ND only. The data type can be `torch.float32` only.
- **`x1_level1_scale`** (`Tensor`): Required. First-level dequantization parameter of the left matrix, $x1Level1Scale$ in the formula. A 3D Tensor, not transposed, with shape $[M, ceil(K/level1GroupSize), 2]$. The data layout can be ND only. The data type can be `torch.float8_e8m0fnu` only.
- **`x2_level1_scale`** (`Tensor`): Required. First-level dequantization parameter of the right matrix, $x2Level1Scale$ in the formula. A 3D Tensor, transposed, with shape $[N, ceil(K/level1GroupSize), 2]$. The data layout can be ND only. The data type can be `torch.float8_e8m0fnu` only.
- **`bias`** (`Tensor`): Optional. Bias input matrix, $bias$ in the formula. A 1D Tensor with shape $[N]$. Non-contiguous Tensors are not supported. The data layout can be ND only. The data type can be `torch.float32` only.
- **`output_dtype`** (`int`): Required. Output type. The value can be `torch.float16`/`torch.half` or `torch.bfloat16`. The default value is `torch.bfloat16`.

## Return Values

**`y`** (`Tensor`): Computation result, $out$ in the formula. The data type is determined by `output_dtype`. The data layout can be ND.

## Constraints

- This API supports training and inference scenarios.
- This API supports only single-operator mode.
- Empty Tensors and non-contiguous Tensors are not supported for any input or output Tensor.
- The first-level dequantization (level1, corresponding to `x1_level1_scale` and `x2_level1_scale`) supports only MX quantization, with level1GroupSize=32.
- The zeroth-level dequantization (level0, corresponding to `x1_level0_scale` and `x2_level0_scale`) supports only per-group quantization, and level0GroupSize must be 512.

## Examples

Single-operator call

```python
import torch
import torch_npu
import math
class A4W4Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias=None,
                output_dtype=torch.bfloat16):
        return torch_npu.npu_dual_level_quant_matmul(x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale,
                                                     x2_level1_scale, bias=bias, output_dtype=output_dtype)
def main():
    m = 256
    k = 1024
    n = 2048
    l0_group_size = 512
    l1_group_size = 32
    # The float4_e2m1fn_x2 data type can be represented by int8.
    cpu_x1 = torch.randint(-5, 5, (m, math.ceil(k / 2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (n, math.ceil(k / 2)), dtype=torch.int8)
    x1_level0_scale = torch.randint(-5, 5, (m, math.ceil(k / l0_group_size)), dtype=torch.float32)
    x2_level0_scale = torch.randint(-5, 5, (math.ceil(k / l0_group_size), n), dtype=torch.float32)
    # The float8_e8m0fnu data type can be represented by uint8.
    x1_level1_scale = torch.randint(124, 130, (m, math.ceil(k / l1_group_size / 2), 2), dtype=torch.uint8)
    x2_level1_scale = torch.randint(124, 130, (n, math.ceil(k / l1_group_size / 2), 2), dtype=torch.uint8)
    x2 = torch_npu.npu_format_cast(cpu_x2.npu(), 29, customize_dtype=cpu_x2.dtype)
    bias = torch.randint(-5, 5, (n,), dtype=torch.float32)
    model = A4W4Net().npu()
    npu_out = model(cpu_x1.npu(), x2.npu(), x1_level0_scale.npu(), x2_level0_scale.npu(), x1_level1_scale.npu(),
                    x2_level1_scale.npu(), bias=bias.npu(), output_dtype=torch.bfloat16)
    print(npu_out.cpu())
if __name__ == '__main__':
    main()
```

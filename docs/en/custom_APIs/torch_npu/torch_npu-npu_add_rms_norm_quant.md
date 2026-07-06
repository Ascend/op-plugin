# torch_npu.npu_add_rms_norm_quant

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components</term>|    √     |
| <term>Atlas inference products</term>                            |    √     |

## Function

- Description: The RMSNorm operator is a normalization operation commonly used in foundation models. Compared with the LayerNorm operator, it removes the mean subtraction step. The `torch_npu.npu_add_rms_norm_quant` operator fuses the Add operator before RMSNorm and the Quantize operator after RMSNorm, reducing data transfer operations.
- Formulas:
  - The AddRMSNorm computation process is as follows:

  $$
  x_i={x1}+{x2}
  $$

  $$
  y=\operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - Quantization computation process:
      - When `div_mode` is `True`:

      $$
      y1=round((y/scales1)+zero\_points1)
      $$

      $$
      y2=round((y/scales2)+zero\_points2)
      $$
    - When `div_mode` is `False`:

      $$
      y1=round((y*scales1)+zero\_points1)
      $$

      $$
      y2=round((y*scales2)+zero\_points2)
      $$

## Prototype

```python
torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, beta=None, scales2=None, zero_points2=None, *, axis=-1, epsilon=1e-06, div_mode=True) -> (y1, y2, x)
```

## Parameters

  - **`x1`** (`Tensor`): Required. Source data tensor in the normalization process, $x1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type can be `float16` or `bfloat16`. The shape must have 1 to 8 dimensions.
  - **`x2`** (`Tensor`): Required. Source data tensor in the normalization process, $x2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type can be `float16` or `bfloat16`. The shape must have 1 to 8 dimensions. The data type and shape must be identical to those of `x1`.
  - **`gamma`** (`Tensor`): Required. Weight tensor in the normalization process, $gamma$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type can be `float16` or `bfloat16`. The shape must have 1 to 8 dimensions. The shape must be identical to the dimensions of `x1` that require normalization. The data type must be identical to that of `x1`.
  - **`scales1`** (`Tensor`): Required. Scales tensor obtained during `y1` quantization, $scales1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type can be `float32` or `bfloat16`. The shape must be identical to that of `gamma`. When `div_mode` is set to `True`, this parameter value must not be 0.
  - **`zero_points1`** (`Tensor`): Optional. Offset tensor obtained during `y1` quantization, $zero\_points1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type can be `int32` or `bfloat16`. The shape must be identical to that of `gamma`.
  - **`beta`** (`Tensor`): Optional. Bias term in the normalization process, $beta$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `gamma` and can be `float16` or `bfloat16`. The shape must be identical to that of `gamma`. The default value is `None`.
  - **`scales2`** (`Tensor`): Optional. Scales tensor obtained during `y2` quantization, $scales2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `scales1` and can be `float32` or `bfloat16`. The shape must be identical to that of `gamma`. When `div_mode` is set to `True`, this parameter value must not be 0. The default value is `None`.
  - **`zero_points2`** (`Tensor`): Optional. Offset tensor obtained during `y2` quantization, $zero\_points2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `zero_points1` and can be `int32` or `bfloat16`. The shape must be identical to that of `gamma`. The default value is `None`.
  - **`axis`** (`int64_t`): Optional. Elementwise axis along which quantization is performed, while other axes are broadcast. The specified axis must not exceed the number of dimensions of the input `x`. Currently, only the default value `-1` is supported. Other values do not take effect.
  - **`epsilon`** (`double`): Optional. Input $epsilon$ in the formulas, used to prevent division-by-zero errors. The data type must be `double`. A small positive number is recommended. The default value is `1e-6`.
  - `div_mode` (`bool`): Optional. Determines whether the quantization formula uses division. The data type must be `bool`. The default value is `True`.

## Return Values

  - **`y1`** (`Tensor`): Output tensor after quantization, $y1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type is `int8`. The shape must be identical to that of the input `x1`.
  - **`y2`** (`Tensor`): Output tensor after quantization, $y2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type is `int8`. The shape must be identical to that of the input `x1`.
  - **`x`** (`Tensor`): Sum of `x1` and `x2`, $x$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. The data type and shape must be identical to those of the input `x1`.

## Constraints

- Atlas inference products: The number of data elements in the last dimension of `x1` and `x2` must be greater than or equal to 32. The number of data elements in `gamma`, `beta`, `scales1`, `zero_points1`, `scales2`, and `zero_points2` must be greater than or equal to 32.

- **Boundary value scenarios**

  - Atlas inference products: Inputs containing `inf` or `nan` are not supported.
  - Atlas A2 training products/Atlas 800I A2 inference products/A200I A2 Box heterogeneous components and Atlas A3 training products/Atlas A3 inference products: When the input is `inf`, the output is `inf`. When the input is `nan`, the output is `nan`.

- **Dimension boundaries**

  The size of each dimension in the shapes of `x1`, `x2`, `gamma`, `scales1`, `zero_points1`, `beta`, `scales2`, `zero_points2`, `y1`, `y2`, and `x` must be less than or equal to 2147483647, which is the maximum value of `int32`. 

- **Data types supported by different product models**
  - Atlas A2 training products/Atlas 800I A2 inference products/A200I A2 Box heterogeneous components and Atlas A3 training products/Atlas A3 inference products:

     | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | beta | y1 | y2 | x |
     | ---------- | ---------- | ------------- | --------------- | ----------------------- | --------------------------- | --------------------------- | -------------------- | ------------- | ------------- | ------------ |
     | float16    | float16    | float16       | float32         | float32                 | int32                       | int32                       | float16              | int8          | int8          | float16      |
     | bfloat16   | bfloat16   | bfloat16      | bfloat16        | bfloat16                | bfloat16                    | bfloat16                    | bfloat16             | int8          | int8          | bfloat16     |

  - Atlas inference products

    | x1 | x2 | gamma | scales1| scales2 | zero_points1 | zero_points2 | beta | y1 | y2 | x |
    | ---------- | ---------- | ------------- | --------------- | ----------------------- | --------------------------- | --------------------------- | -------------------- | ------------- | ------------- | ------------ |
    | float16    | float16    | float16       | float32         | float32                 | int32                       | int32                       | float16              | int8          | int8          | float16      |
    
## Example

```python
import math

import numpy as np
import torch

import torch_npu

def test_npu_add_rms_norm_quant():
    shape_list = [[[16, ], [16, ]],
                  [[2, 16], [16, ]],
                  [[2, 16], [2, 16]],
                  [[16, 32], [16, 32]],
                  [[16, 32], [32, ]],
                  [[2, 2, 2, 8, 16, 32], [2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 8, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 16, 32], [2, 2, 2, 2, 2, 16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [2, 2, 2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [32, ]]]
    for item in shape_list:
        x_shape = item[0]
        quant_shape = item[1]
        x1 = torch.randn(x_shape, dtype=torch.float16)
        x2 = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        scales1 = torch.randn(quant_shape, dtype=torch.float32)
        zero_points1 = torch.randint(-10, 10, quant_shape, dtype=torch.int32)

        x1_npu = x1.npu()
        x2_npu = x2.npu()
        gamma_npu = gamma.npu()
        beta_npu = beta.npu()
        scales1_npu = scales1.npu()
        zero_points1_npu = zero_points1.npu()

        y1_v1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
        y1_v2, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu, beta_npu) 

if __name__ == "__main__":
    test_npu_add_rms_norm_quant()
```

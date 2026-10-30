# (beta) torch_npu.npu_format_cast

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

Changes the data layout of `input` to the target layout. The new tensor after the change does not replace the original tensor.

## Prototype

```python
torch_npu.npu_format_cast(input, acl_format, *, customize_dtype=None, input_dtype=None) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor to be processed.

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: 2 to 6 dimensions are supported, and the data types can be `torch.int8`, `torch.uint8`, `torch.float8_e4m3fn`, `torch.float8_e8m0fnu`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, `torch.int32`, `torch.float16`, and `torch.bfloat16`.
  <!-- end id6 -->

- **`acl_format`** (`int`/`Format`): Required. Target format. An integer or a `torch_npu.Format` type can be provided. The `torch_npu.Format` type will be automatically converted to the corresponding integer value. For example, when modifying the data layout format of `input` to the ND layout format, provide either `2` or `torch_npu.Format.ND`. `torch_npu.Format` indicates the data layout format of TorchNPU. The following table describes the data formats supported by TorchNPU.

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: currently, only 29 (ACL\_FORMAT\_FRACTAL\_NZ) and 2 (ACL\_FORMAT\_ND) are supported.
  <!-- end id7 -->
  - torch_npu supports the following data layouts.

    |torch_npu.Format Type|Integer Value|Description|
    | ------| ------|:------: |
    |torch_npu.Format.UNDEFINED|-1|Unknown data type. The corresponding AscendCL data layout format is `ACL_FORMAT_UNDEFINED`.|
    |torch_npu.Format.NCHW|0|NCHW layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_NCHW`.|
    |torch_npu.Format.NHWC|1|NHWC layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_NHWC`.|
    |torch_npu.Format.ND|2|Indicates that any format is supported. Except for operators that process a single input on itself such as `Square` and `Tanh`, other operators must be used with caution. The corresponding AscendCL data layout format is `ACL_FORMAT_ND`.|
    |torch_npu.Format.NC1HWC0|3|5D data layout format. `C0` is strongly related to the microarchitecture. This value is identical to the size of the cube unit, for example, 16. `C1` is obtained by slicing the C dimension based on `C0`: `C1=C/C0`. If the result is not divisible, the last part of the data must be padded to `C0`. The corresponding AscendCL data layout format is `ACL_FORMAT_NC1HWC0`.|
    |torch_npu.Format.FRACTAL_Z|4|Layout format of convolution weights. The corresponding AscendCL data layout format is `ACL_FORMAT_FRACTAL_Z`.|
    |torch_npu.Format.NC1HWC0_C04|12|5D data layout format. `C0` is fixed to 4. `C1` is obtained by slicing the C dimension based on `C0`: `C1=C/C0`. If the result is not divisible, the last part of the data must be padded to `C0`. It is not supported by the current version. The corresponding AscendCL data layout format is `ACL_FORMAT_NC1HWC0_C04`.|
    |torch_npu.Format.HWCN|16|HWCN layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_HWCN`.|
    |torch_npu.Format.NDHWC|27|NDHWC layout format. For 3D images, a layout format with a depth (D) dimension must be used. The corresponding AscendCL data layout format is `ACL_FORMAT_NDHWC`.|
    |torch_npu.Format.FRACTAL_NZ|29|Internal layout format, which currently does not need to be used by users. The corresponding AscendCL data layout format is `ACL_FORMAT_FRACTAL_NZ`.|
    |torch_npu.Format.NCDHW|30|NCDHW layout format. For 3D images, a layout format with a depth (D) dimension must be used. The corresponding AscendCL data layout format is `ACL_FORMAT_NCDHW`.|
    |torch_npu.Format.NDC1HWC0|32|6D data layout format. Compared with `NC1HWC0`, only a depth (D) dimension is added. The corresponding AscendCL data layout format is `ACL_FORMAT_NDC1HWC0`.|
    |torch_npu.Format.FRACTAL_Z_3D|33|3D convolution weight layout format. Operators such as `Conv3D`, `MaxPool3D`, and `AvgPool3D` must be expressed in this layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_FRACTAL_Z_3D`.|
    |torch_npu.Format.NC|35|2D data layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_NC`.|
    |torch_npu.Format.NCL|47|3D data layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_NCL`.|

    > [!NOTE]
    > For details about data layouts, see the <a href="https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/en/guide/technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md">Data Layout</a> section in *CANN Ascend C Operator Development*.

<!-- npu="950,A3,910b" id15 -->
- **`customize_dtype`** (`int`): Optional. Target data type for the layout conversion. This parameter can control the C0 value. The default value is `None`. The default C0 value of the `torch.float32` and `torch.int32` data types is 16, and the default C0 value of the `torch.int8` data type is 32.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: when `3` (corresponding to int(`torch.int32`)) is passed, the C0 value in the FRACTAL_NZ format is 8.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - For MatMul scenarios where only the weight is quantized, when converting the weight W to a private format, the data type of matrix A must be passed to infer the C0 axis size of W. The data types can be `torch.int8`, `torch.float8_e4m3fn`, `torch.float16`, and `torch.bfloat16`. If the default value `None` is used, the dtype of A is the same as the dtype of W, and the C0 axis size of W is inferred accordingly.
    - For MX scale scenarios, `torch.float8_e8m0fnu` is supported. The default value `None` uses the actual data type specified by `input_dtype`.
  <!-- end id9 -->

<!-- end id15 -->
- **`input_dtype`** (`int`): Optional. Actual data type of `input`. `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, and `torch.float8_e8m0fnu` are supported.

  <!-- npu="A3,910b,910,310p" id10 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products</term>, and <term>Atlas A3 training products</term>: this parameter is not supported currently. Use the default value `None`.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: this parameter is supported. The default value `None` indicates that the actual type of `input` is the same as the tensor data type.
  <!-- end id11 -->

## Return Values

**`out`** (`Tensor`): Output Tensor after the data layout is changed.

## Constraints

<!-- npu="950,A3,910b" id16 -->
The `customize_dtype` parameter is supported only on Atlas A2 training products/Atlas A3 training products/Ascend 950PR&950DT products with CANN 9.1.0 or later. On other products or with CANN versions earlier than 9.1.0, passing this parameter causes an exception.
<!-- end id16 -->

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:

  Currently, the following combinations of input parameters are supported. When the third combination is passed, the converted format is 50 (ACL\_FORMAT\_FRACTAL\_NZ\_C0\_16).

    | `input` Data Type | `input` Format | `acl_format` Value | `customize_dtype` Value | `input_dtype` Value |
    | --- | --- | --- | --- | --- |
    | `torch.int8` | $ND$ | 29 | `torch.int8` | NA |
    | `torch.float8_e4m3fn` | $ND$ | 29 | `torch.float8_e4m3fn` | NA |
    | `torch.int32` | $ND$ | 29 | `torch.float16`/`torch.bfloat16` | NA |
    | `torch.float16` | $ND$ | 29 | `torch.float16` | NA |
    | `torch.bfloat16` | $ND$ | 29 | `torch.bfloat16` | NA |
    | `torch.float8_e8m0fnu` (carried by `torch.uint8`) | $ND$/$NCL$/$NCHW$ | 29 | `torch.float8_e8m0fnu`/`None` | `torch.float8_e8m0fnu` |
    | `torch_npu.float4_e2m1fn_x2` (carried by `torch.float32`) | $ND$ | 29 | `torch.float8_e4m3fn` | NA |
    | `torch_npu.float4_e2m1fn_x2` (carried by `torch.uint8`/`torch.int8`) | $ND$ | 29 | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e2m1fn_x2` (carried by `torch.uint8`/`torch.int8`) | $FRACTAL\_NZ$ | 2 | NA | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e2m1fn_x2` (carried by `torch.uint8`/`torch.int8`) | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e1m2fn_x2` (carried by `torch.uint8`/`torch.int8`) | $ND$ | 29 | NA | `torch_npu.float4_e1m2fn_x2` |
    | `torch.uint8` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.uint8` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.uint8` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |
    | `torch.int8` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.int8` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.int8` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |
    | `torch.float8_e4m3fn` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.float16` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.bfloat16` | $FRACTAL\_NZ$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_2$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_4$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.int32` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_2$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_4$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_16$ | 2 | NA | NA |
    | `torch.float32` | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | NA |

  Special scenarios:

  - In the MX scale format conversion scenario, the actual data type of `input` is `torch.float8_e8m0fnu`. Only 3-D or 4-D input is supported, every dimension size must be greater than 0, and the last dimension must be 2. The output storage shape is `(..., ceil(N/16), KScale, 16, 2)`, where `KScale` and `N` are the sizes of the third-from-last and second-from-last input dimensions, respectively; preceding dimensions remain unchanged. The logical shape is the same as the input shape.

  Special scenarios not supported currently:

  - When the actual data type of `input` is `torch.float8_e8m0fnu`, MX scale format conversion does not support graph mode and supports only single-operator mode.
  - In the $ND$ to $FRACTAL\_NZ$ scenario, when `srcTensor.dtype` and `additionalDtype` are the same and of the `torch.float16` or `torch.bfloat16` type, if the dimensions are expressed as [k, n], the scenario where k is 1 is not supported currently.
  - After this API is called to convert to the $FRACTAL\_NZ$ format, any operation that can modify the Tensor, including contiguous, pad, view, and slice, is not supported.
  - In the scenario where either of the last two dimensions of the `srcTensor` shape is equal to 1, after conversion to $FRACTAL\_NZ$, any operation that can modify the Tensor, including transpose, is not allowed.
  - In the $FRACTAL\_NZ$ to $ND$ scenario, non-contiguous input `srcTensor` is not supported.
<!-- end id12 -->

## Return Values

`Tensor`

Returns the new tensor after the change.

## Example

- Example of calling with an integer value:

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> torch_npu.get_npu_format(x)
    0
    >>> x1 = torch_npu.npu_format_cast(x, 2)
    >>> torch_npu.get_npu_format(x1)
    2
    ```

- Example of calling with a Format type:

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> x2 = torch_npu.npu_format_cast(x, torch_npu.Format.NHWC)
    >>> torch_npu.get_npu_format(x2)
    1
    ```

<!-- npu="A3,910b" id13 -->
- Using customize_dtype to specify C0=8 (only for <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term> with CANN >= 9.1.0):

    ```python
    >>> import torch
    >>> import torch_npu
    >>> # On Atlas A2/A3 products, allow_internal_format defaults to False. The private format switch must be enabled first; otherwise, the FRACTAL_NZ target is degraded to ND.
    >>> torch_npu.npu.config.allow_internal_format = True
    >>> t = torch.randint(0, 100, (32, 32), dtype=torch.int32).npu()
    >>> # customize_dtype=3 means ACL_INT32, and in this case C0=8
    >>> out = torch_npu.npu_format_cast(t, 29, customize_dtype=3)
    >>> torch_npu.get_npu_format(out)
    29
    ```
<!-- end id13 -->
<!-- npu="950" id14 -->
- Using `input_dtype` (only for <term>Ascend 950PR&950DT products</term>):

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randint(-5, 5, (1,64), dtype=torch.int8).npu()
    >>> y = torch_npu.npu_format_cast(x, 29, customize_dtype=torch.int8, input_dtype=torch.int8)
    >>> torch_npu.get_npu_format(y)
    29
    ```
<!-- end id14 -->

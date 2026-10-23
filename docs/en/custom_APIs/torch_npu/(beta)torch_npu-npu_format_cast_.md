# (beta) torch_npu.npu_format_cast_

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

Modifies the data format of `input` to the target format in place.

## Prototype

```python
torch_npu.npu_format_cast_(input, src, *, customize_dtype=None, input_dtype=None) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input Tensor to be processed.

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: 2D and 3D data are supported. The data type can be `torch.int8`, `torch.float8_e4m3fn`, `torch.int32`, `torch.float16`, or `torch.bfloat16`.
  <!-- end id6 -->

- **`src`** (`Tensor`/`int`/`Format`): Required. Target format. The value can be a Tensor, an integer, or a torch_npu.Format value.

  - If a Tensor is input, the data format of `input` is changed to the format of this Tensor. For example, to convert the data format of `input` to the ND format, input *a Tensor in ND format* here.
  - If an integer is input, the data format of `input` is changed to the torch_npu.Format corresponding to the integer value. For example, to convert the data format of `input` to the ND format, input `2` here.

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: Currently, only 29 (ACL\_FORMAT\_FRACTRAL\_NZ) is supported.
    <!-- end id7 -->

  - If a torch_npu.Format value is input, the data format of `input` is changed to that format. For example, to convert the data format of `input` to the ND format, input `torch_npu.Format.ND` here. torch_npu.Format indicates the data format of torch_npu. torch_npu supports the following data formats:

    |torch_npu.Format Value|Integer Value|Description|
    | ------| ------|:------: |
    |torch_npu.Format.UNDEFINED|-1|Unknown data format. The corresponding AscendCL data format is ACL_FORMAT_UNDEFINED.|
    |torch_npu.Format.NCHW|0|NCHW format. The corresponding AscendCL data format is ACL_FORMAT_NCHW.|
    |torch_npu.Format.NHWC|1|NHWC format. The corresponding AscendCL data format is ACL_FORMAT_NHWC.|
    |torch_npu.Format.ND|2|Indicates that any format is supported. Except for operators such as Square and Tanh that process a single input on their own, other operators must be used with caution. The corresponding AscendCL data format is ACL_FORMAT_ND.|
    |torch_npu.Format.NC1HWC0|3|5D data format. C0 is strongly related to the microarchitecture and is equal to the size of the cube unit, for example, 16. C1 is obtained by splitting the C dimension by C0: C1 = C/C0. If the result is not divisible, the last piece of data needs to be padded to C0. The corresponding AscendCL data format is ACL_FORMAT_NC1HWC0.|
    |torch_npu.Format.FRACTAL_Z|4|Format of convolution weights. The corresponding AscendCL data format is ACL_FORMAT_FRACTAL_Z.|
    |torch_npu.Format.NC1HWC0_C04|12|5D data format. C0 is fixed to 4. C1 is obtained by splitting the C dimension by C0: C1 = C/C0. If the result is not divisible, the last piece of data needs to be padded to C0. This format is not supported in the current version. The corresponding AscendCL data format is ACL_FORMAT_NC1HWC0_C04.|
    |torch_npu.Format.HWCN|16|HWCN format. The corresponding AscendCL data format is ACL_FORMAT_HWCN.|
    |torch_npu.Format.NDHWC|27|NDHWC format. For 3D images, the format with the D (Depth) dimension is required. The corresponding AscendCL data format is ACL_FORMAT_NDHWC.|
    |torch_npu.Format.FRACTAL_NZ|29|Internal format, which does not need to be used by users. The corresponding AscendCL data format is ACL_FORMAT_FRACTAL_NZ.|
    |torch_npu.Format.NCDHW|30|NCDHW format. For 3D images, the format with the D (Depth) dimension is required. The corresponding AscendCL data format is ACL_FORMAT_NCDHW.|
    |torch_npu.Format.NDC1HWC0|32|6D data format. Compared with NC1HWC0, only the D (Depth) dimension is added. The corresponding AscendCL data format is ACL_FORMAT_NDC1HWC0.|
    |torch_npu.Format.FRACTAL_Z_3D|33|3D convolution weight format, which is required by operators such as Conv3D, MaxPool3D, and AvgPool3D. The corresponding AscendCL data format is ACL_FORMAT_FRACTAL_Z_3D.|
    |torch_npu.Format.NC|35|2D data format. The corresponding AscendCL data format is ACL_FORMAT_NC.|
    |torch_npu.Format.NCL|47|3D data format. The corresponding AscendCL data format is ACL_FORMAT_NCL.|

    > [!NOTE]
    > For details about the data layout formats, see the "<a href="https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/en/guide/technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md">Data Layout Formats</a>" section in the *CANN Ascend C Operator Development* guide.

<!-- npu="950,A3,910b" id14 -->
- **`customize_dtype`** (`int`): Optional. Specifies the destination data type for the format conversion. This parameter can control the C0 value. The default value is `None`. The default C0 value is 16 for the `torch.float32` and `torch.int32` data types, and 32 for the `torch.int8` data type.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: When `3` (corresponding to int(`torch.int32`)) is passed, the C0 value in the FRACTAL_NZ format is 8.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: For MatMul scenarios where only the weight is quantized, when converting the weight W to a private format, the data type of matrix A must be passed to infer the C0 axis size of W. The data type can be `torch.int8`, `torch.float8_e4m3fn`, `torch.float16`, or `torch.bfloat16`. If the default value `None` is used, the dtype of A is the same as the dtype of W, and the C0 axis size of W is inferred accordingly.
  <!-- end id9 -->

<!-- end id14 -->
- **`input\_dtype`** (`int`): Indicates the actual data type of `input`. It is mainly used for data types not native to torch, such as `float4`.

  <!-- npu="A3,910b,910,310p" id10 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products</term>, and <term>Atlas A3 training products</term>: This parameter is not supported. Use the default value `None`.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter is supported. The default value `None` indicates that the actual type of `input` is the same as the data type of the tensor.
  <!-- end id11 -->

## Constraints

<!-- npu="950,A3,910b" id15 -->
The `customize_dtype` parameter is supported only for Atlas A2 training products/Atlas A3 training products/Ascend 950PR&950DT products with CANN 9.1.0 or later. For other products or CANN versions earlier than 9.1.0, passing this parameter causes an exception.

<!-- end id15 -->
<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:

  Currently, the following combinations of input parameters are supported. When the third combination is passed, the converted format is 50 (ACL\_FORMAT\_FRACTRAL\_NZ\_C0\_16).

    | `input` Data Type | `input` Format | `acl_format` Value | `customize_dtype` Value | `input_dtype` Value |
    | --- | --- | --- | --- | --- |
    | `torch.int8` | $ND$ | 29 | `torch.int8` | NA |
    | `torch.float8_e4m3fn` | $ND$ | 29 | `torch.float8_e4m3fn` | NA |
    | `torch.int32` | $ND$ | 29 | `torch.float16`/`torch.bfloat16` | NA |
    | `torch.float16` | $ND$ | 29 | `torch.float16` | NA |
    | `torch.bfloat16` | $ND$ | 29 | `torch.bfloat16` | NA |
    | `torch_npu.float4_e2m1fn_x2` (`torch.float32` storage) | $ND$ | 29 | `torch.float8_e4m3fn` | NA |
    | `torch_npu.float4_e2m1fn_x2` (`torch.uint8`/`torch.int8` storage) | $ND$ | 29 | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e2m1fn_x2` (`torch.uint8`/`torch.int8` storage) | $FRACTAL\_NZ$ | 2 | NA | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e2m1fn_x2` (`torch.uint8`/`torch.int8` storage) | $FRACTAL\_NZ\_C0\_32$ | 2 | NA | `torch_npu.float4_e2m1fn_x2` |
    | `torch_npu.float4_e1m2fn_x2` (`torch.uint8`/`torch.int8` storage) | $ND$ | 29 | NA | `torch_npu.float4_e1m2fn_x2` |
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

  The following special scenarios are currently not supported:

  - In the $ND$ to $FRACTAL\_NZ$ scenario, when `srcTensor.dtype` is the same as `additionalDtype` and the data type is `torch.float16` or `torch.bfloat16`, if the dimensions are expressed as [k, n], the case where k is 1 is not supported.
  - After this API is called to convert to the $FRACTAL\_NZ$ format, operations that modify the Tensor are not supported, including contiguous, pad, view, slice, and so on.
  - If either of the last two dimensions of the shape of `srcTensor` is 1, no operation that modifies the Tensor (including transpose) is allowed after the conversion to $FRACTAL\_NZ$.
  - In the $FRACTAL\_NZ$ to $ND$ scenario, non-contiguous `srcTensor` input is not supported.
<!-- end id12 -->

## Return Values

`Tensor`

Returns the `input` modified in place.

## Examples

- Integer value call example:

    ```python
     >>> import torch
     >>> import torch_npu
     >>> x = torch.rand(2, 3, 4, 5).npu()
     >>> torch_npu.get_npu_format(x)
     0
     >>> torch_npu.get_npu_format(torch_npu.npu_format_cast_(x, 2))
     2
    ```

- Format type call example:

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> x2 = torch_npu.npu_format_cast_(x, torch_npu.Format.NHWC)
    >>> torch_npu.get_npu_format(x2)
    1
    ```

<!-- npu="950" id13 -->
- Using `input_dtype` (only for <term>Ascend 950PR&950DT products</term>):

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randint(-5, 5, (1,64), dtype=torch.int8).npu()
    >>> y = torch_npu.npu_format_cast(x, 29, customize_dtype=torch.int8, input_dtype=torch.int8)
    >>> torch_npu.get_npu_format(y)
    29
    ```
<!-- end id13 -->

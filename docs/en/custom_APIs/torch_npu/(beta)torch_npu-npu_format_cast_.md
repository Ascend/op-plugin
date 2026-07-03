# (beta) torch_npu.npu_format_cast_

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Modifies the data layout format of `input` to the target format in place.

## Prototype

```python
torch_npu.npu_format_cast_(input, src) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor to be processed.
- **`src`** (`Tensor`/`int`/`Format`): Required. Target format. A tensor, integer, or `torch_npu.Format` type can be provided.

  - If a tensor is provided, the data layout format of `input` is modified to that of this tensor. For example, if the data layout format of `input` is converted to ND, a tensor in ND layout format can be provided here.
  - If an integer is provided, the data layout format of `input` is modified to the `torch_npu.Format` corresponding to the integer value. For example, if the data layout format of `input` is converted to ND, `2` can be provided here.
  - If a `torch_npu.Format` type is provided, the data layout format of `input` is modified to this format. For example, if the data layout format of `input` is converted to ND, `torch_npu.Format.ND` can be provided here. `torch_npu.Format` indicates the data layout format of `torch_npu`. The following table describes the data formats supported by `torch_npu`.

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
    |torch_npu.Format.FRACTAL_Z_3D|33|3D convolution weight layout format. Operators such as `Conv3D`, `MaxPool3D`, and `AvgPool3D` must be expressed in this layout format. The corresponding AscendCL data layout format is `ACL_FRACTAL_Z_3D`.|
    |torch_npu.Format.NC|35|2D data layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_NC`.|
    |torch_npu.Format.NCL|47|3D data layout format. The corresponding AscendCL data layout format is `ACL_FORMAT_NCL`.|
    
    > [!NOTE]  
    > For details about the data layout format, see <a href="https://www.hiascend.com/document/detail/en/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_0099.html">Data Layout Formats</a> in *CANN Ascend C Operator Development*.

## Return Values

`Tensor`

Returns the modified `input`.

## Examples

- Call with an integer value:

    ```python
     >>> x = torch.rand(2, 3, 4, 5).npu()
     >>> torch_npu.get_npu_format(x)
     0
     >>> torch_npu.get_npu_format(torch_npu.npu_format_cast_(x, 2))
     2
    ```

- Call with a format type:

    ```python
    >>> torch_npu.get_npu_format(torch_npu.npu_format_cast_(x, torch_npu.Format.NHWC))
    1
    ```

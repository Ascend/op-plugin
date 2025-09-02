# （beta）torch_npu.npu_format_cast

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

修改`input`的数据格式为目标格式，修改后的新张量不会替换原有张量。

## 函数原型

```
torch_npu.npu_format_cast(input, acl_format) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：必选参数，待处理的输入张量。
- **acl_format**（`int`/`Format`）：必选参数，目标格式。可输入整数或torch_npu.Format类型，torch_npu.Format类型会被自动转换为对应格式的整数值。例如将`input`的数据格式修改为ND格式时，此处既可以输入`2`，也可以输入`torch_npu.Format.ND`。torch_npu.Format表示torch_npu的数据格式，torch_npu支持如下数据格式：

    |torch_npu.Format类型|整数值|说明|
    | ------| ------|:------: |
    |torch_npu.Fromat.UNDEFINED|-1|未知数据类型。对应的AscendCL数据格式为ACL_FORMAT_UNDEFINED。|
    |torch_npu.Fromat.NCHW|0|NCHW格式。对应的AscendCL数据格式为ACL_FORMAT_NCHW。|
    |torch_npu.Fromat.NHWC|1|NHWC格式。对应的AscendCL数据格式为ACL_FORMAT_NHWC。|
    |torch_npu.Fromat.ND|2|表示支持任意格式，除了Square、Tanh等这些单输入对自身处理的算子外，其他算子需谨慎使用。对应的AscendCL数据格式为ACL_FORMAT_ND。|
    |torch_npu.Fromat.NC1HWC0|3|5维数据格式。其中，C0与微架构强相关，该值等于cube单元的size，例如16；C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要填充到C0。对应的AscendCL数据格式为ACL_FORMAT_NC1HWC0。|
    |torch_npu.Fromat.FRACTAL_Z|4|卷积的权重的格式。对应的AscendCL数据格式为ACL_FORMAT_FRACTAL_Z。|
    |torch_npu.Fromat.NC1HWC0_C04|12|5维数据格式。其中，C0固定为4，C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要padding到C0。当前版本不支持。对应的AscendCL数据格式为ACL_FORMAT_NC1HWC0_C04。|
    |torch_npu.Fromat.HWCN|16|HWCN格式。对应的AscendCL数据格式为ACL_FORMAT_HWCN。|
    |torch_npu.Fromat.NDHWC|27|NDHWC格式。对于3维图像就需要使用带D（Depth）维度的格式。对应的AscendCL数据格式为ACL_FORMAT_NDHWC。|
    |torch_npu.Fromat.FRACTAL_NZ|29|内部格式，用户目前无需使用。对应的AscendCL数据格式为ACL_FORMAT_FRACTAL_NZ。|
    |torch_npu.Fromat.NCDHW|30|NCDHW格式。对于3维图像就需要使用带D（Depth）维度的格式。对应的AscendCL数据格式为ACL_FORMAT_NCDHW。|
    |torch_npu.Fromat.NDC1HWC0|32|6维数据格式。相比于NC1HWC0，仅多了D（Depth）维度。对应的AscendCL数据格式为ACL_FORMAT_NDC1HWC0。|
    |torch_npu.Fromat.FRACTAL_Z_3D|33|3D卷积权重格式，例如Conv3D/MaxPool3D/AvgPool3D这些算子均需以这种格式来表达。对应的AscendCL数据格式为ACL_FRACTAL_Z_3D。|
    |torch_npu.Fromat.NC|35|2维数据格式。对应的AscendCL数据格式为ACL_FORMAT_NC。|
    |torch_npu.Fromat.NCL|47|3维数据格式。对应的AscendCL数据格式为ACL_FORMAT_NCL。|
    
    >**说明：**<br>
    数据排布格式具体可参考《CANN Ascend C算子开发指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html">数据排布格式”</a>章节。

## 返回值说明
`Tensor`

返回修改后的新张量。

## 调用示例

- 整数值调用示例：

    ```python
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> torch_npu.get_npu_format(x)
    0
    >>> x1 = torch_npu.npu_format_cast(x, 2)
    >>> torch_npu.get_npu_format(x1)
    2
    ```

- Format类型调用示例：

    ```python
    >>> x2 = torch_npu.npu_format_cast(x, torch_npu.Format.NHWC)
    >>> torch_npu.get_npu_format(x2)
    1
    ```

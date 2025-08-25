# （beta）torch_npu.npu_format_cast_

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

原地修改`self`张量格式，与`src`格式保持一致。src，即source tensor，源张量。

## 函数原型

```
torch_npu.npu_format_cast_(self, src) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：输入张量。
- **src**（`Tensor`，`Int`）：目标格式。可输入整数或Format枚举类型，Format枚举类型会被自动转换为对应格式的整数值。例如输入2，即代表ACL_FORMAT_ND格式；也可以直接输入Format枚举类型，如`Format.ND`。数据格式具体也可参考《CANN AscendCL应用软件开发指南 (Python)》中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0914.html">aclFormat</a>”章节。数据排布格式具体可参考《CANN Ascend C算子开发指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html">数据排布格式”</a>章节。

    |Format枚举类型|说明|
    | ------|:------: |
    |Fromat.UNDEFINED = -1|未知数据类型。|
    |Fromat.NCHW = 0|NCHW格式。|
    |Fromat.NHWC = 1|NHWC格式。|
    |Fromat.ND = 2|表示支持任意格式，仅有Square、Tanh等这些单输入对自身处理的算子外，其它需要慎用。|
    |Fromat.NC1HWC0 = 3|5维数据格式。其中，C0与微架构强相关，该值等于cube单元的size，例如16；C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要填充到C0。|
    |Fromat.FRACTAL_Z = 4|卷积的权重的格式。|
    |Fromat.NC1HWC0_C04 = 12|5维数据格式。其中，C0固定为4，C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要padding到C0。当前版本不支持。|
    |Fromat.HWCN = 16|HWCN格式。|
    |Fromat.NDHWC = 27|NDHWC格式。对于3维图像就需要使用带D（Depth）维度的格式。|
    |Fromat.FRACTAL_NZ = 29|内部格式，用户目前无需使用。|
    |Fromat.NCDHW = 30|NCDHW格式。对于3维图像就需要使用带D（Depth）维度的格式。|
    |Fromat.NDC1HWC0 = 32|6维数据格式。相比于NC1HWC0，仅多了D（Depth）维度。|
    |Fromat.FRACTAL_Z_3D = 33|3D卷积权重格式，例如Conv3D/MaxPool3D/AvgPool3D这些算子均需以这种格式来表达。|
    |Fromat.NC = 35|2维数据格式。|
    |Fromat.NCL = 47|3维数据格式。|

## 调用示例

```python
>>> x = torch.rand(2, 3, 4, 5).npu()
>>> torch_npu.get_npu_format(x)
0
>>> torch_npu.get_npu_format(torch_npu.npu_format_cast_(x, 2))
2
```


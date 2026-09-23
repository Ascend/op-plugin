# （beta）torch_npu.contrib.function.dropout_with_byte_mask

> [!NOTICE]
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

应用NPU兼容的dropout_with_byte_mask操作，仅支持NPU设备。此方法生成无状态随机uint8掩码，并根据该掩码执行dropout。

## 函数原型

```python
torch_npu.contrib.function.dropout_with_byte_mask(input1, p=0.5, training=True, inplace=False)
```

## 参数说明

- **input1** (`Tensor`)：必选参数，输入张量。
- **p** (`float`)：可选参数，dropout概率，默认值为0.5。
- **training** (`bool`)：可选参数，是否启动dropout，当设置为True时启动，False时不启动。默认值为True。
- **inplace** (`bool`)：可选参数，是否原地生效，当设置为True时将原地修改入参包含的值。默认值为False。

## 约束说明

仅在32核设备场景下性能提升。

## 使用示例

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_functional as F
input = torch.randn(4,4).npu()
input = torch_npu.npu_format_cast(input, 2)
output = F.dropout_with_byte_mask(input, p=0.2, training=True)
print(output)
```

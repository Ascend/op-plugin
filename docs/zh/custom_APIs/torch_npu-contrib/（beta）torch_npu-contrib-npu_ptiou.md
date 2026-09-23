# （beta）torch_npu.contrib.npu_ptiou

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

提供NPU版本的PTIoU计算操作。计算时不会为重叠区域添加极小值。

## 函数原型

```python
torch_npu.contrib.npu_ptiou(boxes1, boxes2, mode="ptiou", is_normalized=False, normalized_scale=100.)
```

## 参数说明

- **boxes1**（`Tensor`）：shape为(n, 4)的预测检测框。
- **boxes2**（`Tensor`）：shape为(m, 4)的Anchor框。
- **is_normalized**（`bool`）：坐标值是否已经标准化。默认值为False。
- **normalized_scale**（`float`）：设置恢复坐标的标准化比例，默认值为100。

## 约束说明

该函数常用于bbox和anchor匹配。当前无对应的后向运算符，因此不能用于IOU_Loss。此外，计算公式中分母加上了0.001以避免除以0，若输入框为归一化数据，0.001的分量会太重，建议放大输入值，避免0.001影响过大。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> box1 = torch.randint(0, 256, size=(32, 4)).npu()
>>> box2 = torch.randint(0, 256, size=(16, 4)).npu()
>>> iou = torch_npu.contrib.npu_ptiou(box1, box2)
```

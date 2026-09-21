# （beta）torch_npu.contrib.npu_iou

> [!NOTICE]
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

提供NPU版本的IoU计算操作。计算时会为重叠区域添加极小值，避免除零问题。

## 函数原型

```python
torch_npu.contrib.npu_iou(boxes1, boxes2, mode="ptiou", is_normalized=False, normalized_scale=100.)
```

## 参数说明

- **boxes1**（`Tensor`）：shape为(n, 4)的预测检测框。
- **boxes2**（`Tensor`）：shape为(m, 4)的预测检测框。
- **mode**（`str`）：选择IoU的计算方式，取值为"iou"、"ptiou"。"iou"=（重叠面积+0.001）/（并集面积+0.001），"ptiou"=重叠面积/（并集面积+0.001）。默认值为"ptiou"。
- **is_normalized**（`bool`）：坐标值是否已经标准化。默认为False。
- **normalized_scale**（`float`）：设置恢复坐标的标准化比例，默认100。

## 约束说明

该函数常用于bbox和anchor匹配。到目前为止，这个函数还没有对应的后向运算符，所以不能用在IoU_Loss中。由于计算公式中分母加上了0.001以避免除以0，当输入框是归一化数据时，0.001的占比太大。此时需要放大输入值，避免0.001影响过大。

## 调用示例

```python
>>> import torch
>>> import torch_npu.contrib
>>> box1 = torch.randint(0, 256, size=(32, 4)).npu()
>>> box2 = torch.randint(0, 256, size=(16, 4)).npu()
>>> iou = torch_npu.contrib.npu_iou(box1, box2)
```

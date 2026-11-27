# （beta）torch_npu.contrib.npu_iou

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

计算两组检测框之间的交并比。

## 函数原型

```python
torch_npu.contrib.npu_iou(boxes1, boxes2, mode="ptiou", is_normalized=False, normalized_scale=100.)
```

## 参数说明

- **boxes1**（`Tensor`）：shape为(n, 4)的预测检测框。
- **boxes2**（`Tensor`）：shape为(m, 4)的预测检测框。
- **mode**（`str`）：选择IoU的计算方式，取值为"iou"、"ptiou"。`"iou"`调用`torch_npu.npu_iou`，`"ptiou"`调用`torch_npu.npu_ptiou`。默认值为`"ptiou"`。
- **is_normalized**（`bool`）：坐标值是否已经归一化。默认为False。
- **normalized_scale**（`float`）：设置恢复坐标的标准化比例，坐标恢复方式为归一化坐标乘以该比例，取值需大于0，需与is_normalized配合使用，默认为100。

## 约束说明

该函数常用于bbox和anchor匹配。到目前为止，这个函数还没有对应的后向运算符，所以不能用在IoU_Loss中。对归一化的坐标，可设置`is_normalized=True`，以指定的`normalized_scale`放大坐标。具体计算由所选算子及其设备实现决定。

## 调用示例

示例计算32个预测框和16个候选框之间的两两重叠分数。

```python
import torch
import torch_npu.contrib
box1 = torch.randint(0, 256, size=(32, 4)).npu()
box2 = torch.randint(0, 256, size=(16, 4)).npu()
iou = torch_npu.contrib.npu_iou(box1, box2)
```

结果包含32×16个配对分数。

# （beta）torch_npu.contrib.npu_giou

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

提供NPU版本的GIoU计算接口。

## 函数原型

```python
torch_npu.contrib.npu_giou(boxes1, boxes2, is_permuted=True)
```

## 参数说明

- **boxes1**（`Tensor`）：格式为xywh、shape为(n, 4)的预测检测框。
- **boxes2**（`Tensor`）：相应的gt检测框，shape为(n, 4)。
- **is_permuted**（`bool`）：坐标值是否已经标准化。默认为True。

## 调用示例

```python
>>> import torch, torch_npu
>>> box1 = torch.randn(32, 4).npu()
>>> box1.requires_grad = True      
>>> box2 = torch.randn(32, 4).npu()
>>> box2.requires_grad = True
>>> iou = torch_npu.contrib.npu_giou(box1, box2)
>>> l = iou.sum()
>>> l.backward()
```

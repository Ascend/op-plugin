# （beta）torch_npu.contrib.npu_giou

>**须知：**<br>
>该接口计划废弃，可以使用torch_npu.npu_giou接口进行替换。

## 函数原型

```
torch_npu.contrib.npu_giou(boxes1, boxes2, is_permuted=True)
```

## 功能说明

提供NPU版本的GIoU计算接口。

## 参数说明

- **boxes1** (`Tensor`) - 格式为xywh、shape为(4, n)的预测检测框。
- **boxes2** (`Tensor`) - 相应的gt检测框，shape为(4, n)。
- **is_permuted** (`bool`) - 坐标值是否已经标准化。默认为True。

## 约束说明

仅trans=True（仅支持xywh，不支持xyxy），is_cross=False（仅支持a.shape==b.shape的场景，不支持((n,4),(m,4))的场景）。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> box1 = torch.randn(32, 4).npu()
>>> box1.requires_grad = True      
>>> box2 = torch.randn(32, 4).npu()
>>> box2.requires_grad = True
>>> iou = torch_npu.contrib.npu_giou(box1, box2)
>>> l = iou.sum()
>>> l.backward()
```


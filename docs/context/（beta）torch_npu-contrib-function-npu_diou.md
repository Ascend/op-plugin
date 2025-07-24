# （beta）torch_npu.contrib.function.npu_diou

>**须知：**<br>
>该接口计划废弃，可以使用torch_npu.npu_diou接口进行替换。

## 函数原型

```
torch_npu.contrib.function.npu_diou(boxes1, boxes2, trans=True, is_cross=False, mode=0)
```

## 功能说明

应用基于NPU的DIoU操作。考虑到目标之间距离，以及距离和范围的重叠率，不同目标或边界需趋于稳定。

## 参数说明

- **boxes1** (`Tensor`) - 格式为xywh、shape为(4, n)的预测检测框。
- **boxes2** (`Tensor`) - 相应的gt检测框，shape为(4, n)。
- **trans** (`Bool`，默认值为True) - 是否有偏移。
- **is_cross** (`Bool`，默认值为False) - box1和box2之间是否有交叉操作。
- **mode** (`Int`，默认值为0) - 选择DIoU的计算方式。

## 输出说明

Tensor - IoU，size为[1,n]。

## 约束说明

到目前为止，DIoU反向只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_diou
>>> box1 = torch.randn(4, 32).npu()
>>> box1.requires_grad = True
>>> box2 = torch.randn(4, 32).npu()
>>> box2.requires_grad = True
>>> diou = npu_diou(box1, box2) 
>>> l = diou.sum()
>>> l.backward()
```


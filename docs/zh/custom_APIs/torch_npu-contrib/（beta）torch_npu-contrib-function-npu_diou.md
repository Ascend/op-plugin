# （beta）torch_npu.contrib.function.npu_diou

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch_npu.npu_diou`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

应用基于NPU的DIoU操作。考虑到目标之间距离，以及距离和范围的重叠率，不同目标或边界需趋于稳定。

## 函数原型

```
torch_npu.contrib.function.npu_diou(boxes1, boxes2, trans=True, is_cross=False, mode=0)
```

## 参数说明

- **boxes1**（`Tensor`）：格式为xywh、shape为(4, n)的预测检测框。
- **boxes2**（`Tensor`）：相应的gt检测框，shape为(4, n)。
- **trans**（`bool`）：是否有偏移，默认值为True。
- **is_cross**（`bool`）：box1和box2之间是否有交叉操作，默认值为False。
- **mode**（`int`）：选择DIoU的计算方式，默认值为0。

## 返回值说明

`Tensor`
 
 IoU，size为[1,n]。

## 约束说明

到目前为止，DIoU反向只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。

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


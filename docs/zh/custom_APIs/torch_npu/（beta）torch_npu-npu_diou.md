# （beta）torch_npu.npu_diou

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

该接口用于实现基于NPU的DIoU（距离交并比）计算。该算法综合考虑预测框与真实框中心点之间的距离以及边界框的重叠率，使边界框回归趋于稳定，提升目标定位精度。

## 函数原型

```python
torch_npu.npu_diou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：格式为xywh，shape为(4, n)的预测检测框。
- **gtboxes** (`Tensor`)：相应的gt检测框，shape为(4, n)。
- **trans** (`bool`)：是否有偏移，默认值为False。
- **is_cross** (`bool`)：box1和box2之间是否有交叉操作，默认值为False。
- **mode** (`int`)：选择DIoU的计算方式。0表示IoU，1表示IoF。默认值为0。

## 返回值说明

`Tensor`

DIoU计算结果。`is_cross`为False时，shape为(1, n)；`is_cross`为True时，shape为(gtboxes.shape[1], self.shape[1])。

## 约束说明

到目前为止，DIoU反向只支持当前版本中的trans==True、is_cross==False、mode==0。如果需要反向传播，确保参数正确。

## 调用示例

```python
    >>> import torch
    >>> import torch_npu
    >>> box1 = torch.randn(4, 32).npu()
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32).npu()
    >>> box2.requires_grad = True
    >>> diou = torch_npu.npu_diou(box1, box2, trans=True)
    >>> l = diou.sum()
    >>> l.backward()
```

# （beta）torch_npu.npu_diou

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

mask操作的结果。

## 约束说明

到目前为止，DIoU反向只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。


## 调用示例

```python
    >>> box1 = torch.randn(4, 32).npu()
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32).npu()
    >>> box2.requires_grad = True
    >>> diou = torch_npu.contrib.function.npu_diou(box1, box2) 
    >>> l = diou.sum()
    >>> l.backward()
```


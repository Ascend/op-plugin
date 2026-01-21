# （beta）torch_npu.npu_ciou
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

应用基于NPU的CIoU操作。在DIoU的基础上增加了penalty term，并propose CIoU。

## 函数原型

```
torch_npu.npu_ciou(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False) -> Tensor
```



## 参数说明

- **boxes1** (`Tensor`)：必选参数，格式为xywh、shape为(4, n)的预测检测框。
- **boxes2** (`Tensor`)：必选参数，相应的gt检测框，shape为(4, n)。
- **trans** (`bool`)：可选参数，是否有偏移。默认值为False。
- **is_cross** (`bool`)：可选参数，box1和box2之间是否有交叉操作。默认值为True。
- **mode** (`int`)：可选参数，选择CIoU的计算方式。0表示IoU，1表示IoF。默认值为0。
- **atan_sub_flag** (`bool`)：可选参数，是否将正向的第二个值传递给反向。默认值为False。

## 返回值说明
`Tensor`

mask操作的结果。

## 约束说明

到目前为止，CIoU向后只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。


## 调用示例

```python
>>> box1 = torch.randn(4, 32).npu()
>>> box1.requires_grad = True
>>> box2 = torch.randn(4, 32).npu()
>>> box2.requires_grad = True
>>> ciou = torch_npu.npu_ciou(box1, box2, trans=True, is_cross=False, mode=0)
>>> l = ciou.sum()
>>> l.backward()
```


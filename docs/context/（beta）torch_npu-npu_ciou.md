# （beta）torch_npu.npu_ciou

## 函数原型

```
torch_npu.npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor
```

## 功能说明

应用基于NPU的CIoU操作。在DIoU的基础上增加了penalty item，并propose CIoU。

## 参数说明

- boxes1 (Tensor)：格式为xywh、shape为(4, n)的预测检测框。
- boxes2 (Tensor)：相应的gt检测框，shape为(4, n)。
- trans (Bool，默认值为False)：是否有偏移。
- is_cross (Bool，默认值为True)：box1和box2之间是否有交叉操作。
- mode (Int，默认值为0)：选择CIoU的计算方式。0表示IoU，1表示IoF。
- atan_sub_flag (Bool，默认值为False)：是否将正向的第二个值传递给反向。

## 输出说明

torch.Tensor：mask操作的结果。

## 约束说明

到目前为止，CIoU向后只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

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


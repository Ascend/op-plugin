# （beta）torch_npu.contrib.npu_ptiou

>**须知：**<br>
>该接口计划废弃，可以使用torch_npu.npu_iou接口进行替换。

## 函数原型

```
torch_npu.contrib.npu_ptiou(boxes1, boxes2, mode="ptiou", is_normalized=False, normalized_scale=100.)
```

## 功能说明

提供NPU版本的PTIoU计算操作。计算时不会为重叠区域添加极小值。

## 参数说明

- boxes1 (Tensor) - shape为(n, 4)的预测检测框。
- boxes2 (Tensor) -  shape为(m, 4)的预测检测框。
- is_normalized(bool) - 坐标值是否已经标准化。默认为False。
- normalized_scale(Float) - 设置恢复坐标的标准化比例，默认100。

## 约束说明

该函数常用于bbox和anchor匹配时。到目前为止，这个函数还没有对应的后向运算符，所以不能用在IOU_Loss中，由于计算公式中分母加上了0.001以避免除以0，当输入框是归一化数据时，0.001的分量会太重。此时需要放大输入值，避免0.001影响过大。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> box1 = torch.randint(0, 256, size=(32, 4)).npu()
>>> box2 = torch.randint(0, 256, size=(16, 4)).npu()
>>> iou = torch_npu.contrib.npu_ptiou(box1, box2)
```


# （beta）torch_npu.contrib.module.LabelSmoothingCrossEntropy

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

使用NPU API计算Label Smoothing Cross Entropy。

## 函数原型

```python
torch_npu.contrib.module.LabelSmoothingCrossEntropy(num_classes=1000, smooth_factor=0.)
```

## 参数说明

**计算参数**

- **num_classes** (`int`)：用于计算非目标类别的平滑权重`smooth_factor / (num_classes - 1)`。取大于1的整数，并与`pred.size(1)`保持一致。默认值为1000。
- **smooth_factor** (`float`)：如果正在使用LabelSmoothing，建议设置为0.1。此参数取值范围为[0, 1]。默认值为0。

**计算输入**

- **pred** (`Tensor`)：必选参数，形状为 `(N, num_classes)`的未归一化分类分数。示例使用`torch.float32`。
- **target** (`Tensor`)：必选参数，形状为 `(N,)`的类别索引，与`pred`在同一设备。索引取整数，范围为 `[0, num_classes - 1]`。

## 返回值说明

`Tensor`

交叉熵计算结果。

## 调用示例

示例为2个样本分别提供10个分类分数与真实类别索引。构造时设置10类和0.1的标签平滑系数，并对损失执行反向传播。

```python
import torch, torch_npu
from torch_npu.contrib.module import LabelSmoothingCrossEntropy
pred = torch.randn(2, 10).npu()
target = torch.randint(0, 10, size=(2,)).npu()
pred.requires_grad = True
m = LabelSmoothingCrossEntropy(10, smooth_factor=0.1)
npu_output = m(pred, target)
npu_output.backward()
print(npu_output)
```

输出是一个float32标量张量，`backward()`将梯度写入`pred.grad`。

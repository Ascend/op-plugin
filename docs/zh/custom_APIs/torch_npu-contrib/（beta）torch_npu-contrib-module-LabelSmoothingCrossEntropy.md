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

- **num_classes** (`float`)：用于onehot的class数量。
- **smooth_factor** (`float`)：如果正在使用LabelSmoothing，建议设置为0.1。此参数取值范围为[0, 1]。默认值为0。

**计算输入**

- **pred**(`Tensor`)：模型预测结果。
- **target**(`Tensor`)：真实标签。

## 返回值说明

`Tensor`

交叉熵计算结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import LabelSmoothingCrossEntropy
>>> pred = torch.randn(2, 10).npu()
>>> target = torch.randint(0, 10, size=(2,)).npu()
>>> pred.requires_grad = True
>>> m = LabelSmoothingCrossEntropy(10)
>>> npu_output = m(pred, target)
>>> npu_output.backward()
>>> print(npu_output)
tensor(1.9443, device='npu:0', grad_fn=<MeanBackward1>)
```

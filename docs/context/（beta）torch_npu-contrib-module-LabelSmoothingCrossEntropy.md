# （beta）torch_npu.contrib.module.LabelSmoothingCrossEntropy

## 函数原型

```
torch_npu.contrib.module.LabelSmoothingCrossEntropy(nn.Module)
```

## 功能说明

使用NPU API进行LabelSmoothing Cross Entropy。

## 参数说明

- smooth_factor (Float，默认值为0) -如果正在使用LabelSmoothing，请改为0.1([0, 1])。
- num_classes (Float) - 用于onehot的class数量。

## 输出说明

Float - shape为(k, 5)和(k, 1)的张量。标签以0为基础。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import LabelSmoothingCrossEntropy
>>> x = torch.randn(2, 10).npu()
>>> y = torch.randint(0, 10, size=(2,)).npu()
>>> x.requires_grad = True
>>> m = LabelSmoothingCrossEntropy(10)
>>> npu_output = m(x, y)
>>> npu_output.backward()
```


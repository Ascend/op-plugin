# （beta）torch_npu.contrib.module.LabelSmoothingCrossEntropy
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

使用NPU API进行LabelSmoothing Cross Entropy。

## 函数原型

```
torch_npu.contrib.module.LabelSmoothingCrossEntropy(num_classes=1000, smooth_factor=0.)
```

## 参数说明
**计算参数**

- **num_classes** (`float`)：用于onehot的class数量。
- **smooth_factor** (`float`)：如果正在使用LabelSmoothing，请改为0.1([0, 1])。默认值为0。

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
>>> npu_output
tensor(1.9443, device='npu:0', grad_fn=<MeanBackward1>)
```


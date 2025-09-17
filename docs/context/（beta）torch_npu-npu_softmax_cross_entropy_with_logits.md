# （beta）torch_npu.npu_softmax_cross_entropy_with_logits

## 函数原型

```
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
```

## 功能说明

- API功能: 计算softmax的交叉熵损失。

- 计算公式: 
$$
     loss = -\sum_{i=1}^{N}y_i * log(softmax(x_i))
$$
其中，$x_i$对应输入的features，$y_i$对应输入的labels，$N$表示输入特征的长度。

## 参数说明

- features (Tensor): 必选参数，输入特征，大小为`[1, batch_size * num_classes]`的矩阵。
- labels (Tensor): 必选参数，输入标签，shape和数据类型与features保持一致。

## 返回值说明
`Tensor`

表示softmax和cross entropy的交叉熵损失计算结果，公式中loss。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例
```python
>>> import torch, torch_npu
>>> batch_size = 4
>>> num_classes = 12
>>> features = torch.rand(1, batch_size * num_classes).npu()
>>> labels = torch.rand(1, batch_size * num_classes).npu() 
>>> output = torch_npu.npu_softmax_cross_entropy_with_logits(features, labels)
>>> output
tensor([97.9450], device='npu:0')
```

# （beta）torch_npu.npu_softmax_cross_entropy_with_logits

> [!NOTICE]
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- API功能：计算softmax的交叉熵损失。

- 计算公式：

$$
     \mathrm{loss}_i = -\sum_{c=1}^{C} y_{i,c}\log\left(\mathrm{softmax}(x_{i,:})_c\right)
$$
其中 $x$ 为features，$y$ 为labels。对每一行i，softmax沿最后一维的C个类别计算。

## 函数原型

```python
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
```

## 参数说明

- **features** (`Tensor`)：必选参数，输入特征。二维logits的形状为 `[batch_size, num_classes]`。数据格式支持$ND$，支持非连续的Tensor。输入最大支持2维。支持空Tensor。数据类型支持`torch.float`、`torch.float16`、`torch.bfloat16`。

  <!-- npu="910,310p" id5 -->
  - <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：数据类型不支持`torch.bfloat16`。
  <!-- end id5 -->

- **labels** (`Tensor`)：必选参数，输入标签，shape和数据类型与`features`保持一致。每一行给出对应样本的目标类别分布。数据格式支持$ND$，支持非连续的Tensor。输入最大支持2维。支持空Tensor。数据类型支持`torch.float`、`torch.float16`、`torch.bfloat16`。

  <!-- npu="910,310p" id6 -->
  - <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：数据类型不支持`torch.bfloat16`。
  <!-- end id6 -->

## 返回值说明

`Tensor`

对应公式中的$loss$，表示softmax交叉熵损失的计算结果。

## 调用示例

使用4个样本的logits和one-hot标签。返回每个样本的交叉熵损失。

```python
import torch
import torch_npu

features = torch.rand(4, 12, device="npu")
targets = torch.tensor([0, 3, 7, 11])
labels = torch.nn.functional.one_hot(targets, num_classes=12).to(features)
loss = torch_npu.npu_softmax_cross_entropy_with_logits(features, labels)
print(loss.shape)
```

输出：

```text
torch.Size([4])
```

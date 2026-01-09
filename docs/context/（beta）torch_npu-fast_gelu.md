# （beta）torch\_npu.fast\_gelu

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算FastGelu。支持FakeTensor模式。

## 函数原型

```
torch_npu.fast_gelu(self) -> Tensor
```

## 参数说明

**input** (`Tensor`)：对应公式中的$x$。数据格式支持$ND$，支持非连续的Tensor。输入最大支持8维。支持空Tensor。

- <term>Atlas 训练系列产品</term>：数据类型支持`float16`、`float32`。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。
- <term>Atlas 推理系列产品</term>：数据类型仅支持`float16`、`float32`。

## 调用示例

示例一：

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(2).npu()
>>> x
tensor([0.5991, 0.4094], device='npu:0')
>>> torch_npu.fast_gelu(x)
tensor([0.4403, 0.2733], device='npu:0')
```

示例二：

```python
>>> import torch
>>> import torch_npu
# FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2).npu()
...     torch_npu.fast_gelu(x)
>>> FakeTensor(..., device='npu:0', size=(2,))
```


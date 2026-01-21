# （beta）torch_npu.contrib.module.Focus

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

使用NPU亲和写法替换YOLOv5中的原生Focus。

## 函数原型

```
torch_npu.contrib.module.Focus(c1, c2, k=1, s=1, p=None, g=1, act=True)
```


## 参数说明
**计算参数**

- **c1** (`int`)：输入图像中的通道数。
- **c2** (`int`)：卷积产生的通道数。
- **k** (`int`)：卷积核大小，默认值为1。
- **s** (`int`)：卷积步长，默认值为1。
- **p** (`int`)：填充，默认值为None。
- **g** (`int`)：从输入通道到输出通道的分组数，默认值为1。
- **act** (`bool`)：是否使用激活函数，默认值为True。

**计算输入**

- **x**(`Tensor`): 输入张量。

## 返回值说明
`Tensor`

Focus计算结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import Focus
>>> input = torch.randn(4, 8, 300, 40).npu()
>>> input.requires_grad_(True)
>>> fast_focus = Focus(8, 13).npu()
>>> output = fast_focus(input)
>>> output.sum().backward()
>>> output.shape
torch.Size([4, 13, 150, 20])
```


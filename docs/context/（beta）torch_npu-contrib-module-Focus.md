# （beta）torch_npu.contrib.module.Focus

## 函数原型

```
torch_npu.contrib.module.Focus(nn.Module)
```

## 功能说明

使用NPU亲和写法替换YOLOv5中的原生Focus。

## 参数说明

- c1 (Int) - 输入图像中的通道数。
- c2 (Int) - 卷积产生的通道数。
- k(Int，默认值为1) - 卷积核大小。
- s(Int，默认值为1) - 卷积步长。
- p (Int) - 填充。
- g (Int，默认值为1) - 从输入通道到输出通道的分组数。
- act (Bool，默认值为True) - 是否使用激活函数。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import Focus
>>> input = torch.randn(4, 8, 300, 40).npu()
>>> input.requires_grad_(True)
>>> fast_focus = Focus(8, 13).npu()
>>> output = fast_focus(input)
>>> output.sum().backward()
```


# （beta）torch_npu.contrib.module.Mish

>**须知：**<br>
>该接口计划废弃，可以使用torch.nn.Mish接口进行替换。

## 函数原型

```
torch_npu.contrib.module.Mish(nn.Module)
```

## 功能说明

应用基于NPU的Mish操作。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import Mish
>>> m = Mish()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
```


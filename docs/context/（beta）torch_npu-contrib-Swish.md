# （beta）torch_npu.contrib.Swish

>**须知：**<br>
>该接口计划废弃，可以使用torch_npu.contrib.ModulationDeformCon接口进行替换。

## 函数原型

```
torch_npu.contrib.Swish()
```

## 功能说明

应用基于NPU的Sigmoid线性单元（SiLU）函数，按元素方向。SiLU函数也称为Swish函数。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> m = torch_npu.contrib.Swish().npu()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
```


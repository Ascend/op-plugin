# （beta）torch_npu.npu_mish

>**须知：**<br>
>该接口计划废弃，可以使用torch.nn.functional.mish接口进行替换。

## 函数原型

```
torch_npu.npu_mish(self) -> Tensor
```

## 功能说明

按元素计算self的双曲正切。

## 参数说明

self (Tensor) - 数据类型：float16、float32。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(10, 30, 10).npu()
>>> y = torch_npu.npu_mish(x)
>>> y.shape
torch.Size([10, 30, 10])
```


# （beta）torch_npu.one_

>**须知：**<br>
>该接口计划废弃，可以使用torch.fill_或torch.ones_like接口进行替换。

## 函数原型

```
torch_npu.one_(self) -> Tensor
```

## 功能说明

用1填充self张量。

## 参数说明

“self“  (Tensor)：输入张量。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(2, 3).npu()

>>> x
tensor([[0.8517, 0.1428, 0.0839],
        [0.1416, 0.9540, 0.9125]], device='npu:0')

>>> torch_npu.one_(x)
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='npu:0')
```


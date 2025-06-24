# （beta）torch_npu.npu_dtype_cast

>**须知：**<br>
>该接口计划废弃，可以使用torch.to接口进行替换。

## 函数原型

```
torch_npu.npu_dtype_cast(input, dtype) -> Tensor
```

## 功能说明

执行张量数据类型（dtype）转换。支持FakeTensor模式。

## 参数说明

- input (Tensor) - 输入张量。
- dtype (torch.dtype) - 返回张量的目标数据类型。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

示例一：

```python
>>> torch_npu.npu_dtype_cast(torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
```

示例二：

```python
//FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2, dtype=torch.float32).npu()
...     res = torch_npu.npu_dtype_cast(x, torch.float16)
...
>>> res
FakeTensor(..., device='npu:0', size=(2,), dtype=torch.float16)
```


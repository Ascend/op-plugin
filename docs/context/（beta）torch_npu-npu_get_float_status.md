# （beta）torch_npu.npu_get_float_status

## 函数原型

```
torch_npu.npu_get_float_status(self) -> Tensor
```

## 功能说明

获取溢出检测结果。

## 参数说明

self (Tensor) - 数据内存地址张量，数据类型为float32。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(2).npu()
>>> torch_npu.npu_get_float_status(x)
tensor([0, 0, 0, 0, 0, 0, 0, 0], device='npu:0', dtype=torch.int32)
```


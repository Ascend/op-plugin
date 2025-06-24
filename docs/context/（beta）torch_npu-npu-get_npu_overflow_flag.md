# （beta）torch\_npu.npu.get\_npu\_overflow\_flag

## 函数原型

```
torch_npu.npu.get_npu_overflow_flag()
```

## 功能说明

检测npu计算过程中是否有数值溢出。

## 支持的型号

<term>Atlas 训练系列产品</term>

## 调用示例

```python
>>>a = torch.Tensor([65535]).npu().half()
>>>a = a + a
>>>ret = torch_npu.npu.get_npu_overflow_flag()
```


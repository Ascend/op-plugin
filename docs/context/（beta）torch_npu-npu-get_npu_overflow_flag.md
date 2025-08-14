# （beta）torch\_npu.npu.get\_npu\_overflow\_flag
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

检测NPU计算过程中是否有数值溢出。

## 函数原型

```
torch_npu.npu.get_npu_overflow_flag()
```

## 调用示例

```python
>>>a = torch.Tensor([65535]).npu().half()
>>>a = a + a
>>>ret = torch_npu.npu.get_npu_overflow_flag()
```


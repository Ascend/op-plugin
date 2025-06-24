# （beta）torch_npu.npu_confusion_transpose

## 函数原型

```
torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor
```

## 功能说明

混淆reshape和transpose运算。

## 参数说明

- self (Tensor) - 数据类型：float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64。
- perm (ListInt) - self张量的维度排列。
- shape (ListInt) - 输入shape。
- transpose_first (Bool) - 如果值为True，首先执行transpose，否则先执行reshape。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(2, 3, 4, 6).npu()
>>> x.shape
torch.Size([2, 3, 4, 6])
>>> y = torch_npu.npu_confusion_transpose(x, (0, 2, 1, 3), (2, 4, 18), True)
>>> y.shape
torch.Size([2, 4, 18])
>>> y2 = torch_npu.npu_confusion_transpose(x, (0, 2, 1), (2, 12, 6), False)
>>> y2.shape
torch.Size([2, 6, 12])
```


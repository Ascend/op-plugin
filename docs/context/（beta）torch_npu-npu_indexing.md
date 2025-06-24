# （beta）torch_npu.npu_indexing

## 函数原型

```
torch_npu.npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor
```

## 功能说明

使用“begin,end,strides”数组对index结果进行计数。

## 参数说明

- self (Tensor) - 输入张量。
- begin (ListInt) - 待选择的第一个值的index。
- end (ListInt) - 待选择的最后一个值的index。
- strides (ListInt) - index增量。
- begin_mask (Int，默认值为0) - 位掩码（bitmask），其中位“i”为“1”意味着忽略开始值，尽可能使用最大间隔。
- end_mask (Int，默认值为0) - 类似于“begin_mask”。
- ellipsis_mask (Int，默认值为0) - 位掩码，其中位“i”为“1”意味着第“i”个位置实际上是省略号。
- new_axis_mask (Int，默认值为0) - 位掩码，其中位“i”为“1”意味着在第“i”位创建新的1D shape。
- shrink_axis_mask (Int，默认值为0) - 位掩码，其中位“i”意味着第“i”位应缩小维数。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> input = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.int32).to("npu")
>>> input
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]], device='npu:0', dtype=torch.int32)
>>> output = torch_npu.npu_indexing(input, [0, 0], [2, 2], [1, 1])
>>> output
tensor([[1, 2],
        [5, 6]], device='npu:0', dtype=torch.int32)
```


# （beta）torch_npu.npu_broadcast

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.broadcast_to`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

返回self张量的新视图，其单维度扩展，结果连续。张量也可以扩展更多维度，新的维度添加在最前面。

## 函数原型

```
torch_npu.npu_broadcast(self, size) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：输入张量。
- **size**（`List[int]`）：对应扩展尺寸。

## 调用示例

```python
>>> x = torch.tensor([[1],[2],[3]]).npu()
>>> x.shape
torch.Size([3, 1])
>>> torch_npu.npu_broadcast(x, [3,4])
tensor([[1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]], device='npu:0')
```


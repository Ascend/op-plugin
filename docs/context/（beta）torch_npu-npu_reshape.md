# （beta）torch_npu.npu_reshape

>**须知：**<br>
>该接口计划废弃，可以使用`torch.reshape`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

reshape张量。仅更改张量shape，其数据不变。

## 函数原型

```
torch_npu.npu_reshape(self, shape, bool can_refresh=False) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：输入张量。
- **shape**（`List[int]`）：定义输出张量的shape。
- **can_refresh**（`bool`）：是否就地刷新reshape，默认值为False。

## 约束说明

该运算符不能被aclopExecute API直接调用。

## 调用示例

```python
>>> a=torch.rand(2,8).npu()
>>> out=torch_npu.npu_reshape(a,(4,4))
>>> out
tensor([[0.6657, 0.9857, 0.7614, 0.4368],
        [0.3761, 0.4397, 0.8609, 0.5544],
        [0.7002, 0.3063, 0.9279, 0.5085],
        [0.1009, 0.7133, 0.8118, 0.6193]], device='npu:0')
```


# （beta）torch_npu.npu_min

>**须知：**<br>
>该接口计划废弃，可以使用`torch.min`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

使用dim对最小结果进行计算。类似于`torch.min`，优化NPU设备实现。

## 函数原型

```
torch_npu.npu_min(self, dim, keepdim=False) -> (Tensor, Tensor)
```

## 参数说明

- **self**（`Tensor`）：输入张量。
- **dim**（`int`）：待降低维度。
- **keepdim**（`bool`）：输出张量是否保留`dim`。

## 返回值说明

- **values**（`Tensor`）：输入张量中的最小值。
- **indices**（`Tensor`）：输入张量中最小值的index。

## 调用示例

```python
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> input
tensor([[[[-0.9909, -0.2369],
          [-0.9569, -0.6223]],

        [[ 0.1157, -0.3147],
          [-0.7761,  0.1344]]],

        [[[ 1.6292,  0.5953],
          [ 0.6940, -0.6367]],

        [[-1.2335,  0.2131],
          [ 1.0748, -0.7046]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_min(input, 2)
>>> outputs
tensor([[[-0.9909, -0.6223],
        [-0.7761, -0.3147]],

        [[ 0.6940, -0.6367],
        [-1.2335, -0.7046]]], device='npu:0')
>>> indices
tensor([[[0, 1],
        [1, 0]],

        [[1, 1],
        [0, 1]]], device='npu:0', dtype=torch.int32)
```


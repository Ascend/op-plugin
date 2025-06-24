# （beta）torch_npu.npu_max

## 函数原型

```
torch_npu.npu_max(self, dim, keepdim=False) -> (Tensor, Tensor)
```

## 功能说明

使用dim对最大结果进行计算。类似于torch.max，优化NPU设备实现。

## 参数说明

- self (Tensor) - 输入张量。
- dim (Int) - 待降低维度。
- keepdim (Bool，默认值为False) - 输出张量是否保留dim。

## 输出说明

- values (Tensor) - 输入张量中的最大值。
- indices (Tensor) - 输入张量中最大值的index。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> input
tensor([[[[-1.8135,  0.2078],
          [-0.6678,  0.7846]],

        [[ 0.6458, -0.0923],
          [-0.2124, -1.9112]]],

        [[[-0.5800, -0.4979], 
         [ 0.2580,  1.1335]],

          [[ 0.6669,  0.1876],
          [ 0.1160, -0.1061]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_max(input, 2)
>>> outputs
tensor([[[-0.6678,  0.7846],
        [ 0.6458, -0.0923]],

        [[ 0.2580,  1.1335],
        [ 0.6669,  0.1876]]], device='npu:0')
>>> indices
tensor([[[1, 1],
        [0, 0]],

        [[1, 1],
        [0, 0]]], device='npu:0', dtype=torch.int32)
```


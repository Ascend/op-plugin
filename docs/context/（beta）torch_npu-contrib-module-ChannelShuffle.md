# （beta）torch_npu.contrib.module.ChannelShuffle

## 函数原型

```
torch_npu.contrib.module.ChannelShuffle(nn.Module)
```

## 功能说明

应用NPU兼容的通道shuffle操作。为避免NPU上效率不高的连续操作，我们用相同语义重写替换原始操作。以下两个不连续操作已被替换：transpose和chunk。

## 参数说明

- Input (Tensor) - 输入张量。 (N, C_\{in\}, L_\{in\}), (N, C_\{in\}, L_\{in\})。
- in_channels (Int) - 输入张量中的通道总数。
- groups (Int，默认值为2) - shuffle组数。
- split_shuffle (Bool，默认值为True) - shuffle后是否执行chunk操作。默认值：True。

## 输出说明

Output (Tensor) - 输出张量(N, C_\{out\}, L_\{out\})。

## 约束说明

只实现了groups=2，请自行修改其他groups场景。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import ChannelShuffle
>>> x1 = torch.randn(2,32,7,7).npu()
>>> x2 = torch.randn(2,32,7,7).npu()
>>> m = ChannelShuffle(64, split_shuffle=True)
>>> output = m(x1, x2)
```


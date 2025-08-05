# （beta）torch_npu.npu_layer_norm_eval

>**须知：**<br>
>该接口计划废弃，可以使用torch.nn.functional.layer_norm接口进行替换。

## 函数原型

```
torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor
```

## 功能说明

对层归一化结果进行计算。与torch.nn.functional.layer_norm相同，优化NPU设备实现。

## 参数说明

- input (Tensor) - 输入张量。
- normalized_shape (ListInt) - size为预期输入的shape。
- weight (Tensor，可选，默认值为None) - gamma张量。
- bias (Tensor，可选默认值为None) - beta张量。
- eps (Float，默认值为1e-5) - 为保证数值稳定性添加到分母中的ε值。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> input = torch.rand((6, 4), dtype=torch.float32).npu()
>>> input
tensor([[0.1863, 0.3755, 0.1115, 0.7308],
        [0.6004, 0.6832, 0.8951, 0.2087],
        [0.8548, 0.0176, 0.8498, 0.3703],
        [0.5609, 0.0114, 0.5021, 0.1242],
        [0.3966, 0.3022, 0.2323, 0.3914],
        [0.1554, 0.0149, 0.1718, 0.4972]], device='npu:0')
>>> normalized_shape = input.size()[1:]
>>> normalized_shape
torch.Size([4])
>>> weight = torch.Tensor(*normalized_shape).npu()
>>> weight
tensor([        nan,  6.1223e-41, -8.3159e-20,  9.1834e-41], device='npu:0')
>>> bias = torch.Tensor(*normalized_shape).npu()
>>> bias
tensor([5.6033e-39, 6.1224e-41, 6.1757e-39, 6.1224e-41], device='npu:0')
>>> output = torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
>>> output
tensor([[        nan,  6.7474e-41,  8.3182e-20,  2.0687e-40],
        [        nan,  8.2494e-41, -9.9784e-20, -8.2186e-41],
        [        nan, -2.6695e-41, -7.7173e-20,  2.1353e-41],
        [        nan, -1.3497e-41, -7.1281e-20, -6.9827e-42],
        [        nan,  3.5663e-41,  1.2002e-19,  1.4314e-40],
        [        nan, -6.2792e-42,  1.7902e-20,  2.1050e-40]], device='npu:0')
```


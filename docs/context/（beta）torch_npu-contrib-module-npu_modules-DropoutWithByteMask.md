# （beta）torch_npu.contrib.module.npu_modules.DropoutWithByteMask

## 函数原型

```
torch_npu.contrib.module.npu_modules.DropoutWithByteMask(Module)
```

## 功能说明

应用NPU兼容的DropoutWithByteMask操作。

## 参数说明

- Input (Tensor) - 输入张量，可为任何shape。
- p (Float，默认值为0.5) - 元素归零的概率。
- inplace (Bool，默认值为False) - 如果设置为True，原地执行此操作。

## 输出说明

Output (Tensor) - 输出张量与输入张量的shape相同。

## 约束说明

- Maxseed是一个与底层算子强相关的超参数。请检查算子包的dropoutv2.py文件中的MAX(2 \*\* 31 - 1 / 2 \*\* 10 - 1)以匹配设置。默认情况下，它与PyTorch和算子包匹配。
- 仅支持NPU设备。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module.npu_modules import DropoutWithByteMask
>>> m = DropoutWithByteMask(p=0.5)
>>> input = torch.randn(16, 16).npu()
>>> output = m(input)
```


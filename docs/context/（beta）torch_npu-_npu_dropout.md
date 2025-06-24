# （beta）torch\_npu.\_npu\_dropout

## 函数原型

```
torch_npu._npu_dropout(self, p) -> (Tensor, Tensor)
```

## 功能说明

不使用种子（seed）进行dropout结果计数。

## 参数说明

-   self \(Tensor\) - 输入张量。
-   p \(Float\) - 丢弃概率。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> input = torch.tensor([1.,2.,3.,4.]).npu()
>>> input
tensor([1., 2., 3., 4.], device='npu:0')
>>> prob = 0.3
>>> output, mask = torch_npu._npu_dropout(input, prob)
>>> output
tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
>>> mask
tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
      253, 255], device='npu:0', dtype=torch.uint8)
```


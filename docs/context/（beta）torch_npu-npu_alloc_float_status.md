# （beta）torch_npu.npu_alloc_float_status

## 函数原型

```
torch_npu.npu_alloc_float_status(self) -> Tensor
```

## 功能说明

为溢出检测模式申请tensor作为入参。

## 参数说明

self (Tensor) - 任何张量。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> input    = torch.randn([1,2,3]).npu()
>>> output = torch_npu.npu_alloc_float_status(input)
>>> input
tensor([[[ 2.2324,  0.2478, -0.1056],
        [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
>>> output
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
```


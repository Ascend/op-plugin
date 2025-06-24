# （beta）torch_npu.npu_linear

## 函数原型

```
torch_npu.npu_linear(input, weight, bias=None) -> Tensor
```

## 功能说明

将矩阵“a”乘以矩阵“b”，生成“a\*b”。

## 参数说明

- input (Tensor) - 2D矩阵张量。数据类型：float32、float16、int32、int8。格式：[ND, NHWC, FRACTAL_NZ]。
- weight (Tensor) - 2D矩阵张量。数据类型：float32、float16、int32、int8。格式：[ND, NHWC, FRACTAL_NZ]。
- bias (Tensor，可选，默认值为None) - 1D张量。数据类型：float32、float16、int32。格式：[ND, NHWC]。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x=torch.rand(2,16).npu()
>>> w=torch.rand(4,16).npu()
>>> b=torch.rand(4).npu()
>>> output = torch_npu.npu_linear(x, w, b)
>>> output
tensor([[3.6335, 4.3713, 2.4440, 2.0081],
        [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
```


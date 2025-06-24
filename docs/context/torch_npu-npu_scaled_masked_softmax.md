# torch_npu.npu_scaled_masked_softmax

## 函数原型

```
torch_npu.npu_scaled_masked_softmax(x, mask, scale=1.0, fixed_triu_mask=False) -> Tensor
```

## 功能说明

计算输入张量`x`缩放并按照`mask`遮蔽后的`Softmax`结果。

## 参数说明

- **x**（`Tensor`）- 输入的logits。支持数据类型：`float16`、`float32`、`bfloat16`。支持格式：$[ND，FRACTAL\_NZ]$。
- **mask**（`Tensor`）- 输入的掩码。支持数据类型：`bool`。支持格式：$[ND，FRACTAL\_NZ]$。
- **scale**（`float`），默认值为`1.0`,`x`的缩放系数。
- **fixed_triu_mask**（`bool`，默认值为`False`）- 为预留参数，功能未完成，当前只支持`False`。该功能完成后可支持自动生成上三角`bool`掩码。

## 返回值
`Tensor`

一个`Tensor`类型的输出，输入`x`经过`mask`后在最后一维的`Softmax`结果，输出shape与`x`一致。支持数据类型：`float16`、`float32`、`bfloat16`。支持格式：$[ND，FRACTAL\_NZ]$。

## 约束说明

- 当前输入`x`的shape，只支持转为$[NCHW]$格式后，H和W轴长度大于等于32、小于等于4096、且能被32整除的场景。
- 输入`mask`的shape，必须能被broadcast成`x`的shape。

## 支持的型号

- <term>Atlas 训练系列产品</term> 
- <term>Atlas A2 训练系列产品</term> 
- <term>Atlas A3 训练系列产品</term> 
- <term>Atlas 推理系列产品</term> 

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> 
>>> shape = [4, 4, 2048, 2048]
>>> x = torch.rand(shape).npu()
>>> mask = torch.zeros_like(x).bool()
>>> scale = 1.0
>>> fixed_triu_mask = False
>>> 
>>> output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
>>> output.shape
torch.size([4, 4, 2048, 2048])
```


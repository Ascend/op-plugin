# （beta）torch_npu.npu_layer_norm_eval

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.layer_norm`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term> | √   |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term> | √   |

## 功能说明

- API功能：对层归一化（Layer Normalization）结果进行计算，语义与`torch.nn.functional.layer_norm`一致，由NPU侧实现加速。
- 计算公式：设 $x$ 为`input`末尾若干维组成的切片向量（共 $N$ 个元素），$\gamma_i$ 和 $\beta_i$ 分别代表`weight`和`bias`，$E[x]$ 为 $x$ 的均值，$Var[x]$ 为总体方差（分母为 $N$），$\varepsilon$ 对应`eps`，则
    $$
    \begin{cases}
    \hat{x}_i = \dfrac{x_i - E[x]}{\sqrt{Var[x] + \varepsilon}} \\[6pt]
    y_i = \gamma_i \cdot \hat{x}_i + \beta_i
    \end{cases}
    $$
    其中
    $$
    E[x] = \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad Var[x] = \frac{1}{N}\sum_{i=1}^{N}\left(x_i - E[x]\right)^2
    $$
    `normalized_shape`长度为 $r$ 时，$N = \prod_{j=1}^{r}\text{normalized\_shape}[j]$，每组切片对应`input`前若干维的一组固定下标，各切片独立计算。

## 函数原型

```python
torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数，输入张量，数据格式支持$ND$，支持非连续的Tensor。
- **normalized_shape** (`List[int]`)：必选参数，归一化维度的形状，须与`input`最后若干维逐维一致。
- **weight** (`Tensor`)：可选参数，缩放参数（$\gamma_i$），shape须与`normalized_shape`一致，默认值为`None`。
- **bias** (`Tensor`)：可选参数，偏移参数（$\beta_i$），shape须与`normalized_shape`一致，默认值为`None`。
- **eps** (`float`)：可选参数，为保证数值稳定性添加到分母中的ε值，默认值为`1e-5`。

## 返回值说明

`Tensor`

层归一化计算结果，shape和数据类型与`input`一致，数据格式支持$ND$，支持非连续的Tensor。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 不传`weight`和`bias`时，其参数默认值为`None`；计算时分别按与`normalized_shape`同shape的全1（`weight`）和全0（`bias`）处理。

## 调用示例

```python
import torch
import torch_npu

input = torch.tensor(
    [[0.1863, 0.3755, 0.1115, 0.7308],
     [0.6004, 0.6832, 0.8951, 0.2087],
     [0.8548, 0.0176, 0.8498, 0.3703],
     [0.5609, 0.0114, 0.5021, 0.1242],
     [0.3966, 0.3022, 0.2323, 0.3914],
     [0.1554, 0.0149, 0.1718, 0.4972]],
    dtype=torch.float32,
).npu()
normalized_shape = input.size()[1:]
weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
bias = torch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
output = torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
# 执行上述代码的输出类似如下
# tensor([[-0.6879,  0.1022, -1.0002,  1.5859],
#         [ 0.0143,  0.3474,  1.1999, -1.5616],
#         [ 0.9422, -1.4361,  0.9280, -0.4341],
#         [ 1.1061, -1.2204,  0.8571, -0.7428],
#         [ 0.9685, -0.4173, -1.4434,  0.8922],
#         [-0.3078, -1.1025, -0.2151,  1.6255]], device='npu:0')
```

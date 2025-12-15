# torch_npu.npu_group_norm_swish

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |


## 功能说明

- API功能：计算输入`input`的组归一化结果`y`，均值`mean`，标准差的倒数`rstd`，以及swish的输出。
- 计算公式：
  - GroupNorm: 公式中的$x$代表`input`， $E[x] = \bar{x}$ 代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^{n} (x_i - E[x])^2$ 代表$x$的方差，$\gamma$代表`weight`，$\beta$代表`bias`，则公式如下：
  $$
  \begin{cases}
  y & = \frac{x - E[x]}{\sqrt{Var[x]} + eps} * \gamma + \beta \\ 
  mean & = E[x] \\ 
  rstd & = \frac{1}{\sqrt{Var[x]} + eps}
  \end{cases}
  $$

  - swish：swish计算公式的$x$为GroupNorm公式得到的$y$。
  $$
  y = \frac{x}{1 + e^{-scale \cdot x}}
  $$
  

> [!NOTE]  
> 需要计算反向梯度场景时，若需要输出结果排除随机性，则需要[设置确定性计算开关](determin_API_list.md)。

## 函数原型

```
torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

-   **input**(`Tensor`)：必选参数，表示需要进行组归一化的数据，支持2-8D张量，数据类型支持`float16`，`float32`，`bfloat16`。
-   **num_groups**(`int`)：必选参数，表示将`input`的第1维分为`num_groups`组，`input`的第1维必须能被`num_groups`整除。
-   **weight**(`Tensor`)：必选参数，表示权重，支持1D张量，并且第0维大小与`input`的第1维相同；数据类型支持`float16`，`float32`，`bfloat16`，并且需要与`input`一致。
-   **bias**(`Tensor`)：必选参数，表示偏置，支持1D张量，并且第0维大小与`input`的第1维相同；数据类型支持`float16`，`float32`，`bfloat16`，并且需要与`input`一致。
-   **eps**(`float`)：可选参数，计算组归一化时加到分母上的值，以保证数值的稳定性。默认值为1e-5。
-   **swish_scale**(`float`)：可选参数，用于进行swish计算的值。默认值为1.0。

## 返回值说明

**y**(`Tensor`)：表示组归一化和swish计算的结果。

**mean**(`Tensor`)：表示分组后的均值。

**rstd**(`Tensor`)：表示分组后的标准差的倒数。

## 约束说明

需要计算反向梯度场景时，`input`的第1维除以`num_groups`的结果不能超过4000，`input`、`weight`、`bias`参数不支持含有-inf、inf或nan值。

## 调用示例

```python
import torch
import torch_npu
 
input = torch.randn(3, 4, 6, dtype=torch.float32).npu()
weight = torch.randn(input.size(1), dtype=torch.float32).npu()
bias = torch.randn(input.size(1), dtype=torch.float32).npu()
num_groups = input.size(1)
eps = 1e-5
swish_scale = 1.0
  out, mean, rstd = torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)
```


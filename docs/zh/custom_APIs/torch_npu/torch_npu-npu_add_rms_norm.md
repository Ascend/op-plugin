# torch_npu.npu_add_rms_norm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |


## 功能说明

- API功能：将Add计算与RMSNorm归一化融合，常用于大模型中将残差连接后的张量进行归一化处理。
- 计算公式：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## 函数原型
```
torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=1e-06) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x1**(`Tensor`)：必选参数，表示用于Add计算的第一个输入，对应公式中的$x1$。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型支持`float32`、`float16`、`bfloat16`。支持1-8维张量。
- **x2**(`Tensor`)：必选参数，表示用于Add计算的第二个输入，对应公式中的$x2$。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型支持`float32`、`float16`、`bfloat16`。支持1-8维张量。
- **gamma**(`Tensor`)：必选参数，表示RmsNorm的缩放因子（权重）。对应公式中的$gamma$。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型与`x1`的数据类型保持一致，shape需要与`x1`后几维保持一致，后几维为`x1`需要归一化处理的维度。
- **epsilon**(`float`)：可选参数，表示添加到分母中的值，以确保数值稳定，对应公式中的$epsilon$。默认值为`1e-6`。

## 返回值说明

- **yOut**(`Tensor`)：对应公式中的$RmsNorm(x)$，表示最后的输出。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型和shape与输入`x1`的数据类型和shape保持一致。

- **rstdOut**(`Tensor`)：对应公式中$Rms(x)$的倒数，表示归一化后标准差的倒数。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型支持`float32`。shape与`x1`前几维保持一致，前几维表示不需要归一化处理的维度。`rstdOut`shape与`x1`shape，`gamma`shape关系举例：
  - 若`x1`shape:(2，3，4，8)，`gamma`shape:(8)，`rstdOut`shape(2，3，4，1)；
  - 若`x1`shape:(2，3，4，8)，`gamma`shape:(4，8)，`rstdOut`shape(2，3，1，1)。

- **xOut**(`Tensor`)：对应公式中的$x$，表示add计算的结果。数据格式支持$ND$，不支持空Tensor，支持非连续Tensor。数据类型和shape与输入`x1`的数据类型和shape保持一致。

## 约束说明

- 边界值场景说明：
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。
- Atlas 推理系列产品：
  - 参数`x1`、`x2`、`gamma`、`yOut`、`xOut`的数据类型不支持bfloat16。
  - 参数`rstdOut`在当前产品使用场景下无效。

## 调用示例

```python
import torch
import torch_npu

x1 = torch.rand(4, 8, dtype=torch.float16, device='npu') * 100
x2 = torch.rand(4, 8, dtype=torch.float16, device='npu') * 100
gamma = torch.rand(8, dtype=torch.float16, device='npu') * 100
epsilon = 1e-6

y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)

print("y:", y)
print("y.shape:", y.shape)
print("y.dtype:", y.dtype)
print("rstd:", rstd)
print("rstd.dtype:", rstd.dtype)
print("x:", x)
```
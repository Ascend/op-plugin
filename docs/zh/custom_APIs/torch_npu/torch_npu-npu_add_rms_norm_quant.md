# torch_npu.npu_add_rms_norm_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |

## 功能说明

- **API功能**：
    RMSNorm是大模型常用的标准化操作，相比LayerNorm其去掉了减去均值的部分。torch_npu.npu_add_rms_norm_quant算子将RMSNorm前的Add算子以及RMSNorm后的Quantize算子融合起来，减少搬入搬出操作。

- **计算公式**：
  - AddRMSNorm计算过程：

  $$
  x_i={x1}+{x2}
  $$

  $$
  y=\operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - 量化计算过程：
      - 当`div_mode`为True时：

      $$
      y1=round((y/scales1)+zero\_points1)
      $$

      $$
      y2=round((y/scales2)+zero\_points2)
      $$
    - 当`div_mode`为False时：

      $$
      y1=round((y*scales1)+zero\_points1)
      $$

      $$
      y2=round((y*scales2)+zero\_points2)
      $$

## 函数原型

```python
torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, beta=None, scales2=None, zero_points2=None, *, axis=-1, epsilon=1e-06, div_mode=True， dst_type=None) -> (y1, y2, x)
```

## 参数说明

- **x1**（`Tensor`）：必选参数，表示标准化过程中的源数据张量，公式中的$x1$。数据格式支持$ND$，支持非连续的Tensor。shape支持1-8维。
  - <term>Atlas 推理系列产品</term>：数据类型支持`torch.float16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`torch.float16`、`torch.bfloat16`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。

- **x2**（`Tensor`）：必选参数，表示标准化过程中的源数据张量，公式中的$x2$。数据格式支持$ND$，支持非连续的Tensor。shape支持1-8维，shape和数据类型需要与`x1`保持一致。
- **gamma**（`Tensor`）：必选参数，表示标准化过程中的权重张量，公式中的$gamma$。数据格式支持$ND$，支持非连续的Tensor。shape支持1-8维，shape与`x1`需要norm的维度一致，数据类型需要与`x1`保持一致。
- **scales1**（`Tensor`）：必选参数，表示量化过程中得到`y1`的scales张量，公式中的$scales1$。数据格式支持$ND$，支持非连续的Tensor。shape需要与`gamma`保持一致。当参数`div_mode`的值为True时，该参数的值不能为0。
  - <term>Atlas 推理系列产品</term>：数据类型支持`torch.float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`torch.float32`、`torch.bfloat16`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`。

- **zero_points1**（`Tensor`）：可选参数，表示量化过程中得到`y1`的offset张量，公式中的$zero\_points1$。数据格式支持$ND$，支持非连续的Tensor。shape需要与`gamma`保持一致。
  - <term>Atlas 推理系列产品</term>：数据类型支持`torch.int32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`torch.int32`、`torch.bfloat16`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.int32`、`torch.float32`、`torch.float16`、`torch.bfloat16`。

- **beta**（`Tensor`）：可选参数，表示标准化过程中的偏置项，公式中的$beta$。数据格式支持$ND$，支持非连续的Tensor。shape和数据类型需要与`gamma`保持一致。
- **scales2**（`Tensor`）：可选参数，表示量化过程中得到`y2`的scales张量，公式中的$scales2$。数据格式支持$ND$，支持非连续的Tensor。数据类型需要与`scales1`保持一致。shape需要与`gamma`保持一致。当参数`div_mode`的值为True时，该参数的值不能为0。默认值为None。
- **zero_points2**（`Tensor`）：可选参数，表示量化过程中得到`y2`的offset张量，公式中的$zero\_points2$。数据格式支持$ND$，支持非连续的Tensor。数据类型需要与`zero_points1`保持一致。shape需要与`gamma`保持一致。默认值为None。
- **axis**（`int`）：可选参数，表示需要进行量化的elementwise轴，其他的轴做broadcast，指定的轴索引不能超过输入张量的维度数。当前仅支持默认值-1，传其他值均不生效。
- **epsilon**（`float`）：可选参数，公式中的输入$epsilon$，用于防止除0错误，数据类型为`torch.double`。建议传较小的正数，默认值为1e-6。
- **div_mode**（`bool`）：可选参数，公式中决定量化公式是否使用除法的参数，数据类型为`torch.bool`，默认值为True。
- **dst_type**（`int`）：可选参数，指定量化输出`y1`、`y2`的数据类型，默认值为`torch.int8`。传None时当做int8处理。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas 推理系列产品</term>：支持None或torch.int8。
  - <term>Ascend 950PR/Ascend 950DT</term>：支持None或torch.int8、torch_npu.hifloat8、torch.float8_e5m2、torch.float8_e4m3fn。

## 返回值说明

- **y1**（`Tensor`）：表示量化后的输出Tensor，公式中的$y1$。数据格式支持$ND$，支持非连续的Tensor。shape与输入`x1`一致。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持`torch.int8`。
  - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.int8`、`torch_npu.hifloat8`、`torch.float_e5m2`、`torch.float8_e4m3fn`。

- **y2**（`Tensor`）：表示量化后的输出Tensor，公式中的$y2$。数据格式支持$ND$，支持非连续的Tensor。shape和数据类型与输出`y1`一致。
- **x**（`Tensor`）：表示`x1`和`x2`相加的和，公式中的$x$。数据格式支持$ND$，支持非连续的Tensor。数据类型和shape与输入`x1`一致。

## 约束说明

- 该接口仅支持单算子模式调用。
- <term>Atlas 推理系列产品</term>：`x1`、`x2`的最后一维数据个数不能小于32。`gamma`、`beta`、`scales1`、`zero_points1`、`scales2`、`zero_points2`的数据个数不能小于32。

- 边界值场景说明：
  - <term>Atlas 推理系列产品</term>：输入不支持包含inf和nan。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入是inf时，输出为inf。当输入是nan时，输出为nan。

- 维度的边界说明：
  参数`x1`、`x2`、`gamma`、`scales1`、`zero_points1`、`beta`、`scales2`、`zero_points2`、`y1`、`y2`、`x`的shape中每一维大小都不大于int32的最大值2147483647。  

- 各产品型号数据类型支持说明：
  - <term>Ascend 950PR/Ascend 950DT</term>：

    | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | y1 | y2 | x |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | torch.float32 | torch.float32 | torch.float32 | torch.float32 | torch.float32 | torch.float32 | torch.float32 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.float32 |
    | torch.float16 | torch.float16 | torch.float16 | torch.float32 | torch.float32 | torch.int32 | torch.int32 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.float16 |
    | torch.float16 | torch.float16 | torch.float16 | torch.float16 | torch.float16 | torch.float16 | torch.float16 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.float16 |
    | torch.float16 | torch.float16 | torch.float16 | torch.float32 | torch.float32 | torch.float32 | torch.float32 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.float16 |
    | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.float32 | torch.float32 | torch.int32 | torch.int32 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.bfloat16 |
    | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.bfloat16 |
    | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.float32 | torch.float32 | torch.float32 | torch.float32 | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.int8/torch_npu.hifloat8/torch.float8_e5m2/torch.float8_e4m3fn | torch.bfloat16 |

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | beta | y1 | y2 | x |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | torch.float16 | torch.float16 | torch.float16 | torch.float32 | torch.float32 | torch.int32 | torch.int32 | torch.float16 | torch.int8 | torch.int8 | torch.float16 |
    | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.bfloat16 | torch.int8 | torch.int8 | torch.bfloat16 |

  - <term>Atlas 推理系列产品</term>：

    | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | beta | y1 | y2 | x |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | torch.float16 | torch.float16 | torch.float16 | torch.float32 | torch.float32 | torch.int32 | torch.int32 | torch.float16 | torch.int8 | torch.int8 | torch.float16 |
    
## 调用示例

```python
import math

import numpy as np
import torch

import torch_npu

def test_npu_add_rms_norm_quant():
    shape_list = [[[16, ], [16, ]],
                  [[2, 16], [16, ]],
                  [[2, 16], [2, 16]],
                  [[16, 32], [16, 32]],
                  [[16, 32], [32, ]],
                  [[2, 2, 2, 8, 16, 32], [2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 8, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 16, 32], [2, 2, 2, 2, 2, 16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [2, 2, 2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [32, ]]]
    for item in shape_list:
        x_shape = item[0]
        quant_shape = item[1]
        x1 = torch.randn(x_shape, dtype=torch.float16)
        x2 = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        scales1 = torch.randn(quant_shape, dtype=torch.float32)
        zero_points1 = torch.randint(-10, 10, quant_shape, dtype=torch.int32)

        x1_npu = x1.npu()
        x2_npu = x2.npu()
        gamma_npu = gamma.npu()
        beta_npu = beta.npu()
        scales1_npu = scales1.npu()
        zero_points1_npu = zero_points1.npu()

        y1_v1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
        y1_v2, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu, beta_npu) 

if __name__ == "__main__":
    test_npu_add_rms_norm_quant()
```

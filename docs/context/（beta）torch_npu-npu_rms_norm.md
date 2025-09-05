# （beta）torch_npu.npu_rms_norm

# 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 推理系列产品</term>   |     √    |
|  <term>Atlas 训练系列产品</term>   |     √    |


## 功能说明

- API功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。
- 计算公式：

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i
  $$

  $$
  \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## 函数原型

```
torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor) 
```

## 参数说明

- **self**（`Tensor`）：必选参数，表示进行归一化计算的输入。对应计算公式中的`x`。shape支持2-8维度，数据格式支持$ND$。支持非连续Tensor，支持空Tensor。
- **gamma**（`Tensor`）：必选参数，表示进行归一化计算的缩放因子（权重），对应计算公式中的`g`。shape支持2-8维度，数据格式支持$ND$。shape需要满足gamma_shape = self_shape\[n:\], n < x_shape.dims()，通常为`self`的最后一维。支持非连续Tensor，支持空Tensor。
- **epsilon**（`double`）：可选参数，用于防止除0错误，对应计算公式中的`eps`。数据类型为`double`，默认值为1e-6。

## 返回值说明

- **RmsNorm(x)**（`Tensor`）：表示进行归一化后的最终输出，对应计算公式的最终输出`RmsNorm(x)`。数据类型和shape与输入`self`的数据类型和shape一致。支持非连续Tensor，支持空Tensor。
- **rstd**（`Tensor`）：表示归一化后的标准差的倒数，rms_norm的中间结果，对应计算公式中的`Rms(x)`的倒数，用于反向计算。数据类型为`float32`。shape与入参`self`的shape前几维一致，前几维指x的维度减去gamma的维度，表示不需要norm的维度。支持非连续Tensor，支持空Tensor。

## 约束说明

- <term>Atlas 推理系列产品</term>：`self`、`gamma`输入的尾轴长度必须大于等于32 Bytes。
- 各产品支持数据类型及对应关系说明：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    | `self`数据类型 | `gamma`数据类型 |
    | -------- | -------- |
    | `float16` | `float32` |
    | `bfloat16` | `float32` |
    | `float16` | `float16` |
    | `bfloat16` | `bfloat16` |
    | `float32` | `float32`  |
  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：
    | `self`数据类型 | `gamma`数据类型 |
    | -------- | -------- |
    | `float16` | `float16` |
    | `float32` | `float32` |

## 调用示例

```python
>>> x = torch.randn(24, 1, 128).bfloat16().npu()
>>> w = torch.randn(128).bfloat16().npu()
>>> out1 = torch_npu.npu_rms_norm(x, w, epsilon=1e-5)[0]
>>> out1
tensor([[[-0.1875,  0.2383,  0.2334,  ...,  0.8555, -0.0908, -0.3574]],
        [[ 0.0747,  0.4668,  0.1074,  ...,  1.7500,  0.1953, -0.1992]],
        [[-0.0571, -0.4883,  0.5273,  ..., -2.1250, -0.0312,  2.3281]],
        ...,
        [[ 0.0503,  1.9453,  2.6094,  ..., -0.1357,  0.0869, -2.8906]],
        [[ 0.0195,  0.6680, -0.9336,  ..., -0.6641, -0.1904,  0.4336]],
        [[ 0.0972, -1.2344, -1.0078,  ..., -0.5195,  0.3145, -3.7656]]],
       device='npu:0', dtype=torch.bfloat16)
```


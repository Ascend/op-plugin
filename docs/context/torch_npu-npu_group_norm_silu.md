# torch_npu.npu_group_norm_silu

## 功能说明

计算输入`input`的组归一化结果`out`、均值`meanOut`、标准差的倒数`rstdOut`、以及silu的输出。

## 函数原型

```
torch_npu.npu_group_norm_silu(input, weight, bias, group, eps) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input** (`Tensor`)：必选输入，源数据张量，维度需大于一维，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float`、`bfloat16`。

- **weight** (`Tensor`)：可选输入，索引张量，维度为1且元素数量需与输入`input`的第1维度保持相同，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float`、`bfloat16`。

- **bias** (`Tensor`)：可选输入，更新数据张量，维度为1元素数量需与输入`input`的第1维度保持相同，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float`、`bfloat16`。

- **group** (`int`)：必选输入，表示将输入`input`的第1维度分为`group`组。
- **eps** (`float`)：可选参数，数值稳定性而加到分母上的值，若保持精度，则`eps`需大于0。

## 返回值

- **out** (`Tensor`)：数据类型和shape与`input`相同，支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float`、`bfloat16`。

- **meanOut** (`Tensor`)：数据类型与`input`相同，shape为$(N, group)$支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float`、`bfloat16`。

- **rstdOut** (`Tensor`)：数据类型与`input`相同，shape为$(N, group)$。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float`、`bfloat16`。

## 约束说明

- `input`、`weight`、`bias`、`out`、`meanOut`、`rstdOut`数据类型必须支持的范围之内。
- `out`、`meanOut`、`rstdOut`的数据类型与`input`相同；`weight`、`bias`与`input`可以不同。
- `input`第1维度能整除`group`。
- `out`的shape与`input`相同。
- `meanOut`与`rstdOut`的shape为$(N, group)$，其中$N$为`input`第0维度值。
- `weight`与`bias`的数据类型必须保持一致，且数据类型的精度不能低于`input`的数据类型。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

单算子调用：

```python
>>> import torch
>>> import numpy as np
>>> import torch_npu
>>>
>>> dtype = np.float32
>>> shape_x = [24,320,48,48]
>>> num_groups = 32
>>> shape_c = [320]
>>> eps = 0.00001
>>>
>>> #输入tensor为torch.float32类型
>>> x_npu=torch.randn(shape_x,dtype=torch.float32).npu()          #input
>>> gamma_npu=torch.randn(shape_c,dtype=torch.float32).npu()      #weight
>>> beta_npu=torch.randn(shape_c,dtype=torch.float32).npu()       #bias
>>> out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(x_npu, gamma_npu, beta_npu, group=num_groups, eps=eps)
>>> out_npu.shape, mean_npu.shape, rstd_out.shape
(torch.Size([24, 320, 48, 48]), torch.Size([24, 32]), torch.Size([24, 32]))
>>> out_npu.dtype, mean_npu.dtype, rstd_out.dtype
(torch.float32, torch.float32, torch.float32)
>>>
>>> #输入tensor为torch.bfloat16类型
>>> x_npu=torch.randn(shape_x,dtype=torch.bfloat16).npu()
>>> gamma_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
>>> beta_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
>>> out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(x_npu, gamma_npu, beta_npu, group=num_groups, eps=eps)
>>> out_npu.shape, mean_npu.shape, rstd_out.shape
(torch.Size([24, 320, 48, 48]), torch.Size([24, 32]), torch.Size([24, 32]))
>>> out_npu.dtype, mean_npu.dtype, rstd_out.dtype
(torch.bfloat16, torch.bfloat16, torch.bfloat16)
>>>
>>> #输入tensor为torch.float16类型
>>> x_npu=torch.randn(shape_x,dtype=torch.float16).npu()
>>> gamma_npu=torch.randn(shape_c,dtype=torch.float16).npu()
>>> beta_npu=torch.randn(shape_c,dtype=torch.float16).npu()
>>> out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(x_npu, gamma_npu, beta_npu, group=num_groups, eps=eps)
>>> out_npu.shape, mean_npu.shape, rstd_out.shape
(torch.Size([24, 320, 48, 48]), torch.Size([24, 32]), torch.Size([24, 32]))
>>> out_npu.dtype, mean_npu.dtype, rstd_out.dtype
(torch.float16, torch.float16, torch.float16)

```


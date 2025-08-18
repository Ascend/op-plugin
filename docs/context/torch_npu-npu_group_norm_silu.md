# torch\_npu.npu\_group\_norm\_silu<a name="ZH-CN_TOPIC_0000001979260741"></a>

## 功能说明<a name="zh-cn_topic_0000001935845653_section14441124184110"></a>

-   API功能：计算输入张量input按组归一化的结果，包括张量out、均值meanOut、标准差的倒数rstdOut以及silu的输出。
-   计算公式：
    -   GroupNorm：记$x$为输入input，$\gamma$和$\beta$分别代表输入weight和bias，$E[x] = \bar{x}$代表$x$的均值，$ Var[x]=\frac{1}{n}\sum_{i=1}^{n} (x_i - E[x])^2 $ 代表$x$的方差，则
    $$
    \begin{cases}
    \text{groupnormOut} = \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    \text{meanOut}  = E[x] \\
    \text{rstdOut}  = \frac{1}{\sqrt{Var[x] + eps}}
    \end{cases}
    $$
    -   Silu：
    $$
    \text{out} = \frac{\text{groupnormOut}}{1 + e^{-\text{groupnormOut}}}
    $$

## 函数原型<a name="zh-cn_topic_0000001935845653_section45077510411"></a>

```
torch_npu.npu_group_norm_silu(input, weight, bias, int group, float eps=0.00001) -> (Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001935845653_section112637109429"></a>

-   **input** (`Tensor`)：必选输入，源数据张量，维度需要为2\~8维且第1维度能整除group。数据格式支持ND，支持非连续的Tensor。
    -   <term>Atlas 推理系列产品</term>：数据类型支持float16、float32。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32、bfloat16。

-   **weight** (`Tensor`)：可选输入，索引张量，维度为1且元素数量需与输入input的第1维度保持相同，数据格式支持ND，支持非连续的Tensor。
    -   <term>Atlas 推理系列产品</term>：数据类型支持float16、float32。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32、bfloat16。

-   **bias** (`Tensor`)：可选输入，更新数据张量，维度为1元素数量需与输入input的第1维度保持相同，数据格式支持ND，支持非连续的Tensor。
    -   <term>Atlas 推理系列产品</term>：数据类型支持float16、float32。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32、bfloat16。

-   **group** (`int`)：必选输入，表示将输入input的第1维度分为group组，group需大于0。
-   **eps** (`float`)：可选参数，数值稳定性而加到分母上的值，若保持精度，则eps需大于0。默认值为0.00001。

## 返回值说明<a name="zh-cn_topic_0000001935845653_section22231435517"></a>

-   **out** (`Tensor`)：数据类型和shape与input相同，支持ND，支持非连续的Tensor。
    -   <term>Atlas 推理系列产品</term>：数据类型支持float16、float32。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32、bfloat16。

-   **meanOut** (`Tensor`)：数据类型与input相同，shape为\(N, group\)，其中N为input第0维度值。数据格式支持ND，支持非连续的Tensor。
    -   <term>Atlas 推理系列产品</term>：数据类型支持float16、float32。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32、bfloat16。

-   **rstdOut** (`Tensor`)：数据类型与input相同，shape为\(N, group\)，其中N为input第0维度值。数据格式支持ND，支持非连续的Tensor。
    -   <term>Atlas 推理系列产品</term>：数据类型支持float16、float32。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32、bfloat16。

## 约束说明<a name="zh-cn_topic_0000001935845653_section12345537164214"></a>

-   该接口支持推理、训练场景下使用。
-   input、weight、bias、out、meanOut、rstdOut数据类型必须在支持的范围之内。
-   out、meanOut、rstdOut的数据类型与input相同；weight、bias与input可以不同。
-   weight与bias的数据类型必须保持一致，且数据类型的精度不能低于input的数据类型。
-   weight与bias的维度需为1且元素数量需与输入input的第1维度保持相同。
-   input维度需大于一维且小于等于八维，且input第1维度能整除group。
-   input任意维都需大于0。
-   out的shape与input相同。
-   meanOut与rstdOut的shape为\(N, group\)，其中N为input第0维度值。
-   eps需大于0。
-   group需大于0。

## 支持的型号<a name="zh-cn_topic_0000001935845653_section1691491521017"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
-   <term>Atlas 推理系列产品</term>

## 调用示例<a name="zh-cn_topic_0000001935845653_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import numpy as np
    import torch_npu
     
    dtype = np.float32
    shape_x = [24,320,48,48]
    num_groups = 32
    shape_c = [320]
    eps = 0.00001
     
    input_npu=torch.randn(shape_x,dtype=torch.float32).npu()
    weight_npu=torch.randn(shape_c,dtype=torch.float32).npu()
    bias_npu=torch.randn(shape_c,dtype=torch.float32).npu()
    out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
     
     
    input_npu=torch.randn(shape_x,dtype=torch.bfloat16).npu()
    weight_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
    bias_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
    out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
     
    input_npu=torch.randn(shape_x,dtype=torch.float16).npu()
    weight_npu=torch.randn(shape_c,dtype=torch.float16).npu()
    bias_npu=torch.randn(shape_c,dtype=torch.float16).npu()
    out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
    ```


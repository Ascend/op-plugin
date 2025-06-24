# torch_npu.npu_gelu

## 功能说明

- 算子功能：计算高斯误差线性单元的激活函数。
- 计算公式：

    Gaussian Error Linear Unit(GELU)的表达式为：

    ![](figures/zh-cn_formulaimage_0000002153584017.png)

    Φ(x)是Gaussian Distribution的CDF(Cumulative Distribution Function), 表达式为：

    ![](figures/zh-cn_formulaimage_0000002117988712.png)

## 函数原型

```
torch_npu.npu_gelu(input, approximate='none') -> Tensor
```

## 参数说明

- **input** (`Tensor`)：公式中的$x$，待进行`npu_gelu`计算的入参，数据格式支持$ND$，支持非连续的Tensor。输入最大支持8维。
    - <term>Atlas 训练系列产品</term>：数据类型支持`float16`、`float32`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float32`、`float16`、`bfloat16`。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。

- **approximate** (`Tensor`)：字符串类型，可选参数，计算使用的激活函数模式，可配置为`"none"`或者`"tanh"`。其中`"none"`代表使用erf模式，`"tanh"`代表使用tanh模式。

## 返回值
`Tensor`

数据类型必须和`input`一样，数据格式支持$ND$，shape必须和`input`一样，支持非连续的Tensor。输入最大支持8维。

- <term>Atlas 训练系列产品</term>：数据类型支持`float16`、`float32`。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float32`、`float16`、`bfloat16`。
- <term>Atlas 推理系列产品</term>：数据类型支持`float16`、`float32`。

## 约束说明

- 该接口支持图模式（PyTorch 2.1版本）。
- `input`输入不能含有空指针。

## 支持的型号

- <term>Atlas 训练系列产品</term> 
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> input_tensor = torch.randn(100, 200)
    >>> output_tensor = torch_npu.npu_gelu(input_tensor.npu(), approximate='tanh')
    >>>
    >>> output_tensor
    tensor([[ 0.5795, -0.0274, -0.1477,  ...,  0.2422, -0.0843, -0.1154],
            [-0.0385,  0.8736,  0.1809,  ..., -0.0676,  0.2404,  0.4038],
            [ 0.0438,  0.0205, -0.1536,  ...,  0.2910,  1.1553,  0.3319],
            ...,
            [-0.1698, -0.0031,  0.5120,  ..., -0.1390, -0.0082,  0.6286],
            [ 0.1980,  0.0535, -0.1685,  ..., -0.1528, -0.1484,  1.0703],
            [-0.1351,  1.5851, -0.0222,  ..., -0.0230,  1.4319, -0.1700]],
        device='npu:0')
    >>> output_tensor.shape
    torch.Size([100, 200])
    >>> output_tensor.dtype
    torch.float32
    ```

- 图模式调用

    ```python
    # 入图方式
    import os
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    class Net(torch.nn.Module):
    
        def __init__(self):
            super().__init__()
        def forward(self, self_0, approximate):
            out = torch_npu.npu_gelu(self_0, approximate=approximate)
            return out
    
    x = torch.randn(100, 10, 20).npu()
    model = Net()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = model(x, approximate="none")    
    print(npu_out.shape, npu_out.dtype)

    # 执行上述代码的输出类似如下
    torch.Size([100, 10, 20]) torch.float32
    ```


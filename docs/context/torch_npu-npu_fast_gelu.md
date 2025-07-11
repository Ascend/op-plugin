# torch_npu.npu_fast_gelu

## 功能说明

- 算子功能：快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算`FastGelu`的前向结果。

- 计算公式

    - 公式1：<br>
    ![](figures/zh-cn_formulaimage_0000002232678806.png)
     <br>
         该公式仅支持：<br>
             - <term>Atlas 训练系列产品</term><br>
             - <term>Atlas 推理系列产品</term>
    - 公式2：<br>
    ![](figures/zh-cn_formulaimage_0000002267558185.png)
     <br>
         该公式仅支持：<br>
             - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term><br>
             - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 函数原型

```
torch_npu.npu_fast_gelu(input) -> Tensor
```

## 参数说明

**input** (`Tensor`)：即公式中的$x$。数据格式支持$ND$，支持非连续的Tensor。输入最大支持8维。

- <term>Atlas 训练系列产品</term>：数据类型支持`float16`、`float32`。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`float32`、`bfloat16`。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`float32`、`bfloat16`。
- <term>Atlas 推理系列产品</term>：数据类型仅支持`float16`、`float32`。

## 返回值
`Tensor`

代表`fast_gelu`的计算结果。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- `input`输入不能为None。

## 支持的型号

- <term>Atlas 训练系列产品</term> 
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 
- <term>Atlas 推理系列产品</term> 

## 调用示例

- 单算子调用

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np

    data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
    x = torch.from_numpy(data_var).to(torch.float32).npu()
    y = torch_npu.npu_fast_gelu(x).cpu().numpy()
    ```

- 图模式调用

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    os.environ["ENABLE_ACLNN"] = "false"
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
        def forward(self, x): 
            y = torch_npu.npu_fast_gelu(x)
            return y
            
    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
    x = torch.from_numpy(data_var).to(torch.float32).npu()
    y =npu_mode(x).cpu().numpy()
    print("shape of y:",y.shape)
    print("dtype of y:",y.dtype)
    
    # 执行上述代码的输出类似如下
    shape of y: (4, 2048, 16, 128)
    dtype of y: float32
    ```


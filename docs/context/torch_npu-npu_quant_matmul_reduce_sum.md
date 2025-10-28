# torch_npu.npu_quant_matmul_reduce_sum

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>  | √   |

## 功能说明

- API功能：完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出。

- 计算公式：

$$
out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale_i * x2Scale
$$

## 函数原型

```
torch_npu.npu_quant_matmul_reduce_sum(x1, x2, dims, *, x1_scale=None, x2_scale=None) -> Tensor
```


## 参数说明

- **x1** (`Tensor`)：必选参数，数据类型支持`int8`，数据格式支持ND，shape支持3维，形状为（batch, m, k）。

- **x2** (`Tensor`)：必选参数，数据类型支持`int8`，数据格式支持NZ，shape支持3维，形状为（batch, k, n）。
  - 可通过`x2 = torch_npu.npu_format_cast(x2.contiguous(), 29)`将ND格式的x2转换为NZ格式。

- **x1_scale** (`Tensor`)：必选关键字参数，对应公式中的x1Scale。数据类型支持`float32`，数据格式支持ND，shape支持2维，形状为（batch, m）。
  - 在实际计算时，`x1_scale`会被广播到(batch，m，n)。

- **x2_scale** (`Tensor`)：必选关键字参数，对应公式中的x2Scale。数据类型支持`bfloat16`，数据格式支持ND，shape支持1维，形状为（n,）。
  - 在实际计算时，`x2_scale`会被广播到(batch，m，n)。


## 返回值

`Tensor`:

公式中的$out$，算子的计算结果。输出的数据类型为`bfloat16`，数据格式为ND，shape为2维，形状为(m, n)。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持静态图模式。
- 传入的`x1`、`x2`、`x1_scale`、`x2_scale`不能是空。
- 输入和输出支持以下数据类型组合：
  | x1   | x2   | x1_scale | x2_scale  | out      |
  |------|------|---------|----------|----------|
  | int8 | int8 | float32 | bfloat16 | bfloat16 |


## 调用示例

- 单算子调用
    ```
    import torch
    import torch_npu

    b,m,k,n = (2,3,4,5)
    x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
    x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
    x1_scale = torch.ones((b, m), dtype=torch.float32).npu()
    x2_scale = torch.ones((n,), dtype=torch.bfloat16).npu()
    y = torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=x1_scale, x2_scale=x2_scale)
    ```

- 图模式调用
    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger

    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np

    # "ENABLE_ACLNN"是否使能走aclnn, true: 回调走aclnn, false: 在线编译
    os.environ["ENABLE_ACLNN"] = "false"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
 
        def forward(self, x1, x2, scale, pertoken_scale):
            return torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=pertoken_scale, x2_scale=scale)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)

    b,m,k,n = (2,3,4,5)
    x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
    x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
    pertoken_scale = torch.ones((b, m), dtype=torch.float32).npu()
    scale = torch.ones((n,), dtype=torch.bfloat16).npu()
    npu_out = model(x1, x2, scale, pertoken_scale)
    print(npu_out)
    ```

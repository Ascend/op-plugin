# torch_npu.npu_rotate_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        |    √     |

## 功能说明

API功能：`npu_rotate_quant`是一种融合旋转（Rotate）和量化（Quant）的计算方法。该方法适用于需要对输入数据进行旋转变换后进行量化的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。

## 函数原型

```python
torch_npu.npu_rotate_quant(x, rotation, *, alpha=0.0, dst_dtype=None) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选输入，输入tensor。shape支持2维[m,n]，数据类型支持`bfloat16`和`float16`，数据格式支持ND，支持非连续的Tensor。
- **rotation**（`Tensor`）：必选输入，旋转矩阵tensor。shape支持2维[k,k]，数据类型支持`bfloat16`和`float16`，数据格式支持ND，支持非连续的Tensor。
- **alpha**（`float`）：可选输入，旋转角度缩放因子，数据类型为`float`，默认值为0.0。
- **dst_dtype**（`int`）：可选输入, 指定量化输出的类型, 传None时当做torch.int8处理。

## 返回值说明

- **y**（`Tensor`）：输出的量化结果，shape支持2维[m,n]，与x保持一致，数据类型支持`int4`/`int8`。数据格式支持ND，支持非连续的Tensor。
- **scale**（`Tensor`）：输出的量化因子，shape支持1维[m]，数据类型支持`float32`。数据格式支持ND，支持非连续的Tensor。

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持图模式。
- n支持128-16000，8字节对齐，需要能整除k。
- 输入和输出Tensor支持的数据类型组合如下：

    |x|rotation|dst_dtype|y|scale|
    |--------|--------|--------|--------|--------|
    |`bfloat16`|`bfloat16`|torch.qint8|`int8`|`float32`|
    |`bfloat16`|`bfloat16`|torch.quint4X2|`int4`|`float32`|
    |`float16`|`float16`|torch.qint8|`int8`|`float32`|
    |`float16`|`float16`|torch.quint4X2|`int4`|`float32`|

## 调用示例

- 单算子模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu

    def gen_input_data(M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        return x, rotation

    M = 512
    N = 1024
    K = 1024
    x, rotation = gen_input_data(M, N, K)
    output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=0.0, dst_dtype=torch.int8)
    ```

- 图模式调用：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, rotation):
            output = torch_npu.npu_rotate_quant(x, rotation, alpha=0.0, dst_dtype=torch.int8)
            return output

    def gen_input_data(M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        return x, rotation

    M = 512
    N = 1024
    K = 1024
    x, rotation = gen_input_data(M, N, K)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), rotation.npu())
    ```

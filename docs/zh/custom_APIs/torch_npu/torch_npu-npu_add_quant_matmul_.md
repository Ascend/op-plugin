# torch_npu.npu_add_quant_matmul_

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：

    在micro-batch训练场景，需要做micro-batch的梯度累积，会存在大量QuantBatchMatmul操作接InplaceAdd操作的场景。本算子（QuantBatchMatmulInplaceAdd）将上述操作融合起来，以提高网络性能。

- 计算公式：

    MX量化场景公式如下，更多关于量化技术的介绍参见《CANN 算子库》中的[量化介绍](https://gitcode.com/cann/ops-math/blob/9.2.0/docs/zh/context/quant_mode_introduction.md)。

    $$
    Y_{\mathrm{out}}[m,n]=Y_{\mathrm{in}}[m,n]+\sum_{j=0}^{L-1}\left(\sum_{t=0}^{g_j-1}X_1[m,jg_K+t]X_2[jg_K+t,n]\right)S_1[m,j]S_2[j,n]
    $$

    其中，$X_1$、$X_2$表示应用转置属性后的矩阵，shape分别为$(M,K)$、$(K,N)$；$Y_{\mathrm{in}}$为调用前的`self`，$Y_{\mathrm{out}}$为原地更新后的`self`，两者的shape均为$(M,N)$；$S_1$、$S_2$分别为`x1_scale`、`x2_scale`中对应分组的缩放因子；$g_K=32$，$L=\lceil K/g_K \rceil$，$g_j=\min(g_K, K-jg_K)$，支持尾组（最后一个分组）长度不足$g_K$。

    T-T量化场景计算公式如下。

    $$
    Y_{\mathrm{out}}[m,n]=Y_{\mathrm{in}}[m,n]+\left(\sum_{k=0}^{K-1}X_1[m,k]X_2[k,n]\right)s_1s_2
    $$

    其中，$s_1$、$s_2$分别对应`x1_scale`、`x2_scale`，shape均为$(1,)$。

## 函数原型

```python
torch_npu.npu_add_quant_matmul_(self, x1, x2, x2_scale, *, x1_scale=None, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None) -> torch.Tensor
```

`*`为语法分隔符：其后的参数（`x1_scale`、`group_sizes`、`x1_dtype`等）仅支持关键字（键值对）传参；其前的参数（`self`、`x1`、`x2`、`x2_scale`）为位置参数，也可按名称传入。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| self | Tensor | 必选 | 表示待累加矩阵，对应计算公式中的加法输入矩阵$Y_{\mathrm{in}}$，须以连续布局传入（非连续Tensor会在参数校验阶段报错）。数据格式支持ND。 | torch.float32 | (M, N) |
| x1 | Tensor | 必选 | 表示矩阵乘法中的左矩阵，对应逻辑矩阵shape为(M, K)，须通过shape为(K, M)的连续Tensor执行`transpose(0, 1)`传入，传入后`x1.shape`为(M, K)，如[调用示例](#调用示例)所示；MXFP4量化场景下打包Tensor的shape为(K, M/2)，执行`transpose(0, 1)`后`x1.shape`为(M/2, K)，逻辑矩阵shape仍为(M, K)。数据格式支持ND。 | torch.float8_e5m2、torch.float8_e4m3fn、torch_npu.hifloat8、torch_npu.float4_e2m1fn_x2 | (M, K)（MXFP4量化场景为(M/2, K)） |
| x2 | Tensor | 必选 | 表示矩阵乘法中的右矩阵，`x2.shape`须为(K, N)（对应逻辑矩阵(K, N)），不支持前两维转置的输入；MXFP4量化场景下打包Tensor的`x2.shape`为(K, N/2)。数据格式支持ND。 | torch.float8_e5m2、torch.float8_e4m3fn、torch_npu.hifloat8、torch_npu.float4_e2m1fn_x2 | (K, N)（MXFP4量化场景为(K, N/2)） |
| x2_scale | Tensor | 必选 | 表示矩阵乘法中的右矩阵的缩放因子。MX量化场景数据类型为`torch.float8_e8m0fnu`：传入原生dtype为`torch.float8_e8m0fnu`的Tensor时，`x2_scale_dtype`可保留默认值None；使用`torch.uint8`或`torch.int8`承载E8M0编码时，需配置可选参数`x2_scale_dtype`为`torch_npu.float8_e8m0fnu`，此时`x2_scale`本身的dtype不再生效，但仍需保证`x2_scale`本身的dtype为8bit数据类型，以保证shape正确。T-T量化场景数据类型为`torch.float32`。数据格式支持ND。 | torch.float8_e8m0fnu、torch.float32 | MX量化场景为(ceil(K/64), N, 2)；T-T量化场景为(1, ) |
| x1_scale | Tensor | 可选 | 表示矩阵乘法中的左矩阵的缩放因子，默认值为None。当前支持的MX量化与T-T量化场景均必须提供该参数，不能传入None。MX量化场景数据类型为`torch.float8_e8m0fnu`：传入原生dtype为`torch.float8_e8m0fnu`的Tensor时，`x1_scale_dtype`可保留默认值None；使用`torch.uint8`或`torch.int8`承载E8M0编码时，需配置可选参数`x1_scale_dtype`为`torch_npu.float8_e8m0fnu`，此时`x1_scale`本身的dtype不再生效，但仍需保证`x1_scale`本身的dtype为8bit数据类型，以保证shape正确。T-T量化场景数据类型为`torch.float32`。数据格式支持ND。 | torch.float8_e8m0fnu、torch.float32 | MX量化场景为(ceil(K/64), M, 2)；T-T量化场景为(1, ) |
| group_sizes | List[int] | 可选 | 表示在m、n、k维度上的量化分组情况，默认值为None。非None时，仅支持含3个整数的一维列表，形如[group_m, group_n, group_k]，前两项可填0（0表示由接口根据`x1`、`x2`、`x1_scale`、`x2_scale`输入shape重新推导，假设group_m=0，表示m方向量化分组值由接口推断，推断公式为group_m=m/scale_m，m与x1 shape中的m一致，scale_m与x1_scale shape中的m一致）。MX量化场景要求推导后的分组为[1,1,32]：eager模式下group_k在K为64的整数倍时可由shape推导（如K=64时传入None即可推导为[1,1,32]），K非64的整数倍时（如K=96会推导为24）须显式传入32；T-T量化场景可传入None或[0,0,0]。 | int | - |
| x1_dtype | int | 可选 | 用于显式指定`x1`的实际数据类型。传入值时表示忽略x1本身的dtype，将x1中的数据视为x1_dtype传入的类型进行计算，不传入时则直接取x1的dtype进行计算；传入时仍需保证x1本身的dtype为8bit数据类型，以保证shape正确。默认值为None。 | torch.float8_e4m3fn、torch.float8_e5m2、torch_npu.hifloat8、torch_npu.float4_e2m1fn_x2 | - |
| x2_dtype | int | 可选 | 用于显式指定`x2`的实际数据类型，使用方式同`x1_dtype`。默认值为None。 | torch.float8_e4m3fn、torch.float8_e5m2、torch_npu.hifloat8、torch_npu.float4_e2m1fn_x2 | - |
| x1_scale_dtype | int | 可选 | 用于显式指定`x1_scale`的逻辑数据类型。None：默认值，表示输入真实数据类型与输入`x1_scale`的dtype相同。MX量化场景下当前仅支持传入`torch_npu.float8_e8m0fnu`（`x1_scale`张量本身的dtype可为`torch.float8_e8m0fnu`，或以`torch.uint8`/`torch.int8`承载E8M0编码）；T-T量化场景下两路scale直接传入原生`torch.float32` Tensor即可，无需传入该参数。 | torch_npu.float8_e8m0fnu（MX量化场景） | - |
| x2_scale_dtype | int | 可选 | 用于显式指定`x2_scale`的逻辑数据类型。默认值为`None`，其余用法同`x1_scale_dtype`。 | torch_npu.float8_e8m0fnu（MX量化场景） | - |

> [!NOTE]
>
> - PyTorch 2.7及之前版本：使用8bit Tensor存储已打包的FP4数据，并通过`x1_dtype`、`x2_dtype`指定逻辑数据类型为`torch_npu.float4_e2m1fn_x2`。
> - PyTorch 2.8及以上版本：单算子（eager）模式下还可使用原生数据类型`torch.float4_e2m1fn_x2`构造输入Tensor，此时`x1_dtype`、`x2_dtype`可使用默认值`None`；图模式下须显式传入`torch_npu.float4_e2m1fn_x2`。

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| self | Tensor | 必选 | 表示QuantBatchMatmul计算完成后与待累加矩阵相加得到的最后结果矩阵，原地更新到self并返回，数据类型、shape、数据格式均与输入self保持一致。 | torch.float32 | (M, N) |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- 数据类型约束：

    | 场景 | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | MXFP8量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` |
    | MXFP4量化 | `torch.float4_e2m1fn_x2` | `torch.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` |
    | T-T量化 | `torch_npu.hifloat8`（通过`x1_dtype`指定） | `torch_npu.hifloat8`（通过`x2_dtype`指定） | `torch.float32` | `torch.float32` | `torch.float32` |

- shape约束：

    | 场景 | x1（存储shape/实参shape） | x2（实参shape） | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | MX量化 | (K, M)/(M, K) | (K, N) | (ceil(K/64), N, 2) | (ceil(K/64), M, 2) | (M, N) |
    | T-T量化 | (K, M)/(M, K) | (K, N) | (1, ) | (1, ) | (M, N) |

    上表中x1为“转置前的连续存储shape/传入接口的实参shape”，对应逻辑矩阵shape为(M, K)；MXFP4量化场景下，x1的存储shape为(K, M/2)、实参shape为(M/2, K)，x2的实参shape为(K, N/2)，逻辑矩阵shape不变。

- M、N须为非负整数，K须为正整数（K=0时即使输出为空Tensor也会被拒绝）。
- `x1`须通过shape为(K, M)的Tensor执行`transpose(0, 1)`后传入（底层为(K, M)排布，即传入转置视图），不支持直接传入连续Tensor；图模式下`transpose(0, 1)`应写在图内，如[调用示例](#调用示例)所示。
- MX量化场景下，`x1_scale`须为(ceil(K/64), M, 2)排布，支持shape为(ceil(K/64), M, 2)的连续Tensor或其前两维转置后的视图（如调用示例所示）；`x2_scale`须为(ceil(K/64), N, 2)排布：单算子（eager）模式下支持shape为(ceil(K/64), N, 2)的连续Tensor或其前两维转置后的视图；TorchAir图模式下，转置融合仅消除`x1`与`x1_scale`前的转置节点，会保留`x2_scale`前的转置节点，图中传给算子的`x2_scale`最终轴序须满足(ceil(K/64), N, 2)（R=ceil(K/64)，例如转置前数据按(N, R, 2)排布时，图内执行`transpose(0, 1)`后即满足要求），最终轴序不满足该要求时会在形状校验阶段报错，或导致分组轴与通道轴对应的缩放因子交换、计算结果错误。
- MXFP4量化约束：
  - MXFP4量化场景下，每个字节存放两个FP4编码，`x1`、`x2`打包后的原始存储shape会变为(K,M/2)、(K,N/2)。
  - 每个字节的低4 bit对应原始存储最后一轴的偶数索引元素，高4 bit对应奇数索引元素。M、N必须为偶数。
  - 当M、N、K均大于0时，必须同时满足M × K > 2、N × K > 2。

## 确定性计算

- 确定性说明：默认支持确定性计算。

## 调用示例

- 单算子模式调用
  - MXFP8量化单算子模式调用

    ```python
    import math
    import torch
    import torch_npu
    M = 576
    N = 7168
    K = 512
    y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2_scale = torch.randint(-1, 1, (math.ceil(K/64), N, 2), dtype=torch.int8).npu()
    x1_scale = torch.randint(-1, 1, (math.ceil(K/64), M, 2), dtype=torch.int8).npu().transpose(0,1)
    y = torch_npu.npu_add_quant_matmul_(y, x1, x2, x2_scale,x1_scale = x1_scale, x1_scale_dtype=torch_npu.float8_e8m0fnu, x2_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes = [1,1,32])
    ```

  - MXFP4量化单算子模式调用

    本示例使用原生数据类型`torch.float4_e2m1fn_x2`，要求PyTorch 2.8及以上版本，并安装配套且支持该数据类型的torch_npu版本。

    ```python
    import torch
    import torch_npu

    M = 576
    N = 7168
    K = 512
    R = (K + 63) // 64
    y = torch.ones((M, N), dtype=torch.float32).npu()
    x1 = torch.full((K, M // 2), 0x22, dtype=torch.uint8)
    x1 = x1.view(torch.float4_e2m1fn_x2).npu().transpose(0, 1)
    x2 = torch.full((K, N // 2), 0x22, dtype=torch.uint8)
    x2 = x2.view(torch.float4_e2m1fn_x2).npu()
    x1_scale = torch.full((R, M, 2), 127, dtype=torch.uint8).npu()
    x2_scale = torch.full((R, N, 2), 127, dtype=torch.uint8).npu()
    y = torch_npu.npu_add_quant_matmul_(
        y, x1, x2, x2_scale,
        x1_scale=x1_scale,
        group_sizes=[1, 1, 32],
        x1_dtype=torch.float4_e2m1fn_x2,
        x2_dtype=torch.float4_e2m1fn_x2,
        x1_scale_dtype=torch_npu.float8_e8m0fnu,
        x2_scale_dtype=torch_npu.float8_e8m0fnu,
    )
    torch.testing.assert_close(y, torch.full_like(y, K + 1), rtol=0, atol=0)
    print(y.cpu())
    ```

  - T-T量化单算子模式调用

    ```python
    import math
    import torch
    import torch_npu
    M = 16
    N = 16
    K = 16
    y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
    x1 = torch.randint(0, 1, (K, M), dtype=torch.uint8).npu().transpose(0,1)
    x2 = torch.randint(0, 1, (K, N), dtype=torch.uint8).npu()
    x2_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
    y = torch_npu.npu_add_quant_matmul_(y, x1, x2, x2_scale,x1_scale = x1_scale,x1_dtype =  torch_npu.hifloat8, x2_dtype = torch_npu.hifloat8, x1_scale_dtype=None, x2_scale_dtype=None, group_sizes = [0,0,0])
    ```

- 图模式调用
  - MXFP8量化图模式调用

    ```python
    import math
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 576
    N = 7168
    K = 512
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, x1_scale, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_matmul_(y, x1.transpose(0,1), x2, x2_scale, x1_scale = x1_scale.transpose(0, 1), x1_scale_dtype=x1_scale_dtype, x2_scale_dtype=x2_scale_dtype)
    def main():
        y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2_scale = torch.randint(-1, 1, (math.ceil(K/64),N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(-1, 1, (math.ceil(K/64),M, 2), dtype=torch.int8).npu()
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, x1_scale, torch_npu.float8_e8m0fnu, torch_npu.float8_e8m0fnu)
        print(y.cpu())

    if __name__ == '__main__':
        main()
    ```

  - T-T量化图模式调用

    ```python
    import math
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 16
    N = 16
    K = 16
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, x1_scale, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_matmul_(y, x1.transpose(0,1), x2, x2_scale, x1_dtype =  torch_npu.hifloat8, x2_dtype = torch_npu.hifloat8, x1_scale = x1_scale, x1_scale_dtype=None, x2_scale_dtype=None)
    def main():
        y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
        x1 = torch.randint(0, 1, (K, M), dtype=torch.uint8).npu()
        x2 = torch.randint(0, 1, (K, N), dtype=torch.uint8).npu()
        x2_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
        x1_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, x1_scale, None, None)
        print(y.cpu())

    if __name__ == '__main__':
        main()
    ```

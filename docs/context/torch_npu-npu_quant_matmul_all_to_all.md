# torch_npu.npu_quant_matmul_all_to_all

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |
## 功能说明

- API功能：完成量化的Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。
    支持K-C量化模式
- 计算公式:
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)
    - K-C量化模式：
      $$
      computeOut = (x1 @ x2 + bias) * x1Scale * x2Scale \\
      permutedOut = computeOut.view(BS, worldSize, H2 / worldSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(worldSize * BS, H2 / worldSize)
      $$
    - K-C量化模式带bias：
      $$
      computeOut = (x1 @ x2) * x1Scale * x2Scale  + bias \\
      permutedOut = computeOut.view(BS, worldSize, H2 / worldSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(worldSize * BS, H2 / worldSize)
      $$

## 函数原型

```
torch_npu.npu_quant_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, Tensor? common_scale=None, Tensor? x1_offset=None, Tensor? x2_offset=None, int x1_quant_mode=3, int x2_quant_mode=2, int common_quant_mode=0, int[]? group_sizes=None, int[]? all2all_axes=None, int comm_quant_dtype=0, int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None, int? x2_scale_dtype=None, int? output_scale_dtype=None, int? comm_scale_dtype=None, int? y_dtype=None) -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，融合算子的左矩阵输入，对应公式中的 `x1`。该输入作为 MatMul 计算的左矩阵输入。
    - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
    - 维数为 2 维，shape 为 `(BS, H1)`。
    - 数据类型支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。

- **x2** (`Tensor`)：必选参数，融合算子的右矩阵输入，对应公式中的 `x2`。直接作为 MatMul 计算的右矩阵输入。
    - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
    - 维数为 2 维，shape 为 `(H1, H2)`。
    - 数据类型支持 `FLOAT8_E4M3FN`、`FLOAT8_E5M2`。

- **hcom** (`string`)：必选参数，通信域名。
    - 字符串长度要求为 `(0, 128)`。
    - 数据类型为 `STRING`。

- **world_size** (`int`)：必选参数，通信域内的rank总数。
    - 对应公式中的 `worldSize`，支持值为 `2`、`4`、`8`、`16`。
    - 数据类型为 `INT`。

- **bias** (`Tensor`)：可选参数，阵乘运算后累加的偏置，对应公式中的 `bias`。
    - 传入非空时生效。
    - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
    - 维数为 1 维，shape 为 `(H2)`。
    - 数据类型支持 `FLOAT32`。

- **x1_scale** (`Tensor`)：必选参数，左矩阵的量化系数。
    - 对应公式中的 `x1Scale`。
    - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
    - 维数为 1 维，shape 为 `(BS)`（K-C 量化模式下）。
    - 数据类型支持 `FLOAT32`。

- **x2_scale** (`Tensor`)：必选参数，右矩阵的量化系数。
    - 对应公式中的 `x2Scale`。
    - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
    - 维数为 1 维，shape 为 `(H2)`（K-C 量化模式下）。
    - 数据类型支持 `FLOAT32`。

- **comm_scale** (`Tensor`)：可选参数，低比特通信的量化系数。
    - 预留参数，暂不支持低比特通信。

- **x1_offset** (`Tensor`)：可选参数，左矩阵的量化偏置。
    - 预留参数，暂不支持。

- **x2_offset** (`Tensor`)：可选参数，右矩阵的量化偏置。
    - 预留参数，暂不支持。

- **x1_quant_mode** (`int`)：必选参数，左矩阵的量化方式。
    - 当前仅支持配置为 `3`，表示 PerToken。
    - 数据类型为 `INT`。

- **x2_quant_mode** (`int`)：必选参数，右矩阵的量化方式。
    - 当前仅支持配置为 `2`，表示 PerChannel。
    - 数据类型为 `INT`。

- **comm_quant_mode** (`int`)：可选参数，低比特通信的量化方式。
    - 预留参数，当前仅支持配置为 `0`，表示不量化。
    - 数据类型为 `INT`。

- **group_sizes** (`List[int]`)：可选参数，用于 MatMul 计算三个方向上的量化分组大小。
    - 预留参数，K-C 量化模式下仅支持配置为 `0`，取值不生效。
    - `groupSize` 输入由三个方向的 `groupSizeM`、`groupSizeN`、`groupSizeK` 拼接组成，每个值占 16 位，共占用 `int64_t` 类型 `groupSize` 的低 48 位。
    - 计算公式为：`groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32`。
    - 数据类型为 `List[int]`。

- **all2all_axes** (`List[int]`)：可选参数，AlltoAll 和 Permute 数据交换的方向。
    - 支持配置空或者 `[-1, -2]`，传入空时默认按 `[-1, -2]` 处理，表示将输入由 `(BS, H2)` 转为 `(BS * worldSize, H2 / worldSize)`。
    - 维数为 1 维，shape 为 `(2)`。
    - 数据类型为 `List[int]`。

- **comm_quant_dtype** (`int`)：可选参数，低比特通信的量化类型。
    - 预留参数，暂不支持。

- **x1_dtype** (`int`)：可选参数，`x1` 的数据类型。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

- **x2_dtype** (`int`)：可选参数，`x2` 的数据类型。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

- **x1_scale_dtype** (`int`)：可选参数，`x1_scale` 的数据类型。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

- **x2_scale_dtype** (`int`)：可选参数，`x2_scale` 的数据类型。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

- **output_scale_dtype** (`int`)：可选参数，对输出 tensor 进行量化 scale 的数据类型。
    - 预留参数，暂不支持。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

- **comm_scale_dtype** (`int`)：可选参数，低比特通信的量化 scale 数据类型。
    - 预留参数，暂不支持。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

- **y_dtype** (`int`)：可选参数，输出 `output` 的数据类型。
    - 用于在 PyTorch 侧标识海思特有的数据类型。

x1_quant_mode、x2_quant_mode、comm_quant_mode的枚举值跟量化模式关系如下:
* 0: 不量化
* 1: pertensor
* 2: perchanenl
* 3: pertoken
* 4: pergroup
* 5: perblock
* 6: mx量化
* 7: pertoken动态量化

## 返回值说明
`Tensor`

代表`npu_quant_matmul_all_to_all`的计算结果，对应公式中的output，shape为`(BS * worldSize, H2 / worldSize)`，数据类型支持`FLOAT16`、`BFLOAT16`、`FLOAT32`。

## 约束说明
* 默认支持确定性计算
* 右矩阵和输出矩阵的H2必须整除worldSize
* 仅支持左矩阵perToken量化，x1QuantMode=3，右矩阵perChannel量化,x2QuantMode=2
* Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：传入的x1、x2、biasOptional、x1Scale、x2Scale或者output不为空指针
* Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：x1、x2计算输入的数据类型必须为INT8，output计算输出的数据类型为BFLOAT16时，biasOptional的数据类型为FLOAT或BFLOAT16，output的数据类型为FLOAT16时，biasOptional的数据类型为FLOAT16
* H1范围仅支持[1, 65535]
* worldSize仅支持2,4,8,16
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：支持2、4、8卡
    - 昇腾910_95 AI处理器：支持2,4,8,16卡
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用
* 不支持跨超节点通信，只支持超节点内

## 调用示例
- 单算子模式调用
  ```python
    import torch
    import torch_npu
    
    # 初始化输入
    x1 = torch.randint(-1, 2, (16, 32), dtype=torch.float8_e4m3fn).npu()
    x2 = torch.randint(-1, 2, (32, 32), dtype=torch.float8_e4m3fn).npu()
    bias = torch.randint(-1, 2, (32,), dtype=torch.float32).npu()
    x1Scale = torch.randint(-1, 2, (32,), dtype=torch.float32).npu()
    x2Scale = torch.randint(-1, 2, (32,), dtype=torch.float32).npu()
    # 其他参数
    hcom = "fake group info"
    world_size = 2
    # 调用QuantMatmulAlltoAll算子
    res = torch_npu.npu_quant_matmul_all_to_all(x1, x2, hcom, world_size,
                bias=bias, x1_scale=x1Scale, x2_scale=x2Scale, group_size=[0], all2all_axes=[-1, -2])
    # 输出res的形状类似于
    tensor([[-1.3330,  0.3899,  0.3205,  ...,  0.3482, -0.4815,  0.4637],
            [ 0.3511,  1.7768, -0.2588,  ..., -0.3826,  0.7349, -0.0110],
            [ 1.2357, -1.0084, -1.4964,  ..., -0.4084, -0.9424, -0.5332],
            ...,
            [ 1.2147, -1.7824,  1.6442,  ...,  0.6114, -1.6988,  1.1249],
            [ 0.5168, -0.7395, -0.3615,  ...,  1.6038,  0.0826,  0.8413],
            [ 1.0905, -0.1828,  0.7823,  ...,  0.2952, -0.7067,  0.9949]])
  ```


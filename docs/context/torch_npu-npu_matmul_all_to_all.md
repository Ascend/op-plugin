# torch_npu.npu_matmul_all_to_all

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

- API功能：完成Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。
- 计算公式:
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)
  $$
  computeOut = x1 @ x2 + bias \\
  permutedOut = computeOut.view(BS, worldSize, H2/worldSize).permute(1, 0, 2) \\
  output = AlltoAll(permutedOut).view(worldSize*BS, H2/worldSize)
  $$

## 函数原型

```
torch_npu.npu_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, int[]? all2all_axes=None) -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，融合算子的左矩阵输入，对应公式中的 `x1`。该输入作为 MatMul 计算的左矩阵输入。
  - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
  - 维数为 2 维，shape 为 `(BS, H1)`。
  - 数据类型支持 `FLOAT16`、`BFLOAT16`。

- **x2** (`Tensor`)：必选参数，融合算子的右矩阵输入，也是 MatMul 计算的右矩阵。直接作为 MatMul 计算的右矩阵输入。
  - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
  - 维数为 2 维，shape 为 `(H1, H2)`。
  - 数据类型支持 `FLOAT16`、`BFLOAT16`。

- **hcom** (`string`)：必选参数，通信域名。
  - 字符串长度要求为 `(0, 128)`。
  - 数据类型为 `STRING`。

- **world_size** (`int`)：必选参数，通信域内的rank总数。
  - 对应公式中的 `worldSize`，支持值为 `2`、`4`、`8`、`16`。
  - 数据类型为 `INT`。

- **bias** (`Tensor`)：可选参数，阵乘运算后累加的偏置，对应公式中的 `bias`。
  - 数据格式支持 $ND$，暂不支持非连续的 Tensor。
  - 维数为 1 维，shape 为 `(H2)`。
  - 数据类型支持 `FLOAT16`、`BFLOAT16`、`FLOAT32`。

- **all2all_axes** (`List[int]`)：可选参数，AlltoAll 和 Permute 数据交换的方向。
  - 支持配置空或者 `[-1, -2]`，传入空时默认按 `[-1, -2]` 处理，表示将输入由`(BS, H2)`转为`(BS * worldSize, H2 / worldSize)`。
  - 维数为 1 维，shape 为 `(2)`。
  - 数据类型为 `List[int]`。

## 返回值说明
`Tensor`

代表`npu_matmul_all_to_all`的计算结果，对应公式中的output，shape为`(BS * worldSize, H2 / worldSize)`，数据类型与输入x1和x2一致。

## 约束说明
* 默认支持确定性计算
* 右矩阵和输出矩阵的H2必须整除worldSize
* 仅支持BS为0的空tensor
* H1范围仅支持[1, 65535]
* worldSize仅支持2,4,8,16
* x1、x2的数据类型必须一致
* bias的数据类型可以为x1和x2的数据类型，也可以为float32
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用
* 不支持跨超节点通信，只支持超节点内

## 调用示例
- 单算子模式调用
  ```python
  import torch
  import torch_npu
  
  # 初始化输入
  x1 = torch.randint(-1, 2, (16, 32), dtype=torch.float16).npu()
  x2 = torch.randint(-1, 2, (32, 32), dtype=torch.float16).npu()
  bias = torch.randint(-1, 2, (32,), dtype=torch.float16).npu()
  # 其他参数
  hcom = "fake group info"
  world_size = 2
  # 调用MatmulAlltoAll算子
  res = torch_npu.npu_matmul_all_to_all(x1, x2, hcom, world_size, bias=bias, all2all_axes=[-1,-2])
  # 输出res的形状类似于
  tensor([[ 0.7483, -0.2262, -0.5252,  ..., -0.6810,  1.5576, -0.3997],
          [ 0.3010,  0.9226,  0.7723,  ..., -1.0785,  0.8963, -0.4670],
          [-1.6897, -0.6455,  0.0328,  ..., -0.6632, -0.0950, -0.0679],
          ...,
          [ 0.2493, -0.2285,  1.1562,  ..., -0.2299,  0.1272,  0.4843],
          [-0.2058, -0.5962, -0.2101,  ...,  1.2606,  0.8451,  1.1243],
          [-0.8396,  1.3138, -0.4066,  ..., -0.7683,  1.3133,  0.7471]])
  ```


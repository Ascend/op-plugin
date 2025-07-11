# torch_npu.matmul_checksum
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品  | √   |
| Atlas 训练系列产品                                       |    √     |

## 功能说明

提供基于原生torch.matmul和Tensor.matmul接口的aicore错误硬件故障接口，内部执行矩阵计算结果校验过程，并对校验误差和实时计算的校验门限进行对比，判断校验误差是否超越门限，若超越则认为发生了aicore错误。

## 函数原型

```
torch_npu.matmul_checksum(a, b, c) -> Tensor
```

## 参数说明

- **a** (`Tensor`)：必选输入，进行原生matmul计算的输入`input`。
- **b** (`Tensor`)：必选输入，进行原生matmul计算的输入`other`。
- **c** (`Tensor`)：必选输入，原生matmul计算的输出`out`。

## 返回值说明
`Tensor`

返回NPU上的bool标量，结果为True，标识存在aicore错误的硬件故障。

## 约束说明

该接口仅支持bf16格式，且device为NPU的场景。

## 调用示例


   ```python
    >>> import torch
    >>> import torch_npu
    >>> matrix1 = torch.randn(2000, 2000, device='npu', dtype=torch.bfloat16)
    >>> matrix2 = torch.randn(2000, 2000, device='npu', dtype=torch.bfloat16)
    >>> product = torch.matmul(matrix1, matrix2)
    >>> checksum = torch_npu.matmul_checksum(matrix1, matrix2, product)
    >>> print(checksum)
    tensor(False, device='npu:0')
   ```

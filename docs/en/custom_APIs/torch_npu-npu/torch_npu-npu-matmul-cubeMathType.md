# torch_npu.npu.matmul.cube_math_type

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Sets or queries the cube computation type for matmul operators.

`torch_npu.npu.matmul.cube_math_type` sets or queries the computation precision type of matmul operators. Setting different `CubeMathType` enumeration values controls the mathematical computation mode of the operators.

## Prototype

```python
torch_npu.npu.matmul.cube_math_type = CubeMathType
```

`CubeMathType` is an enumeration class that can be imported through `torch_npu.npu.CubeMathType` or `from torch_npu.npu import CubeMathType` to query and set configurations.

## Parameters

**`cube_math_type`**: Cube computation type. The options are defined in the following table.

| Enumeration Value                     | Value  | Description                              |
| --------------------------- | ---- | ---------------------------------- |
| CubeMathType.KEEP_DTYPE     | 0    | Retains the original data type without precision conversion.  |
| CubeMathType.ALLOW_FP32_DOWN_PRECISION | 1    | Allows FP32 precision reduction.                    |
| CubeMathType.USE_FP16       | 2    | Enables FP16 computation mode.                  |
| CubeMathType.USE_HF32       | 3    | Enables HF32 computation mode.                  |
| CubeMathType.USE_FP32_ADD | 4    | Enables high-precision mode.              |

The `aclnnAddmv`, `aclnnAddbmm`, `aclnnInplaceAddbmm`, `aclnnBatchMatMul`, `aclnnBatchMatMulWeightNz`, `aclnnGemm`, `aclnnAddmmWeightNz`, `aclnnMv`, `aclnnTransposeBatchMatMul`, and `aclnnTransposeBatchMatMulWeightNz` operators do not support `CubeMathType=4` and will fall back to `CubeMathType=0`. 

## Return Values

The `CubeMathType` enumeration type.

## Example

```python
>>>import torch
>>>import torch_npu
>>>from torch_npu.npu import CubeMathType
>>>print(torch_npu.npu.matmul.cube_math_type)
None

>>>torch_npu.npu.matmul.cube_math_type = CubeMathType.USE_HF32
>>>print(torch_npu.npu.matmul.cube_math_type)
_CubeMathType.USE_HF32
>>>torch_npu.npu.matmul.cube_math_type = torch_npu.npu.CubeMathType.KEEP_DTYPE
>>>print(torch_npu.npu.matmul.cube_math_type)
_CubeMathType.KEEP_DTYPE
```

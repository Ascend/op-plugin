# torch_npu.npu.matmul.cube_math_type

> [!NOTICE]  
> 此接口为本版本新增功能，具体依赖要求请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

设置或查询matmul类算子的cube计算类型。

`torch_npu.npu.matmul.cube_math_type` 用于设置或查询matmul算子的计算精度类型。通过设置不同的CubeMathType枚举值，以控制算子的数学计算模式。

## 函数原型

```python
torch_npu.npu.matmul.cube_math_type = CubeMathType
```

`CubeMathType` 是枚举类，可通过 `torch_npu.npu.CubeMathType` 或者`from torch_npu.npu import CubeMathType`导入来查询和设置。

## 参数说明

**CubeMathType**：设置cube计算类型，可选值如下：

| 枚举值                      | 值   | 说明                               |
| --------------------------- | ---- | ---------------------------------- |
| CubeMathType.KEEP_DTYPE     | 0    | 保持原始数据类型，不进行精度转换   |
| CubeMathType.ALLOW_FP32_DOWN_PRECISION | 1    | 允许FP32降精度                     |
| CubeMathType.USE_FP16       | 2    | 使用FP16计算模式                   |
| CubeMathType.USE_HF32       | 3    | 使用HF32计算模式                   |
| CubeMathType.USE_FP32_ADD | 4    | 使用高精度模式               |

## 返回值说明

返回`CubeMathType`枚举类型。

## 调用示例

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

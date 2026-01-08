# （beta）torch_npu.utils.is_combined_tensor_valid
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

校验Tensor列表中的Tensor是否全部属于一个经过`torch_npu.utils.npu_combine_tensors`融合后的新融合Tensor。

## 函数原型

```
torch_npu.utils.is_combined_tensor_valid(combined_tensor, list_of_tensor) -> bool
```

## 参数说明

- **combined_tensor** (`Tensor`)：经过`torch_npu.utils.npu_combine_tensors`融合后的融合Tensor。
- **list_of_tensor** (`List[Tensor]`)：需要进行校验的Tensor列表。

## 返回值说明
`bool`

代表Tensor列表`list_of_tensor`中的Tensor是否全部属于融合Tensor `combined_tensor`。

## 约束说明

融合Tensor `combined_tensor`及`list_of_tensor`中的Tensor须全部为内存连续的、dtype一致的NPU Tensor。


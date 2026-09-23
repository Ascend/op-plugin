# （beta）torch_npu.utils.is_combined_tensor_valid

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

校验tensor列表中的tensor是否全部属于一个经过`torch_npu.utils.npu_combine_tensors`融合后的新融合tensor。

## 函数原型

```python
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

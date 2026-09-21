# （beta）torch_npu.utils.get_part_combined_tensor

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

根据地址偏移及内存大小，从经过`torch_npu.utils.npu_combine_tensors`融合后的融合Tensor中获取局部Tensor。

## 函数原型

```python
torch_npu.utils.get_part_combined_tensor(combined_tensor, index, size) -> Tensor
```

## 参数说明

- **combined_tensor** (`Tensor`)：经过`torch_npu.utils.npu_combine_tensors`融合后的融合Tensor。
- **index** (`Long`)：需获取的局部Tensor相对于融合Tensor的偏移地址。
- **size** (`Long`)：需获取的局部Tensor的大小。

## 返回值说明

`Tensor`

代表从融合Tensor中获取的局部Tensor。

## 约束说明

`index`+`size`不超过`combined_tensor`的内存大小。

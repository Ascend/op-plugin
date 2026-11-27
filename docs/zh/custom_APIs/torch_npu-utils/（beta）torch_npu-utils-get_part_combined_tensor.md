# （beta）torch_npu.utils.get_part_combined_tensor

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

根据元素偏移和元素数量，从经过`torch_npu.utils.npu_combine_tensors`融合后的融合Tensor中获取局部Tensor。

## 函数原型

```python
torch_npu.utils.get_part_combined_tensor(combined_tensor, index, size) -> Optional[Tensor]
```

## 参数说明

- **combined_tensor** (`Tensor`)：经过`torch_npu.utils.npu_combine_tensors`融合后的融合Tensor。
- **index** (`int`)：必选参数，从融合Tensor的第几个元素开始截取，起始元素为`0`。
- **size** (`int`)：必选参数，要截取的元素数量，须为非负整数。

## 返回值说明

`Tensor`或`None`。返回的局部Tensor与融合Tensor共享存储。`combined_tensor`为`None`或`size`为`0`时返回`None`。

## 约束说明

`index`和`size`均以元素为单位，其和不得超过`torch_npu.get_storage_size(combined_tensor)`。例如，`get_part_combined_tensor(combined_tensor, 2, 3)`返回从索引`2`开始的`3`个元素。

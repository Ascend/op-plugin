# （beta）torch_npu.utils.npu_combine_tensors

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

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

应用基于NPU的tensor融合操作，将NPU上的多个tensor融合为内存连续的一个新tensor，访问原tensor时实际访问新融合tensor的对应偏移地址。

## 函数原型

```python
torch_npu.utils.npu_combine_tensors(list_of_tensor, require_copy_value=True) -> Tensor
```

## 参数说明

- **list_of_tensor** (`List[Tensor]`)：需要进行融合的Tensor列表。
- **require_copy_value** (`bool`)：默认值为True，是否将原Tensor的值拷贝到新融合Tensor的对应偏移地址。

## 返回值说明

`Tensor`

代表融合后的新Tensor。

## 约束说明

`list_of_tensor`列表中须全部为内存连续的、dtype相同的NPU Tensor。

# （beta）torch_npu.utils.get_part_combined_tensor

## 函数原型

```
torch_npu.utils.get_part_combined_tensor(combined_tensor, index, size) -> Tensor
```

## 功能说明

根据地址偏移及内存大小，从经过torch_npu.utils.npu_combine_tensors融合后的融合Tensor中获取局部Tensor。

## 参数说明

- combined_tensor(Tensor)：经过torch_npu.utils.npu_combine_tensors融合后的融合Tensor。
- index(Long)：需获取的局部Tensor相对于融合Tensor的偏移地址。
- size(Long)：需获取的局部Tensor的大小。

## 输出说明

torch.Tensor：从融合Tensor中获取的局部Tensor。

## 约束说明

index+size不超过combined_tensor的内存大小。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


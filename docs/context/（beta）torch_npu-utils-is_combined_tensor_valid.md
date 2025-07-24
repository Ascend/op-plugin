# （beta）torch_npu.utils.is_combined_tensor_valid

## 函数原型

```
torch_npu.utils.is_combined_tensor_valid(combined_tensor, list_of_tensor) -> Bool
```

## 功能说明

校验Tensor列表中的Tensor是否全部属于一个经过torch_npu.utils.npu_combine_tensors融合后的新融合Tensor。

## 参数说明

- combined_tensor(Tensor)：经过torch_npu.utils.npu_combine_tensors融合后的融合Tensor。
- List_of_tensor(ListTensor)：需要进行校验的Tensor列表。

## 输出说明

Bool：Tensor列表list_of_tensor中的Tensor是否全部属于融合Tensor combined_tensor。

## 约束说明

融合Tensor combined_tensor及list_of_tensor中的Tensor须全部为内存连续的、dtype一致的NPU Tensor。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


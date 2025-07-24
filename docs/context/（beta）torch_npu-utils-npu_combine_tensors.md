# （beta）torch_npu.utils.npu_combine_tensors

## 函数原型

```
torch_npu.utils.npu_combine_tensors(list_of_tensor, require_copy_value=True) -> Tensor
```

## 功能说明

应用基于NPU的Tensor融合操作，将NPU上的多个Tensor融合为内存连续的一个新Tensor，访问原Tensor时实际访问新融合Tensor的对应偏移地址。

## 参数说明

- List_of_tensor(ListTensor)：需要进行融合的Tensor列表。
- require_copy_value(Bool，默认值为True)：是否将原Tensor的值拷贝到新融合Tensor的对应偏移地址。

## 输出说明

torch.Tensor：融合后的新Tensor。

## 约束说明

list_of_tensor列表中须全部为内存连续的、dtype相同的NPU Tensor。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


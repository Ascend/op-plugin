# （beta）torch_npu.copy_memory_

> **须知：**<br>
>该接口计划废弃，可以使用`torch.Tensor.copy_`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

从src拷贝元素到self张量，并原地返回self张量。

## 函数原型

```
torch_npu.copy_memory_(dst, src, non_blocking=False) -> Tensor
```

## 参数说明

- **dst**（`Tensor`）：必选参数，拷贝源张量。
- **src**（`Tensor`）：必选参数，返回张量所需数据类型。
- **non_blocking**（`bool`）：可选参数，默认值为`False`。如果设置为`True`且此拷贝位于CPU和NPU之间，则拷贝可能相对于主机异步发生。在其他情况下，此参数没有效果。

## 约束说明

copy_memory_仅支持NPU张量。copy_memory_的输入张量应具有相同的dtype和设备index。

## 调用示例

```python
>>> a=torch.IntTensor([0,  0, -1]).npu()
>>> b=torch.IntTensor([1, 1, 1]).npu()
>>> torch_npu.copy_memory_(a, b)
tensor([1, 1, 1], device='npu:0', dtype=torch.int32)
```


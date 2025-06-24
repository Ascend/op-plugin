# （beta）torch\_npu.copy\_memory\_

> **须知：**<br>
>该接口计划废弃，可以使用torch.Tensor.copy\_接口进行替换。

## 函数原型

```
torch_npu.copy_memory_(dst, src, non_blocking=False) -> Tensor
```

## 功能说明

从src拷贝元素到self张量，并原地返回self张量。

## 参数说明

-   dst \(Tensor\) - 拷贝源张量。
-   src \(Tensor\) - 返回张量所需数据类型。
-   non\_blocking \(Bool，默认值为False\) - 如果设置为True且此拷贝位于CPU和NPU之间，则拷贝可能相对于主机异步发生。在其他情况下，此参数没有效果。

## 约束说明

copy\_memory\_仅支持NPU张量。copy\_memory\_的输入张量应具有相同的dtype和设备index。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> a=torch.IntTensor([0,  0, -1]).npu()
>>> b=torch.IntTensor([1, 1, 1]).npu()
>>> torch_npu.copy_memory_(a, b)
tensor([1, 1, 1], device='npu:0', dtype=torch.int32)
```


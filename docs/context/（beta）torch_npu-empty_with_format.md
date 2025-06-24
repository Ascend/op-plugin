# （beta）torch_npu.empty_with_format

>**须知：**<br>
>该接口计划废弃，可以使用torch.empty接口进行替换。

## 函数原型

```
torch_npu.empty_with_format(size, dtype, layout, device, pin_memory, acl_format)
```

## 功能说明

返回一个填充未初始化数据的张量。

## 参数说明

- size (ListInt) - 定义输出张量shape的整数序列。可以是参数数量（可变值），也可以是列表或元组等集合。
- dtype (torch.dtype，可选，默认值为None) - 返回张量所需数据类型。如果值为None，请使用全局默认值（请参见torch.set_default_tensor_type()）。
- layout (torch.layout，可选，默认值为torch.strided) - 返回张量所需布局。
- device (torch.device，可选，默认值为None) - 返回张量所需设备。
- pin_memory (Bool，可选，默认值为False) - 如果设置此参数，返回张量将分配在固定内存中。
- acl_format (Int，默认值为2) - 返回张量所需内存格式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='npu:0')
```


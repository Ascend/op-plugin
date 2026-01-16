# （beta）torch_npu.empty_with_format

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.empty`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

返回一个填充未初始化数据的张量。

## 函数原型

```
torch_npu.empty_with_format(size, dtype, layout, device, pin_memory, acl_format)
```

## 参数说明

- **size**（`List[int]`）：定义输出张量shape的整数序列。可以是参数数量（可变值），也可以是列表或元组等集合。
- **dtype**（`torch.dtype`）：可选参数，返回张量所需数据类型，默认值为None。如果值为None，请使用全局默认值（请参见torch.set_default_tensor_type()）。
- **layout** （`torch.layout`）：可选参数，返回张量所需布局，默认值为torch.strided。
- **device**（`torch.device`）：可选参数，返回张量所需设备，默认值为None。
- **pin_memory**（`bool`）：可选参数，默认值为False。如果设置此参数，返回张量将分配在固定内存中。
- **acl_format**（`int`）：返回张量所需内存格式，默认值为2。

## 调用示例

```python
>>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='npu:0')
```


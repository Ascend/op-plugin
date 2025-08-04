# torch_npu.empty_with_swapped_memory
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

申请一个device信息为NPU且实际内存在host侧的特殊Tensor。

## 函数原型

```
torch_npu.empty_with_swapped_memory(size, *, dtype=None, device=None) -> Tensor
```

## 参数说明

- **size** (`ListInt`)：必选参数，定义输出张量shape的整数序列，可以是参数数量（可变值），也可以是列表或元组等集合。
- **dtype** (`torch.dtype`)：可选参数，必须使用关键字传参，表示生成Tensor的数据类型，默认值为`None`，表示使用dtype全局默认值。
- **device** (`torch.device`)：可选参数，必须使用关键字传参，表示生成Tensor的设备信息，默认值为`None`，表示使用当前默认device。



## 返回值说明
`Tensor`

代表生成的特殊Tensor。

## 约束说明

- 该接口暂不支持图模式。

- 该接口申请的特殊Tensor当前仅支持如下算子：<br>
`torch.fill_`<br>
`torch.zero_`<br>
`torch.mul_`<br>
`torch_npu.npu_apply_adam_w`<br>
`torch_npu.npu_hans_encode`<br>
`torch_npu.npu_hans_decode`<br>

- 该接口申请的特殊Tensor不支持直接打印，需要查看值时要先通过`mul_`转为普通Tensor再打印，详情请参考调用示例。


## 调用示例

单算子模式调用

```python
>>> import torch
>>> import torch_npu
>>>
>>> swapped_tensor = torch_npu.empty_with_swapped_memory([2, 2], dtype=torch.float32, device=torch.device("npu:0"))
>>> tmp_tensor = swapped_tensor.fill_(3.14)
>>> out = torch.empty_like(swapped_tensor).fill_(1).mul_(tmp_tensor)
>>> out
tensor([[3.1400, 3.1400],
        [3.1400, 3.1400]], device='npu:0')
```

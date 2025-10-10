# （beta）torch_npu.contrib.function.roll

## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

使用NPU亲和写法替换swin-transformer中的原生roll。

## 函数原型
```
torch_npu.contrib.function.roll(input1, shifts, dims)
```


## 参数说明

- **input1** (`Tensor`)：输入张量。
- **shifts** (`Tuple` of `ints`)：每个维度张量滚动（roll）的位移量。
- **dims** (`Tuple` of `ints`)：要滚动的维度。

## 返回值说明

`Tensor`

滚动之后的结果。

## 约束说明

input1是4维张量，shift和dims的长度为2。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import roll
>>> input1 = torch.randn(32, 56, 56, 16).npu()
>>> shift_size = 3
>>> shifted_x_npu = roll(input1, shifts=(-shift_size, -shift_size), dims=(1, 2))
>>> shifted_x_npu.shape
torch.Size([32, 56, 56, 16])
```


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

- **input1** (`Tensor`) - 输入张量。
- **shifts** (`int`或`Tuple` of python:ints) - 张量元素移动的位置数。如果该shift组成的是tuple，则dims必须是大小相同的tuple，每个维度都将按相应的值roll。
- **dims** (`int`或`Tuple` of python:ints) - 沿着roll的轴。

## 返回值说明

`Tensor` - shifted input。


## 调用示例

```python
>>> from torch_npu.contrib.function import roll
>>> input1 = torch.randn(32, 56, 56, 16).npu()
>>> shift_size = 3
>>> shifted_x_npu = roll(input1, shifts=(-shift_size, -shift_size), dims=(1, 2))
```


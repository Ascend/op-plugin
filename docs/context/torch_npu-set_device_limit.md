# torch.npu.set_device_limit
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

- 设置一个进程上指定device，执行算子时所使用的cube和vector核数。


## 函数原型

```
torch.npu.set_device_limit(device, cube_num=-1, vector_num=-1) -> None
```

## 参数说明

- **device** (`Device`)：必选参数，设置的卡号。
- **cube_num** (`int`)：可选参数，设置的cube的核数，默认为-1不设置分核。
- **vector_num** (`int`)：可选参数，设置的vector的核数，默认为-1不设置分核。

## 返回值说明
`None`

代表无返回值。

## 约束说明

- 该接口只支持调用一次，仅对当前进程生效。
- 使用该接口前需要`set_device`.

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_device(0)
>>> torch.npu.set_device_limit(0, 12, 24)
 ```

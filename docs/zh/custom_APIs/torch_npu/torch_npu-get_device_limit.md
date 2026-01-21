# torch.npu.get_device_limit
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

- 通过该接口，获取指定Device上的Device资源限制。
- 当前支持资源类型为Cube Core、Vector Core。

## 函数原型

```
torch.npu.get_device_limit(device) ->Dict
```

## 参数说明

**device** (`Device`)：必选参数，设置控核的卡号。

## 返回值说明
`Dict`

代表`Device`的Cube和Vector核数。

## 约束说明

无

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_device(0)
>>> torch.npu.set_device_limit(0,12,20)
>>> print(torch.npu.get_device_limit(0))
{"cube_core_num":12, "vector_core_num":20}
 ```

# （beta）torch_npu.npu.set_autocast_enabled

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

是否在设备上使能AMP。

## 函数原型

```
torch_npu.npu.set_autocast_enabled(bool)
```

## 参数说明

**bool** ：入参为True时，在设备上使能AMP，否则，不使能AMP。


## 调用示例

```python
>>> torch_npu.npu.set_autocast_enabled(True)
```


# （beta）torch_npu.npu.init_dump

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

初始化dump配置。配置前需要确保环境变量`NPU_DUMP_ENABLE=1`已设置，以及已通过`torch_npu.npu.set_dump_config(path="/tmp/dump", mode="all")`配置dump。


## 函数原型

```
torch_npu.npu.init_dump()
```

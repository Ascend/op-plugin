# （beta）torch_npu.jit.optimize

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

实现jit_mod优化。
## 函数原型

```
torch_npu.jit.optimize(jit_mod)
```



## 参数说明

**jit_mod**：用于被优化的ScriptFunction or ScriptModule。


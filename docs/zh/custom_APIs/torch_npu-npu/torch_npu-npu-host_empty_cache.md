# torch_npu.npu.host_empty_cache

## 产品支持情况

| 产品                                                        | 是否支持 |
| ----------------------------------------------------------- | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas A2 推理系列产品</term> |    √     |
|<term>Atlas 推理系列产品</term> |    √     |
|<term>Atlas 训练系列产品</term> |    √     |


## 功能说明

释放当前由缓存持有的所有未占用的host物理内存。


## 定义文件
torch_npu/npu/memory.py

## 函数原型

```
torch_npu.npu.host_empty_cache()
```

## 参数说明
无

## 返回值说明
无

## 约束说明

无


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty([1024, 1024]).pin_memory()
>>> del x
>>> torch_npu.npu.host_empty_cache()
>>> print(torch_npu.npu.host_memory_stats())
```
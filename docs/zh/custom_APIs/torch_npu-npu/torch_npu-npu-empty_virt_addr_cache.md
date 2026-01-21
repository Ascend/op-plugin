# torch_npu.npu.empty_virt_addr_cache

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

轻量化的缓存释放接口，对应于`torch.npu.empty_cache`。只释放虚拟内存，解除虚拟内存与物理内存的映射，但不真正释放物理内存，从而降低调用耗时。


## 定义文件
torch_npu/npu/memory.py

## 函数原型

```
torch_npu.npu.empty_virt_addr_cache() -> None
```

## 参数说明
无

## 返回值说明
无

## 约束说明

该接口需要环境变量`PYTORCH_NPU_ALLOC_CONF`的值设置为`expandable_segments:True`时才生效，否则会runtime报错。


## 调用示例


```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty((15000, 1024, 1024), device="npu")
>>> del x
>>> torch_npu.npu.empty_virt_addr_cache()
>>> print(torch_npu.npu.memory_summary())
```
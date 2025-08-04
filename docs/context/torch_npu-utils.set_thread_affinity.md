# torch_npu.utils.set_thread_affinity
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

设置当前线程的绑核区间。

## 函数原型

```
torch_npu.utils.set_thread_affinity(core_range: List[int] = None)
```

## 参数说明

 **core_range** (`List[int]`)：可选参数，表示用户期望设置的当前线程绑核区间。默认值为`None`，表示将当前线程作为非主要线程进行自动绑核。


## 返回值说明
无

## 约束说明

该接口需要环境变量`CPU_AFFINITY_CONF`的mode设置为1或2时才生效，一般在拉起子线程的位置前使用，指定子线程的绑核方式或绑核区间。推荐和[torch_npu.utils.reset_thread_affinity](torch_npu-utils.reset_thread_affinity.md)配套使用。


## 调用示例


```python
>>> import torch_npu
>>> import threading
>>>
>>> def run_thread():
...   print("This is a child thread.")
>>>
>>> torch_npu.utils.set_thread_affinity([12, 19])
>>> child_thread = threading.Thread(target=run_thread)
>>> child_thread.start()
>>> torch_npu.utils.reset_thread_affinity()
>>> child_thread.join()
```

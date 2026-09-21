# torch_npu.utils.reset_thread_affinity

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

恢复当前线程的绑核区间为主线程。

## 函数原型

```python
torch_npu.utils.reset_thread_affinity()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

该接口需要环境变量`CPU_AFFINITY_CONF`的mode设置为1或2时才生效，一般在拉起子线程的位置后使用，恢复当前线程的绑核区间为主线程区间。推荐和[torch_npu.utils.set_thread_affinity](torch_npu-utils.set_thread_affinity.md)配套使用。

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

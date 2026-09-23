# torch_npu.utils.set_thread_affinity

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

设置当前线程的绑核区间。

## 函数原型

```python
torch_npu.utils.set_thread_affinity(core_range: list[int] | list[list[int]] | None = None)
```

## 参数说明

 **core_range** (`list[int] | list[list[int]] | None`)：可选参数，表示用户期望设置的当前线程绑核区间，默认值为`None`，表示将当前线程作为非主要线程进行自动绑核。如果用户期望配置绑核区间，绑核区间支持两种格式：

- `list[int]`：为当前线程设置单个绑核区间，例如`[0, 3]`表示用户期望设置的当前线程绑核区间为0、1、2、3号CPU核。
- `list[list[int]]`：为当前线程设置多个绑核区间，例如`[[0, 3], [5, 7]]`表示用户期望设置的当前线程绑核区间为0、1、2、3号和5、6、7号CPU核。

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
>>> torch_npu.utils.set_thread_affinity([12, 19]) # 设置单个绑核区间：[12, 19]
>>> torch_npu.utils.set_thread_affinity([[0, 10], [12, 19]]) # 设置多个绑核区间：[0, 10]和[12, 19]
>>> child_thread = threading.Thread(target=run_thread)
>>> child_thread.start()
>>> torch_npu.utils.reset_thread_affinity()
>>> child_thread.join()
```

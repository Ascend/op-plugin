# torch_npu.profiler.profile.disable_profiler_in_child_thread

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |


## 功能说明

注销Profiler采集回调函数。

与[torch_npu.profiler.profile.enable_profiler_in_child_thread](./torch_npu-profiler-profile-enable_profiler_in_child_thread.md)配对使用。

## 函数原型

```python
torch_npu.profiler.profile.disable_profiler_in_child_thread()
```

## 参数说明

无

## 返回值说明

无

## 调用示例

见[torch_npu.profiler.profile.enable_profiler_in_child_thread](./torch_npu-profiler-profile-enable_profiler_in_child_thread.md)。
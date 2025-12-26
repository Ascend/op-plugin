# torch_npu.profiler._KinetoProfile

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

提供PyTorch训练过程中的性能数据采集功能。

## 函数原型

```
torch_npu.profiler._KinetoProfile(activities=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=None)
```

## 参数说明

- **activities** (`enum`)：可选参数，CPU、NPU事件采集列表。可取值以及含义详见[torch_npu.profiler.ProfilerActivity](torch_npu-profiler-ProfilerActivity.md)。

- **record_shapes** (`bool`)：可选参数，算子的InputShapes和InputTypes。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

     开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **profile_memory** (`bool`)：可选参数，算子的内存占用情况。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

     > [!NOTE]  
     > 已知在安装有glibc<2.34的环境上采集memory数据，可能触发glibc的一个已知[Bug 19329](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)，通过升级环境的glibc版本可解决此问题。

- **with_stack** (`bool`)：可选参数，算子调用栈。包括框架层及CPU算子层的调用信息。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

     开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **with_flops** (`bool`)：可选参数，算子浮点操作（该参数暂不支持解析性能数据）。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

     开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **experimental_config**：可选参数，性能数据采集扩展。支持采集项和详细介绍请参见[torch_npu.profiler._ExperimentalConfig](torch_npu-profiler-_ExperimentalConfig.md)。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

prof = torch_npu.profiler._KinetoProfile(activities=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=None)
for epoch in range(epochs):
        train_model_step()
        if epoch == 0:
                prof.start()
        if epoch == 1:
                prof.stop()
prof.export_chrome_trace("result_dir/trace.json")
```
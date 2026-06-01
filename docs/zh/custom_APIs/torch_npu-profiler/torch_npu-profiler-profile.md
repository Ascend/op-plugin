# torch_npu.profiler.profile

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

提供PyTorch训练过程中的性能数据采集功能。

## 函数原型

```python
torch_npu.profiler.profile(activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_modules, with_flops=False, experimental_config=None)
```

## 参数说明

- **activities** (`enum`)：可选参数，CPU、NPU事件采集列表。可取值以及含义详见[torch_npu.profiler.ProfilerActivity](./torch_npu-profiler-ProfilerActivity.md)。

- **schedule** (`callable`)：可选参数，设置不同step的行为。由[schedule类](./torch_npu-profiler-schedule.md)控制。默认不执行任何操作。

- **on_trace_ready** (`callable`)：可选参数，采集结束时自动执行操作。当前仅支持执行[tensorboard_trace_handler函数](./torch_npu-profiler-tensorboard_trace_handler.md)的操作，默认不执行任何操作。

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

- **with_modules** (`bool`)：可选参数，modules层级的Python调用栈，即框架层的调用信息。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **with_flops** (`bool`)：可选参数，算子浮点操作（该参数暂不支持解析性能数据）。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **experimental_config**：可选参数，扩展参数，通过扩展配置性能分析工具常用的采集项。支持采集项和详细介绍请参见[torch_npu.profiler._ExperimentalConfig](./torch_npu-profiler-_ExperimentalConfig.md)。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

torch_npu.profiler.profile采集的性能数据会自动解析到torch_npu.profiler.tensorboard_trace_handler指定的目录，请参见《[MindStudio Insight系统调优](https://gitcode.com/Ascend/msinsight/blob/26.0.0/docs/zh/user_guide/system_tuning.md)》进行可视化展示与分析。

```python
import torch
import torch_npu

...

# 添加Profiling采集扩展配置参数，详细参数介绍可参考下文的参数说明
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone
)

# 添加Profiling采集基础配置参数，详细参数介绍可参考下文的参数说明
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=0),    # 与prof.step()配套使用
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    profile_memory=False,
    with_modules=False,
    experimental_config=experimental_config) as prof:

    # 启动性能数据采集
    for step in range(steps):    # 训练函数
        train_one_step()    # 训练函数
        prof.step()    # 与schedule配套使用
```

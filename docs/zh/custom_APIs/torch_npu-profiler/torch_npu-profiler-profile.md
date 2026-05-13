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
torch_npu.profiler.profile(activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_modules=False, with_flops=False, experimental_config=None, custom_trace_id_callback=None)
```

## 参数说明

- **activities** (`enum`)：可选参数，CPU、NPU事件采集列表。可取值以及含义详见[torch_npu.profiler.ProfilerActivity](./torch_npu-profiler-ProfilerActivity.md)。

- **schedule** (`callable`)：可选参数，设置不同step的行为。由[schedule类](./torch_npu-profiler-schedule.md)控制。默认不执行任何操作。

- **on_trace_ready** (`callable`)：可选参数，采集结束时自动执行操作。当前仅支持执行[tensorboard_trace_handler函数](./torch_npu-profiler-tensorboard_trace_handler.md)的操作，默认不执行任何操作。

- **record_shapes** (`bool`)：可选参数，记录算子的InputShapes和InputTypes。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **profile_memory** (`bool`)：可选参数，记录算子的内存占用情况。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  > [!NOTE]
  > 已知在安装有glibc<2.34的环境上采集memory数据，可能触发glibc的一个已知[Bug 19329](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)，通过升级环境的glibc版本可解决此问题。

- **with_stack** (`bool`)：可选参数，记录算子调用栈。包括框架层及CPU算子层的调用信息。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **with_modules** (`bool`)：可选参数，记录modules层级的Python调用栈，即框架层的调用信息。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **with_flops** (`bool`)：可选参数，记录算子浮点操作（该参数暂不支持解析性能数据）。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **experimental_config**：可选参数，扩展参数，通过扩展配置性能分析工具常用的采集项。支持采集项和详细介绍请参见[torch_npu.profiler._ExperimentalConfig](./torch_npu-profiler-_ExperimentalConfig.md)。

- **custom_trace_id_callback** (`Callable`)：可选参数，为每一份Profiler数据生成一个trace_id进行标识。trace_id输出在profiler_metadata.json文件中。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

- 采集性能数据完整参数示例

  ```python
  import torch
  import torch_npu

  ...

  experimental_config = torch_npu.profiler._ExperimentalConfig(
      export_type=[
          torch_npu.profiler.ExportType.Text
          ],
      profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
      msprof_tx=False,
      aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
      l2_cache=False,
      op_attr=False,
      data_simplification=False,
      record_op_args=False,
      gc_detect_threshold=None
  )

  with torch_npu.profiler.profile(
      activities=[
          torch_npu.profiler.ProfilerActivity.CPU,
          torch_npu.profiler.ProfilerActivity.NPU
          ],
      schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1, skip_first_wait=1),
      on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
      record_shapes=False,
      profile_memory=False,
      with_stack=False,
      with_modules=False,
      with_flops=False,
      experimental_config=experimental_config) as prof:
      for step in range(steps): # 训练函数
          train_one_step() # 训练函数
          prof.step()
  ```

- 生成标识Profiler数据的trace_id

  ```python
  import torch
  import torch_npu
  ...
  
  # 自定义trace_id生成器
  class RepeatTraceIdGenerator:
      def __init__(self):
          self.repeat_count = 0    #从0开始计数
  
      def __call__(self) -> str:
          # 每一轮profile启动，计数+1
          current_id = self.repeat_count
          self.repeat_count += 1
          return str(current_id)
  
  # 创建trace_id生成器
  trace_id_gen = RepeatTraceIdGenerator()
  
  if __name__ == "__main__":
      device = torch.device('npu:1')
      torch.npu.set_device(device)
      x0 =torch.rand(3, 4).npu()
      x1 =torch.rand(3, 4).npu()
  
  
      stream = torch.npu.current_stream()
      stream.synchronize()
  
  # 添加Profiling采集基础配置参数
  with torch_npu.profiler.profile(
      activities=[
          torch_npu.profiler.ProfilerActivity.CPU,
          torch_npu.profiler.ProfilerActivity.NPU
      ],
      schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=1),
      on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
      custom_trace_id_callback=trace_id_gen
  ) as prof:
      for i in range(12):
          add(x0, x1)  # 训练函数
          prof.step()
  ```

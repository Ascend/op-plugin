# torch_npu.profiler.profile

## 函数原型

```
torch_npu.profiler.profile(activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_modules, with_flops=False, experimental_config=None)
```

## 功能说明

提供PyTorch训练过程中的性能数据采集功能。

## 参数说明

- activities：CPU、NPU事件采集列表，Enum类型。可取值以及含义详见[torch_npu.profiler.ProfilerActivity](torch_npu-profiler-ProfilerActivity.md)。

- schedule：设置不同step的行为，Callable类型。由[schedule类](torch_npu-profiler-schedule.md)控制。默认不执行任何操作。
- on_trace_ready：采集结束时自动执行操作，Callable类型。当前仅支持执行[tensorboard_trace_handler函数](torch_npu-profiler-tensorboard_trace_handler.md)的操作，默认不执行任何操作。
- record_shapes：算子的InputShapes和InputTypes，Bool类型。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

    开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- profile_memory算子的内存占用情况，Bool类型。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

    >**说明：**<br>
    >已知在安装有glibc<2.34的环境上采集memory数据，可能触发glibc的一个已知[Bug 19329](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)，通过升级环境的glibc版本可解决此问题。

- with_stack：算子调用栈，Bool类型。包括框架层及CPU算子层的调用信息。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

    开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- with_modules：modules层级的Python调用栈，即框架层的调用信息，Bool类型。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

    开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- with_flops：算子浮点操作，Bool类型（该参数暂不支持解析性能数据）。取值为：

    - True：开启。
    - False：关闭。

     默认值为False。

    开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- experimental_config：扩展参数，通过扩展配置性能分析工具常用的采集项。支持采集项和详细介绍请参见[torch_npu.profiler._ExperimentalConfig](torch_npu-profiler-_ExperimentalConfig.md)。
- use_cuda：昇腾环境不支持。开启采集cuda性能数据开关，Bool类型。取值为：

    - True：开启。
    - False：关闭。默认值。

     torch_npu.profiler._KinetoProfile不支持该参数。


## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
	export_type=[
		torch_npu.profiler.ExportType.Text,
		torch_npu.profiler.ExportType.Db
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
	schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
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


# torch_npu.profiler._ExperimentalConfig

## 函数原型

```
torch_npu.profiler._ExperimentalConfig(export_type=[torch_npu.profiler.ExportType.Text], profiler_level=torch_npu.profiler.ProfilerLevel.Level0, msprof_tx=False, aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone, l2_cache=False, op_attr=False, data_simplification=False, record_op_args=False, gc_detect_threshold = None)
```

## 功能说明

性能数据采集扩展参数。用于构造torch_npu.profiler.profile的experimental_config参数。

## 参数说明

- export_type：设置导出的性能数据结果文件格式，List类型。可取值以及含义详见[torch_npu.profiler.ExportType](torch_npu-profiler-ExportType.md)。
- profiler_level：采集的Level等级，Enum类型。可取值以及含义详见[torch_npu.profiler.ProfilerLevel](torch_npu-profiler-ProfilerLevel.md)。
- msprof_tx：打点控制开关，通过开关开启自定义打点功能，bool类型。可取值True（开启）或False（关闭），默认关闭。
- data_simplification：数据精简模式，开启后将在导出性能数据后删除FRAMEWORK目录数据以及删除多余数据，仅保留ASCEND_PROFILER_OUTPUT目录和PROF_XXX目录下的原始性能数据，以节省存储空间，bool类型。可取值True（开启）或False（关闭），默认开启。
- aic_metrics：AI Core的性能指标采集项，Enum类型，采集的结果数据将在Kernel View呈现。可取值以及含义详见[torch_npu.profiler.AiCMetrics](torch_npu-profiler-AiCMetrics.md)。
- l2_cache：控制l2_cache数据采集开关，bool类型。可取值True（开启）或False（关闭），默认关闭。该采集项在ASCEND_PROFILER_OUTPUT生成l2_cache.csv文件。
- op_attr：控制采集算子的属性信息开关，当前仅支持aclnn算子，bool类型。可取值True（开启）或False（关闭），默认关闭。该参数采集的性能数据仅支持export_type为torch_npu.profiler.ExportType.Db时解析的db格式文件。
- record_op_args：控制算子信息统计功能开关，bool类型。可取值True（开启）或False（关闭），默认关闭。开启后会在\{worker_name\}_\{时间戳\}_ascend_pt_op_args目录输出采集到算子信息文件。

    >**说明：**<br>
    >该参数在AOE工具执行PyTorch训练场景下调优时使用，且不建议与其他性能数据采集接口同时开启。

- gc_detect_threshold：GC检测阈值，float类型。取值范围为大于等于0的数值，单位ms。当用户设置的阈值为数字时，表示开启GC检测，只采集超过阈值的GC事件。默认为None，表示不开启GC检测功能。配置为0时表示采集所有的GC事件（可能造成采集数据量过大，请谨慎配置），推荐设置为1ms。

    >**说明：**<br>
    >- **GC**是Python进程对已经销毁的对象进行内存回收。
    >- 解析结果文件格式配置为torch_npu.profiler.ExportType.Text时，则在解析结果数据trace_view.json中生成GC层。
    >- 解析结果文件格式配置为torch_npu.profiler.ExportType.Db时，则在ascend_pytorch_profiler_\{rank_id\}.db中生成GC_RECORD表。可通过MindStudio Insight工具查看。

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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```


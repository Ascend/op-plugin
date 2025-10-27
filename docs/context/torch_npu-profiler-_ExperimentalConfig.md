# torch_npu.profiler._ExperimentalConfig

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

性能数据采集扩展参数。用于构造torch_npu.profiler.profile的experimental_config参数。

## 函数原型

```
torch_npu.profiler._ExperimentalConfig(export_type=[torch_npu.profiler.ExportType.Text], profiler_level=torch_npu.profiler.ProfilerLevel.Level0, mstx=False, mstx_domain_include=[], mstx_domain_exclude=[], aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone, l2_cache=False, op_attr=False, data_simplification=True, record_op_args=False, gc_detect_threshold=None, host_sys=[], sys_io=False, sys_interconnection=False)
```

## 参数说明

- **export_type** (`list`)：可选参数，设置导出的性能数据结果文件格式。可取值以及含义详见[torch_npu.profiler.ExportType](torch_npu-profiler-ExportType.md)。

- **profiler_level** (`Enum`)：可选参数，采集的Level等级。可取值以及含义详见[torch_npu.profiler.ProfilerLevel](torch_npu-profiler-ProfilerLevel.md)。

- **mstx或msprof_tx** (`bool`)：可选参数，打点控制开关，通过开关开启自定义打点功能。取值为：

    - True：开启。
    - False：关闭。

    默认值为False。

    原参数名称msprof_tx改名为mstx，新版本依旧兼容原参数名。

- **mstx_domain_include** (`str`)：可选参数，输出需要的domain数据。调用torch_npu.npu.mstx系列打点接口，使用默认domain或指定domain进行打点时，可选择只输出本参数配置的domain数据。

    domain名称为用户调用[torch_npu.npu.mstx](torch_npu-npu-mstx.md)系列接口传入的domain或默认domain（'default'），domain名称使用list类型输入。

    与mstx_domain_exclude参数互斥，若同时配置，则只有mstx_domain_include生效。

    须配置mstx=True。

- **mstx_domain_exclude** (`str`)：可选参数，过滤不需要的domain数据。调用torch_npu.npu.mstx系列打点接口，使用默认domain或指定domain进行打点时，可选择不输出本参数配置的domain数据。

    domain名称为用户调用torch_npu.npu.mstx系列接口传入的domain或默认domain（'default'），domain名称使用list类型输入。

    与mstx_domain_include参数互斥，若同时配置，则只有mstx_domain_include生效。

    须配置mstx=True。

- **aic_metrics** (`Enum`)：可选参数，AI Core的性能指标采集项，采集的结果数据将在Kernel View呈现。可取值以及含义详见[torch_npu.profiler.AiCMetrics](torch_npu-profiler-AiCMetrics.md)。

- **l2_cache** (`bool`)：可选参数，控制l2_cache数据采集开关。该采集项在ASCEND_PROFILER_OUTPUT生成l2_cache.csv文件。取值为：

    - True：开启。
    - False：关闭。

    默认值为False。

- **op_attr** (`bool`)：可选参数，控制采集算子的属性信息开关，当前仅支持aclnn算子。取值为：

    - True：开启。
    - False：关闭。

    默认值为False。

    torch_npu.profiler.ProfilerLevel.None时，该参数不生效。

- **data_simplification** (`bool`)：可选参数，开启后将在导出性能数据后删除多余数据，仅保留profiler_*.json文件、ASCEND_PROFILER_OUTPUT目录、PROF_XXX目录下的原始性能数据、FRAMEWORK目录和logs目录，以节省存储空间。取值为：

    - True：开启。
    - False：关闭。

    默认值为True。

- **record_op_args** (`bool`)：可选参数，控制算子信息统计功能开关。开启后会在\{worker_name\}\_\{时间戳\}_ascend_pt_op_args目录输出采集到算子信息文件。取值为：

    - True：开启。
    - False：关闭。

    默认值为False。

- **gc_detect_threshold** (`float`)：可选参数，GC检测阈值。取值范围为大于等于0的数值，单位ms。当用户设置的阈值为数字时，表示开启GC检测，只采集超过阈值的GC事件。配置为0时表示采集所有的GC事件（可能造成采集数据量过大，请谨慎配置），推荐设置为1ms。

    默认为None，表示不开启GC检测功能。
    
    - GC是Python进程对已经销毁的对象进行内存回收。
    - GC层在trace_view.json中生成或在ascend_pytorch_profiler_{Rank_ID}.db中生成GC_RECORD表。
    
- **host_sys** (`list`)：可选参数，Host侧系统数据采集开关。默认未配置，表示未开启Host侧系统数据采集。取值为：

    - torch_npu.profiler.HostSystem.CPU：进程级别的CPU利用率。
    - torch_npu.profiler.HostSystem.MEM：进程级别的内存利用率。
    - torch_npu.profiler.HostSystem.DISK：进程级别的磁盘I/O利用率。
    - torch_npu.profiler.HostSystem.NETWORK：系统级别的网络I/O利用率。
    - torch_npu.profiler.HostSystem.OSRT：进程级别的syscall和pthreadcall。

- **sys_io** (`bool`)：可选参数，NIC、ROCE、MAC采集开关。取值为：

    - True：开启。
    - False：关闭。

    默认值为False。

- **sys_interconnection** (`bool`)：可选参数，集合通信带宽数据（HCCS）、PCIe数据采集开关、片间传输带宽信息采集开关。取值为：

    - True：开启。
    - False：关闭。

    默认值为False。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```
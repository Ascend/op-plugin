# torch_npu.profiler.ProfilerLevel

## 函数原型

```
torch_npu.profiler.ProfilerLevel
```

## 功能说明

采集等级，Enum类型。用于作为_ExperimentalConfig类的profiler_level参数。

## 参数说明

- Level_none：不采集所有Level层级控制的数据，即关闭profiler_level。

- Level0：采集上层应用数据、底层NPU数据以及NPU上执行的算子信息。配置该参数时，仅采集部分数据，其中部分算子信息不采集，详细情况请参见《CANN 性能调优工具用户指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0067.html">op_summary（算子详细信息）</a>”章节。
- Level1：在Level0的基础上多采集CANN层AscendCL数据和NPU上执行的AI Core性能指标信息、开启aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization以及HCCL的communication.json和communication_matrix.json文件。
- Level2：在Level1的基础上多采集CANN层Runtime数据以及AI CPU（data_preprocess.csv文件）。

默认值为Level0。

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
       profiler_level=torch_npu.profiler.ProfilerLevel.Level0
       )
with torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```


# torch_npu.profiler.AiCMetrics

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

AI Core的性能指标采集项，Enum类型。用于作为_ExperimentalConfig类的aic_metrics参数。

## 函数原型

```
torch_npu.profiler.AiCMetrics
```

## 参数说明

以下采集项的结果数据含义可参见《CANN 性能调优工具用户指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0067.html">op_summary（算子详细信息）</a>”章节，但具体采集结果请以实际情况为准。

- **torch_npu.profiler.AiCMetrics.AiCoreNone**：可选参数，关闭AI Core的性能指标采集。
- **torch_npu.profiler.AiCMetrics.PipeUtilization**：可选参数，计算单元和搬运单元耗时占比。
- **torch_npu.profiler.AiCMetrics.ArithmeticUtilization**：可选参数，各种计算类指标占比统计。
- **torch_npu.profiler.AiCMetrics.Memory**：可选参数，外部内存读写类指令占比。
- **torch_npu.profiler.AiCMetrics.MemoryL0**：可选参数，内部内存读写类指令占比。
- **torch_npu.profiler.AiCMetrics.ResourceConflictRatio**：可选参数，流水线队列类指令占比。
- **torch_npu.profiler.AiCMetrics.MemoryUB**：可选参数，内部内存读写指令占比。
- **torch_npu.profiler.AiCMetrics.L2Cache**：可选参数，读写cache命中次数和缺失后重新分配次数。
- **torch_npu.profiler.AiCMetrics.MemoryAccess**：可选参数，算子在核上访存的带宽数据量。

默认值为torch_npu.profiler.AiCMetrics.AiCoreNone。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       aic_metrics = torch_npu.profiler.AiCMetrics.AiCoreNone
       )
with torch_npu.profiler.profile(
        on_trace_ready = torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config = experimental_config) as prof:
        for step in range(steps): # 训练函数
                 train_one_step() # 训练函数
                 prof.step()
```
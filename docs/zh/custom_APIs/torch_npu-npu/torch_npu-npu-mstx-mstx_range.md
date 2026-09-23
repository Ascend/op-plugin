# torch_npu.npu.mstx.mstx_range

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

mstx_range装饰器，用来采集被装饰函数的range执行耗时。

## 函数原型

```python
torch_npu.npu.mstx.mstx_range(msg: str, stream=None, domain: str='default')
```

## 参数说明

- **msg** (`str`)：必选参数，打点携带信息的字符串。
- **stream** (`torch_npu.npu.Stream`)：可选参数，用于执行打点任务的stream，默认为None。
  - 配置为None时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内，标识时间段事件。默认为'default'，表示默认domain，不设置也为默认domain。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu

@torch_npu.npu.mstx.mstx_range("train_one_step")
def train_one_step(step, steps, train_loader, model, optimizer, criterion):
    # 函数代码
    pass

experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level_none,
    mstx=True,
    export_type=[
        torch_npu.profiler.ExportType.Db
        ])
with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    experimental_config=experimental_config) as prof:
       
    for epoch in range(epochs):
        for step in range(steps):
            train_one_step(step, steps, train_loader, model, optimizer, criterion)
            prof.step()
```

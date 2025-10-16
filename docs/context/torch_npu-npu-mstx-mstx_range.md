#  torch_npu.npu.mstx.mstx_range

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

range装饰器，用来采集被装饰函数的range执行耗时。

## 函数原型

```
torch_npu.npu.mstx.mstx_range(msg: str, stream=None, domain: str='default')
```

## 参数说明

- **msg** (`str`)：可选参数，打点携带信息字符串指针。
- **stream** (`int`)：可选参数，用于执行打点任务的stream。
  - 配置为nullptr时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内，标识时间段事件。默认为default，表示默认domain，不设置也为默认domain。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

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
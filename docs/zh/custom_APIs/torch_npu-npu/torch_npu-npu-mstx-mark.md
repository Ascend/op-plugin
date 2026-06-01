# torch_npu.npu.mstx.mark

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

标记瞬时事件。

## 函数原型

```python
torch_npu.npu.mstx.mark(message: str='None', stream=None, domain: str='default') -> None
```

## 参数说明

- **message** (`str`)：可选参数，打点携带信息的字符串，默认为'None'。传入的message字符串长度要求：
  - msPTI场景：不能超过255字节。
  - msProf或torch_npu.profiler采集场景：不能超过156字节。
- **stream** (`torch_npu.npu.Stream`)：可选参数，用于执行打点任务的stream，默认为None。
  - 配置为None时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
- x import torchimport torch_npu​@torch_npu.npu.mstx.mstx_range("train_one_step")def train_one_step(step, steps, train_loader, model, optimizer, criterion):    # 函数代码    pass​experimental_config = torch_npu.profiler._ExperimentalConfig(    profiler_level=torch_npu.profiler.ProfilerLevel.Level_none,    mstx=True,    export_type=[        torch_npu.profiler.ExportType.Db        ])with torch_npu.profiler.profile(    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=1),    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),    experimental_config=experimental_config) as prof:           for epoch in range(epochs):        for step in range(steps):            train_one_step(step, steps, train_loader, model, optimizer, criterion)            prof.step()python

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

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
       
    for step in range(steps):
        train_one_step()    # 用户代码，包含调用mstx接口
        prof.step()
```

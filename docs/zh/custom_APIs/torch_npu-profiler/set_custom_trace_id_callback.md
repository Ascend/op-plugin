# set_custom_trace_id_callback

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

设置生成trace_id的回调。

该接口的功能可通过[torch_npu.profiler.profile](torch_npu-profiler-profile.md)接口的custom_trace_id_callback参数直接配置，建议使用custom_trace_id_callback参数。

## 函数原型

```python
set_custom_trace_id_callback(self, callback)
```

## 参数说明

- **callback** (`Callable`)：必选参数，设置自定义回调函数。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

性能数据采集完成后，trace_id输出在profiler_metadata.json文件中。

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
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    )

    prof.set_custom_trace_id_callback(trace_id_gen)    # 设置生成trace_id的回调
    prof.start()    # 启动性能数据采集
    for i in range(12):    # 训练函数
        add(x0, x1)    # 训练函数
        prof.step()    # 与schedule配套使用
        print(f"step {i}: {prof.get_trace_id()}")
    prof.stop()    # 结束性能数据采集
```

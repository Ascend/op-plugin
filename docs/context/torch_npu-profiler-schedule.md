# torch_npu.profiler.schedule

## 函数原型

```
torch_npu.profiler.schedule (wait, active, warmup = 0, repeat = 0, skip_first = 0)
```

## 功能说明

设置不同step的行为。用于构造torch_npu.profiler.profile的schedule参数。

## 参数说明

- wait：每次重复执行采集跳过的step轮数，int类型。必选。
- active：采集的step轮数，int类型。必选。
- warmup：预热的step轮数，int类型。默认值为0。建议设置1轮预热。可选。
- repeat：重复执行wait+warmup+active的次数，int类型。默认值为0，表示重复执行repeat不停止，建议配置为大于0的整数。可选。
- skip_first：采集前先跳过的step轮数，int类型。默认值为0。动态Shape场景建议跳过前10轮保证性能数据稳定；对于其他场景，可以根据实际情况自行配置。可选。

默认不执行schedule操作。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu

...

with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```


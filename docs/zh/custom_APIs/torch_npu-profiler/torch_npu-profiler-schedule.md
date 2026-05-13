# torch_npu.profiler.schedule

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

设置不同step的行为。用于构造torch_npu.profiler.profile的schedule参数。默认不执行schedule操作。

## 函数原型

```python
torch_npu.profiler.schedule (wait, active, warmup = 0, repeat = 0, skip_first = 0, skip_first_wait = 0)
```

## 参数说明

- **wait** (`int`)：必选参数，每次重复执行采集前跳过的step轮数。

- **active** (`int`)：必选参数，采集的step轮数。

- **warmup** (`int`)：可选参数，预热的step轮数。默认值为0。建议设置1轮预热。

- **repeat** (`int`)：可选参数，重复执行wait+warmup+active的次数。取值范围为大于等于0的整数，默认值为0。

  当使用集群分析工具或MindStudio Insight查看时，建议配置repeat = 1（表示执行1次，仅生成一份性能数据），因为：

  - repeat > 1，则会在同一目录下生成多份性能数据，此时需要手动将采集的性能数据文件夹分为repeat等份，放到不同文件夹下重新解析，分类方式按照文件夹名称中的时间戳先后。
  - repeat = 0时，重复执行的具体次数由总训练步数决定，例如总训练步数为100，wait + active + warmup = 10，skip_first = 10，则repeat = ( 100 - 10 ) / 10 = 9，表示重复执行9次，生成9份性能数据。

- **skip_first** (`int`)：可选参数，采集前先跳过的step轮数。默认值为0。动态Shape场景建议跳过前10轮保证性能数据稳定；对于其他场景，可以根据实际情况自行配置。

- **skip_first_wait** (`int`)：可选参数，采集时跳过第一个wait。默认值为0，表示关闭该参数功能，配置为非0的int值时，表示开启该参数功能。<br>开启该参数功能时，会取消wait参数的第一次wait，即在第一个repeat中的第一个step直接执行下一步动作，但第二个repeat之后仍正常执行。可用于节省采集的时间。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=1,                        # 等待阶段，跳过1个step
        warmup=1,                      # 预热阶段，跳过1个step
        active=2,                      # 记录2个step的活动数据，并在之后调用on_trace_ready
        repeat=2,                      # 循环wait+warmup+active过程2遍
        skip_first=1,                  # 跳过1个step
        skip_first_wait=1            # 跳过第一个wait
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./result')
    ) as prof:
        for _ in range(9):
            train_one_step()
            prof.step()                # 通知profiler完成一个step
```

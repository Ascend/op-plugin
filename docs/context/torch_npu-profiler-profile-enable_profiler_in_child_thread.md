# torch_npu.profiler.profile.enable_profiler_in_child_thread

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

注册Profiler采集回调函数，采集用户子线程下发的torch算子等框架侧数据。该参数中可另外配置[torch_npu.profiler.profile](./torch_npu-profiler-profile.md)的参数（包括record_shapes、profile_memory、with_stack、with_flops、with_modules），作为Profiler子线程的采集配置。

与[torch_npu.profiler.profile.disable_profiler_in_child_thread](./torch_npu-profiler-profile-disable_profiler_in_child_thread.md)配对使用。

## 函数原型

```python
torch_npu.profiler.profile.enable_profiler_in_child_thread(record_shapes=False, profile_memory=False, with_stack=False, with_modules=False, with_flops=False)
```

## 参数说明

- **record_shapes**(Bool)：可选参数，算子的InputShapes和InputTypes。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **profile_memory**(Bool)：可选参数，算子的内存占用情况。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  > **说明：**
  > 已知在安装有glibc<2.34的环境上采集memory数据，可能触发glibc的一个已知[Bug 19329](https://gitee.com/link?target=https%3A%2F%2Fsourceware.org%2Fbugzilla%2Fshow_bug.cgi%3Fid%3D19329)，通过升级环境的glibc版本可解决此问题。

- **with_stack**(Bool)：可选参数，算子调用栈。包括框架层及CPU算子层的调用信息。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **with_modules**(Bool)：可选参数，modules层级的Python调用栈，即框架层的调用信息。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

- **with_flops**(Bool)：可选参数，算子浮点操作（该参数暂不支持解析性能数据）。取值为：

  - True：开启。
  - False：关闭。

  默认值为False。

  开启torch_npu.profiler.ProfilerActivity.CPU时生效。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import threading
import random
import torch
import torch_npu

# 推理模型定义
...

def infer(device, child_thread):
    torch.npu.set_device(device)

    if child_thread:
        # 开始采集子线程的torch算子等框架侧数据
        torch_npu.profiler.profile.enable_profiler_in_child_thread(with_modules=True)
    
    for _ in range(5):
        outputs = model(input_data)

    if child_thread:
        # 停止采集子线程的torch算子等框架侧数据
        torch_npu.profiler.profile.disable_profiler_in_child_thread()


if __name__ == "__main__":
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )

    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_multi", analyse_flag=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=True,
        experimental_config=experimental_config)

    prof.start()

    threads = []
    for i in range(1, 3):
        # 创建2个子线程，分别在device1与device2上进行推理任务
        t = threading.Thread(target=train, args=(i, True))
        t.start()
        threads.append(t)

    # 主线程在device0上运行推理任务
    train(0, False)

    for t in threads:
        t.join()

    prof.stop()
```
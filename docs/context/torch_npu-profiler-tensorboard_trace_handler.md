# torch_npu.profiler.tensorboard_trace_handler

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

将采集到的性能数据导出为TensorBoard工具支持的格式。作为torch_npu.profiler.profile on_trace_ready参数的执行操作。

## 函数原型

```
torch_npu.profiler.tensorboard_trace_handler(dir_name=None, worker_name=None, analyse_flag=True, async_mode=False)
```

## 参数说明

- **dir_name** (`str`)：可选参数，采集到的性能数据的存放路径。路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。若配置tensorboard_trace_handler函数后未指定具体路径，性能数据默认落盘在当前目录；若代码中未使用on_trace_ready=torch_npu.profiler.tensorboard_trace_handler，那么落盘的性能数据为原始数据，需要使用[离线解析](./torch_npu-profiler-profiler-analyse.md)。

    该函数优先级高于ASCEND_WORK_PATH环境变量。

- **worker_name** (`str`)：可选参数，用于区分唯一的工作线程，默认为\{hostname\}_\{pid\}。路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。

- **analyse_flag** (`bool`)：可选参数，性能数据自动解析开关。取值为：

    - True：开启自动解析。
    - False：关闭自动解析，采集完后的性能数据可以使用[离线解析](./torch_npu-profiler-profiler-analyse.md)。

    默认值为True。

- **async_mode** (`bool`)：可选参数，控制是否开启异步解析（表示解析进程不会阻塞AI任务主流程）。取值为：

    - True：开启异步解析。
    - False：关闭异步解析，即同步解析。

    默认值为False。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

with torch_npu.profiler.profile(
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # 训练函数
                train_one_step()  # 训练函数
                prof.step()
```
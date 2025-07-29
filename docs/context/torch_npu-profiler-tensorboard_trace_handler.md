# torch_npu.profiler.tensorboard_trace_handler

## 函数原型

```
torch_npu.profiler.tensorboard_trace_handler(dir_name=None, worker_name=None, analyse_flag=True)
```

## 功能说明

将采集到的性能数据导出为TensorBoard工具支持的格式。作为torch_npu.profiler.profile on_trace_ready参数的执行操作。

## 参数说明

- dir_name：采集的性能数据的输出目录，string类型。路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。若配置tensorboard_trace_handler函数后未指定具体路径，性能数据默认落盘在当前目录。可选。
- worker_name：用于区分唯一的工作线程，string类型，默认为\{hostname\}_\{pid\}。路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。可选。
- analyse_flag：性能数据自动解析开关，bool类型。可取值True（开启自动解析，默认值）、False（关闭自动解析，采集完后的性能数据可以使用离线解析）。可选。

    离线解析详见《CANN 性能调优工具用户指南》中的“
<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html">Ascend PyTorch Profiler</a>”章节。

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
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # 训练函数
                train_one_step()  # 训练函数
                prof.step()
```


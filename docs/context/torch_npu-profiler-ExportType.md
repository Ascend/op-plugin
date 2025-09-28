# torch_npu.profiler.ExportType

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

设置导出的性能数据结果文件格式，List类型。作为_ExperimentalConfig类的export_type参数。

## 函数原型

```
torch_npu.profiler.ExportType
```

## 参数说明

- **torch_npu.profiler.ExportType.Text**：可选参数，表示解析为.json和.csv格式的timeline和summary文件以及汇总所有性能数据的.db格式文件（ascend_pytorch_profiler_{Rank_ID}.db、analysis.db）。
- **torch_npu.profiler.ExportType.Db**：可选参数，表示仅解析为一个汇总所有性能数据的.db格式文件（ascend_pytorch_profiler_{Rank_ID}.db、analysis.db），使用Ascend Insight工具展示。仅支持on_trace_ready接口导出和离线解析导出，需配套安装支持导出db格式的Toolkit软件包。

设置无效值或未配置均取默认值torch_npu.profiler.ExportType.Text。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       export_type=[
              torch_npu.profiler.ExportType.Text
              ],
       )
with torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```
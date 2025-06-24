# torch_npu.profiler.ExportType

## 函数原型

```
torch_npu.profiler.ExportType
```

## 功能说明

设置导出的性能数据结果文件格式，List类型。作为_ExperimentalConfig类的export_type参数。

## 参数说明

- torch_npu.profiler.ExportType.Text：表示解析为.json和.csv格式的timeline和summary文件。
- torch_npu.profiler.ExportType.Db：表示解析为一个汇总所有性能数据的.db格式文件（ascend_pytorch.db、analysis.db），使用Ascend Insight工具展示。仅支持on_trace_ready接口导出和离线解析导出，需配套安装支持导出db格式的Toolkit软件包Ascend-cann-toolkit开发套件包，即CANN 8.0.RC1及以上版本。

设置无效值或未配置均取默认值torch_npu.profiler.ExportType.Text。两个参数可同时配置，表示同时导出timeline、summary和db文件。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       export_type=[
              torch_npu.profiler.ExportType.Text,
              torch_npu.profiler.ExportType.Db
              ],
       )
with torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```


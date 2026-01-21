# torch_npu.profiler.profiler.analyse

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

提供对Ascend PyTorch Profiler采集的性能数据进行离线解析的功能。

## 函数原型

```
torch_npu.profiler.profiler.analyse(profiler_path="", max_process_number=max_process_number, export_type=export_type)
```

## 参数说明

- **profiler_path** (`str`)：必选参数，PyTorch性能数据路径。路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。指定的目录下保存PyTorch性能数据目录\{worker_name\}\_\{时间戳\}_ascend_pt。

- **max_process_number** (`int`)：可选参数，离线解析最大进程数。取值范围为1\~CPU核数，默认为CPU核数的一半。若设置超过该环境的CPU核数，则自动取CPU核数；若设置为非法值，则取默认值CPU核数的一半。

- **export_type** (`list`)：可选参数，设置导出的性能数据结果文件格式。取值为：

  - text：表示解析为.json和.csv格式的timeline和summary文件以及汇总所有性能数据的.db格式文件（ascend_pytorch.db、analysis.db）。
  - db：表示仅解析为一个汇总所有性能数据的.db格式文件（ascend_pytorch.db、analysis.db），使用MindStudio Insight工具展示。仅支持on_trace_ready接口导出和[离线解析](./torch_npu-profiler-profiler-analyse.md)导出，需配套安装支持导出db格式的Toolkit软件包。

  设置无效值或未配置时，则读取profiler_info.json中的export_type字段，确定导出格式。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

创建_\{file_name\}_.py文件，_\{file_name\}_自定义，并编辑如下代码：

```python
from torch_npu.profiler.profiler import analyse

if __name__ == "__main__":
	analyse(profiler_path="./result_data", max_process_number=max_process_number)
```
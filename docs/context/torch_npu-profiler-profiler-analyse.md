# torch_npu.profiler.profiler.analyse

## 函数原型

```
torch_npu.profiler.profiler.analyse(profiler_path="", max_process_number=max_process_number)
```

## 功能说明

提供对Ascend PyTorch Profiler采集的性能数据进行离线解析。

## 参数说明

- profiler_path：PyTorch性能数据路径。路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。指定的目录下保存PyTorch性能数据目录\{worker_name\}_\{时间戳\}_ascend_pt。必选。

- max_process_number：离线解析最大进程数。取值范围为1\~CPU核数，默认为CPU核数的一半。若设置超过该环境的CPU核数，则自动取CPU核数；若设置为非法值，则取默认值CPU核数的一半。可选。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

创建_\{file_name\}_.py文件，_\{file_name\}_自定义，并编辑如下代码：

```python
from torch_npu.profiler.profiler import analyse

if __name__ == "__main__":
	analyse(profiler_path="./result_data", max_process_number=max_process_number)
```


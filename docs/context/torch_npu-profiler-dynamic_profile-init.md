# torch_npu.profiler.dynamic_profile.init

## 函数原型

```
torch_npu.profiler.dynamic_profile.init(path: str)
```

## 功能说明

初始化dynamic_profile动态采集。

## 参数说明

path：dynamic_profile会在path下自动创建模板文件profiler_config.json，用户可基于模板文件自定义修改配置项。profiler_config_path路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。必选。

profiler_config.json文件详细介绍请参见《CANN 性能调优工具用户指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html">Ascend PyTorch Profiler</a>”章节。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
# 加载dynamic_profile模块
from torch_npu.profiler import dynamic_profile as dp
# 设置Profiling配置文件的路径
dp.init("profiler_config_path")
…
for step in steps:
	train_one_step()
	# 划分step
	dp.step()
```


# torch_npu.profiler.dynamic_profile.start

## 函数原型

```
torch_npu.profiler.dynamic_profile.start(config_path: str = None)
```

## 功能说明

触发一次dynamic_profile动态采集。

## 参数说明

config_path：指定为profiler_config.json，需要用户手动创建profiler_config.json配置文件并根据场景需要配置参数。此处须指定具体文件名，例如start("./home/xx/start_config_path/profiler_config.json")。profiler_config_path和start_config_path路径格式仅支持由字母、数字和下划线组成的字符串，不支持软链接。可选。

profiler_config.json文件详细介绍请参见《CANN 性能调优工具用户指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html">Ascend PyTorch Profiler</a>”章节。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
# 加载dynamic_profile模块
from torch_npu.profiler import dynamic_profile as dp
# 设置init接口Profiling配置文件路径
dp.init("profiler_config_path")
…
for step in steps:
	if step==5:
		# 设置start接口Profiling配置文件路径
		dp.start("start_config_path")
	train_one_step()
	# 划分step，需要进行profile的代码需在dp.start()接口和dp.step()接口之间
	dp.step()
```


# torch_npu.profiler.dynamic_profile.step

## 函数原型

```
torch_npu.profiler.dynamic_profile.step()
```

## 功能说明

dynamic_profile动态采集划分step。

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


# torch_npu.profiler.dynamic_profile.step

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

在dynamic_profile动态采集时划分step。

## 函数原型

```
torch_npu.profiler.dynamic_profile.step()
```

## 参数说明

无

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

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
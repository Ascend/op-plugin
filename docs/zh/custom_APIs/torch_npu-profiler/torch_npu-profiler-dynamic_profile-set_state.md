# torch_npu.profiler.dynamic_profile.set_state

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

动态采集时，设置训练已执行到的step步数。

训练未发生断点时，动态采集从step0开始计数，直到训练执行到相应step时启动采集；训练执行过程中发生断点，重新启动训练时，若未配置本接口，则动态采集默认以重新启动训练的位置为step0开始计数，可能导致训练结束还未启动采集。本接口可用于在训练断点时，手动设置训练已执行到的step步数，令动态采集从设置的step步数开始计数。

## 函数原型

```python
torch_npu.profiler.dynamic_profile.set_state(state_step: dict)
```

## 参数说明

**state_step** (`dict`)：可选参数，设置训练已执行到的step步数，需要用户手动根据实际情况设置需要的参数。取值范围为大于等于0的整数，默认值为0。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
# 加载dynamic_profile模块
from torch_npu.profiler import dynamic_profile as dp
# 设置训练已执行到的step步数
dp.set_state({"cur_step": 10})
dp.init("/data/test")
for t in range(50):
    if t <= 10:    # 模拟训练断点时，训练已执行到的step步数
       continue
    train_one_step()
    # 划分step，需要进行profile的代码需在dp.start()接口和dp.step()接口之间
    dp.step()
```

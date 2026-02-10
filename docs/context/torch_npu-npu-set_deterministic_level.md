# torch_npu.npu.set_deterministic_level

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term>                        |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

该接口用于控制CANN侧强一致性功能。具体为重新配置CANN侧参数，参数详细说明可见：[aclSysParamOpt](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_1393.html)。

实际level对应配置如下表所示：
| torch_npu.npu.set_deterministic_level配置 | 调用aclrtSetSysParamOpt配置 | 实际功能 | 与原生接口torch.use_deterministic_algoritms的对应关系 |
| :---------------------------------------- | :-------------------------------------------------------------------------------------------- | :--------- | :------------------------|
| 0                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | 关闭确定性 | torch.use_deterministic_algorithms(False) |
| 1                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | 仅开启确定性 | 关闭确定性 | torch.use_deterministic_algorithms(True) |
| 2                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | 开启强一致性 | 用于解决部分场景下，某些算子因为累加顺序不一致而导致结果不一致的问题 |

开启强一致性计算后，会导致算子执行速度变慢，影响性能。仅建议在需要严格保证不同位置上的相同数据计算结果一致，或进行精度调优时，才开启强一致性计算，以辅助模型调试和优化。

## 函数原型

```
torch_npu.npu.set_deterministic_level(level)
```

## 参数说明

**level**(`int`)：所配置的确定性等级，目前仅可配置为0/1/2，默认值为0。

## 返回值说明

无

## 约束说明

- 该功能当前仅影响`torch.mm`计算结果。

- 该接口不支持动态修改，在算子下发与执行过程中更改配置，可能会引发未知错误。

- 调用该接口时，会根据所配置的`level`覆盖之前`torch.use_deterministic_algorithms`的配置。

- 调用该接口后，不应该再调用`torch.use_deterministic_algorithms`来修改确定性配置。

- 仅支持8.5.0及以上的CANN版本。

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_deterministic_level(2)
```
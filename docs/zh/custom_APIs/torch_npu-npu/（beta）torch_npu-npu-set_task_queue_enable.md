# （beta）torch\_npu.npu.set\_task\_queue\_enable

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  |    √     |
|<term>Atlas 800I A2 推理产品</term> |    √     |
|<term>Atlas 推理系列产品</term> |    √     |
|<term>Atlas 训练系列产品</term> |    √     |
|<term>Ascend 950DT</term> |    √     |

## 功能说明

该接口用于在运行时动态设置TaskQueue优化等级，无需重启进程。

模式取值与`TASK_QUEUE_ENABLE`环境变量一致：

- 0：关闭TaskQueue优化。
- 1：Level 1优化，使用TaskQueue进行算子下发。
- 2：Level 2优化，基于固定内存池进行二级流水线。

模式生效优先级为：`ASCEND_LAUNCH_BLOCKING=1` > 本接口设置值 > `TASK_QUEUE_ENABLE`环境变量（默认值为1）。

## 函数原型

```python
torch_npu.npu.set_task_queue_enable(mode)
```

## 参数说明

**mode**（`int`）：TaskQueue优化等级，取值为0、1、2，传入其他值时抛出RuntimeError。

## 调用示例

```python
import torch
import torch_npu

# 查询当前模式
print(torch_npu.npu.get_task_queue_enable())  # 1（默认值）

# 切换至Level 2优化
torch_npu.npu.set_task_queue_enable(2)
print(torch_npu.npu.get_task_queue_enable())  # 2

# 关闭TaskQueue优化
torch_npu.npu.set_task_queue_enable(0)
print(torch_npu.npu.get_task_queue_enable())  # 0
```

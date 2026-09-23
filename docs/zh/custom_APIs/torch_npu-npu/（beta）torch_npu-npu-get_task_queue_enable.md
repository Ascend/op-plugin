# （beta）torch\_npu.npu.get\_task\_queue\_enable

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id6 -->
- <term>Atlas 800I A2推理服务器</term>：支持
<!-- end id6 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id5 -->

## 功能说明

该接口用于查询当前生效的TaskQueue优化等级，返回值为`torch_npu.npu.set_task_queue_enable`或环境变量`TASK_QUEUE_ENABLE`当前生效的模式。

`ASCEND_LAUNCH_BLOCKING=1`为同步执行模式（故障定位用），与TaskQueue的异步流水线机制互斥。该模式开启时，无论TaskQueue模式设置为何值，本接口恒返回0。

## 函数原型

```python
torch_npu.npu.get_task_queue_enable()
```

## 返回值说明

**int**：当前生效的TaskQueue优化等级，含义如下：

- 0：TaskQueue优化关闭。
- 1：Level 1优化。
- 2：Level 2优化。

## 调用示例

```python
import torch
import torch_npu

# 查询当前模式（未设置时默认为1）
print(torch_npu.npu.get_task_queue_enable())  # 1

# 动态切换后查询
torch_npu.npu.set_task_queue_enable(2)
print(torch_npu.npu.get_task_queue_enable())  # 2
```

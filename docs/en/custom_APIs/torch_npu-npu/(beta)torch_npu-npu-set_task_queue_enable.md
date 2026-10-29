# (beta) torch_npu.npu.set_task_queue_enable

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id6 -->
- <term>Atlas 800I A2</term>: Supported
<!-- end id6 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id5 -->

## Function

This API dynamically sets the TaskQueue optimization level at runtime without restarting the process.

The mode values are the same as those of the `TASK_QUEUE_ENABLE` environment variable:

- 0: TaskQueue optimization is disabled.
- 1: Level 1 optimization, which uses TaskQueue for operator dispatch.
- 2: Level 2 optimization, which uses a fixed memory pool for a two-level pipeline.

The priority of the modes is as follows: `ASCEND_LAUNCH_BLOCKING=1` > the value set by this API > the `TASK_QUEUE_ENABLE` environment variable (default value: 1).

## Prototype

```python
torch_npu.npu.set_task_queue_enable(mode)
```

## Parameters

**`mode`** (`int`): TaskQueue optimization level. The value can be 0, 1, or 2. A RuntimeError is thrown for any other value.

## Examples

```python
import torch
import torch_npu

# Query the current mode.
print(torch_npu.npu.get_task_queue_enable())  # 1 (default)

# Switch to Level 2 optimization.
torch_npu.npu.set_task_queue_enable(2)
print(torch_npu.npu.get_task_queue_enable())  # 2

# Disable TaskQueue optimization.
torch_npu.npu.set_task_queue_enable(0)
print(torch_npu.npu.get_task_queue_enable())  # 0
```

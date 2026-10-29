# (beta) torch_npu.npu.get_task_queue_enable

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

This API queries the currently effective TaskQueue optimization level and returns the mode currently in effect for `torch_npu.npu.set_task_queue_enable` or the `TASK_QUEUE_ENABLE` environment variable.

`ASCEND_LAUNCH_BLOCKING=1` is the synchronous execution mode used for fault locating and is mutually exclusive with the TaskQueue asynchronous pipeline mechanism. When this mode is enabled, this API always returns 0 regardless of the TaskQueue mode setting.

## Prototype

```python
torch_npu.npu.get_task_queue_enable()
```

## Return Values

**int**: Currently effective TaskQueue optimization level. The values are as follows:

- 0: TaskQueue optimization is disabled.
- 1: Level 1 optimization.
- 2: Level 2 optimization.

## Examples

```python
import torch
import torch_npu

# Query the current mode (defaults to 1 when not set).
print(torch_npu.npu.get_task_queue_enable())  # 1

# Query the mode after dynamic switching.
torch_npu.npu.set_task_queue_enable(2)
print(torch_npu.npu.get_task_queue_enable())  # 2
```

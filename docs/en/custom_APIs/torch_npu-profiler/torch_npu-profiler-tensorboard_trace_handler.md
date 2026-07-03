# torch_npu.profiler.tensorboard_trace_handler

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Provides a callback function executed after collection ends, used to control the export of profile data.

## Prototype

```python
torch_npu.profiler.tensorboard_trace_handler(dir_name=None, worker_name=None, analyse_flag=True, async_mode=False)
```

## Parameters

- **`dir_name`** (`str`): Optional. Storage path of the collected profile data. The path can contain only letters, digits, underscores, and hyphens. Symbolic links are not supported. If this parameter is omitted and the `tensorboard_trace_handler` function is configured, profile data is saved to the current directory by default. If `on_trace_ready=torch_npu.profiler.tensorboard_trace_handler` is omitted from the code, raw profile data is saved, which requires [offline parsing](./torch_npu-profiler-profiler-analyse.md).

    This function takes precedence over the `ASCEND_WORK_PATH` environment variable.

- **`worker_name`** (`str`): Optional. Identifies the unique worker thread, and the default value is `{hostname}_{pid}`. The path can contain only letters, digits, underscores, and hyphens. Symbolic links are not supported.

- **`analyse_flag`** (`bool`): Optional. Specifies whether to automatically parse profile data. Valid values are:

    - `True`: enables automatic parsing.
    - `False`: disables automatic parsing. The collected profile data can be parsed using [offline parsing](./torch_npu-profiler-profiler-analyse.md).

    The default value is `True`.

- **`async_mode`** (`bool`): Optional. Specifies whether to enable asynchronous parsing, which allows the parsing process to run without blocking the main AI task workflow. Valid values are:

    - `True`: enables asynchronous parsing.
    - `False`: disables asynchronous parsing, which enables synchronous parsing.

    The default value is `False`.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

...

with torch_npu.profiler.profile(
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # Training function
                train_one_step()  # Training function
                prof.step()
```

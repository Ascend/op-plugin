# torch_npu.profiler.ExportType

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Sets the file format of exported profile data result files, List type. It serves as the `export_type` parameter of the `_ExperimentalConfig` class.

## Prototype

```python
torch_npu.profiler.ExportType
```

## Parameters

- **`torch_npu.profiler.ExportType.Text`**: Optional. Parses data into `.json` and `.csv` timeline and summary files, and a `.db` file (`ascend_pytorch_profiler_{Rank_ID}.db` or `analysis.db`) that aggregates all profile data.
- **`torch_npu.profiler.ExportType.Db`**: Optional. Parses data only into a `.db` format file (`ascend_pytorch_profiler_{Rank_ID}.db` or `analysis.db`) that aggregates all profile data for visualization using the MindStudio Insight tool. This value is supported only for exports through the `on_trace_ready` interface or offline parsing exports. You must install a matching Toolkit package that supports exporting the `.db` format.

If an invalid value is provided or if this parameter is omitted, the default value `torch_npu.profiler.ExportType.Text` is used.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       export_type=[
              torch_npu.profiler.ExportType.Text
              ],
       )
with torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # Training function
                train_one_step() # Training function
                prof.step()
```

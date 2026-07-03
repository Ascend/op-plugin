# torch_npu.profiler.profiler.analyse

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Parses the profile data collected by Ascend PyTorch Profiler offline.

## Prototype

```python
torch_npu.profiler.profiler.analyse(profiler_path="", max_process_number=max_process_number, export_type=export_type)
```

## Parameters

- **`profiler_path`** (`str`): Required. PyTorch profile data path. The path can contain only letters, digits, and underscores. Symbolic links are not supported. The specified directory must store the PyTorch profile data directory `{worker_name}_{timestamp}_ascend_pt`.

- **`max_process_number`** (`int`): Optional. Maximum process number for offline parsing. The value range is `[1, number of CPU cores]`. The default value is half of the number of CPU cores. If the provided value exceeds the number of CPU cores in the environment, the number of CPU cores is automatically used. If an invalid value is provided, the default value is used.

- **`export_type`** (`list`): Optional. File format of the exported profile data result files. Valid values are:

  - `text`: parses data into `.json` and `.csv` format timeline and summary files, and a `.db` format file (`ascend_pytorch.db` or `analysis.db`) that aggregates all profile data.
  - `db`: parses data only into a `.db` format file (`ascend_pytorch.db` or `analysis.db`) that aggregates all profile data for visualization using the MindStudio Insight tool. This value is supported only for exports through the `on_trace_ready` interface or [offline parsing](./torch_npu-profiler-profiler-analyse.md) exports. You must install a matching Toolkit package that supports exporting the `.db` format.

  If an invalid value is provided or if this parameter is omitted, the `export_type` field in `profiler_info.json` is read to determine the export format.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

Create a custom `_{file_name}_.py` file and edit the following code:

```python
from torch_npu.profiler.profiler import analyse

if __name__ == "__main__":
    analyse(profiler_path="./result_data", max_process_number=max_process_number)
```

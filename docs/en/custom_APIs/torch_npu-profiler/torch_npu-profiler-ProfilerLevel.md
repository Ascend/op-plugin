# torch_npu.profiler.ProfilerLevel

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Defines the profile data collection level, Enum type. It serves as the `profiler_level` parameter of the `_ExperimentalConfig` class.

## Prototype

```python
torch_npu.profiler.ProfilerLevel
```

## Parameters

- **`torch_npu.profiler.ProfilerLevel.Level_none`**: Optional. Disables collection of all level-controlled data, which disables `profiler_level`.

- **`torch_npu.profiler.ProfilerLevel.Level0`**: Optional. Collects upper-layer application data, lower-layer NPU data, or information about operators executed on the NPU. If this parameter is provided, the tool collects only partial data and omits some operator information. For details, see <a href="https://gitcode.com/Ascend/msprof/blob/26.2.0/docs/en/user_guide/profile_data_file_references.md#op_summary-operator-details">op_summary (Operator Details)</a> in the *msProf* documentation.
- **`torch_npu.profiler.ProfilerLevel.Level1`**: Optional. Compared with `torch_npu.profiler.ProfilerLevel.Level0`, this level additionally collects CANN layer AscendCL data, AI Core performance metrics executed on the NPU, and HCCL `communication.json` and `communication_matrix.json` files. This level also enables `aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization`.
- **torch_npu.profiler.ProfilerLevel.Level2**: Optional. Compared with `torch_npu.profiler.ProfilerLevel.Level1`, this level additionally collects CANN layer runtime data and AI CPU data (`data_preprocess.csv` file).

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       profiler_level=torch_npu.profiler.ProfilerLevel.Level0
       )
with torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # Training function
                train_one_step() # Training function
                prof.step()
```

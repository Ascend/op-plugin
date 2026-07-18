# torch_npu.profiler._ExperimentalConfig

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Configures the extended profile data collection parameters. This API is used to construct the `experimental_config` parameter of `torch_npu.profiler.profile`.

## Prototype

```python
torch_npu.profiler._ExperimentalConfig(export_type=[torch_npu.profiler.ExportType.Text], profiler_level=torch_npu.profiler.ProfilerLevel.Level0, mstx=False, mstx_domain_include=[], mstx_domain_exclude=[], aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone, l2_cache=False, op_attr=False, data_simplification=True, record_op_args=False, gc_detect_threshold=None, host_sys=[], sys_io=False, sys_interconnection=False)
```

## Parameters

- **`export_type`** (`list`): Optional. File format of the exported profile data result files. For details about the values and meanings, see [torch_npu.profiler.ExportType](torch_npu-profiler-ExportType.md).

- **`profiler_level`** (`Enum`): Optional. Profile data collection level. For details about the values and meanings, see [torch_npu.profiler.ProfilerLevel](torch_npu-profiler-ProfilerLevel.md).

- **`mstx`** or **`msprof_tx`** (`bool`): Optional. Specifies whether to enable custom instrumentation. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `False`.

    The legacy parameter name `msprof_tx` is renamed to `mstx`, which remains compatible in new versions.

- **`mstx_domain_include`** (`list`): Optional. Outputs required domain data. When calling the `torch_npu.npu.mstx` series instrumentation APIs to add instrumentation points using the default domain or specified domains, you can choose to output only the domain data configured in this parameter.

    The domain names can be those provided when you call the [torch_npu.npu.mstx](../torch_npu-npu/torch_npu-npu-mstx.md) series APIs or the default domain (`'default'`). The domain names must be provided using the list type.

    This parameter is mutually exclusive with `mstx_domain_exclude`. If both are configured, only `mstx_domain_include` takes effect.

    `mstx` must be set to `True`.

- **`mstx_domain_exclude`** (`list`): Optional. Filters unrequired domain data. When calling the `torch_npu.npu.mstx` series instrumentation APIs to add instrumentation points using the default domain or a specified domain, you can choose not to output the domain data configured in this parameter.

    The domain name can be the domains provided when calling the `torch_npu.npu.mstx` series APIs or the default domain (`'default'`). The domain names must be provided using the list type.

    This parameter is mutually exclusive with `mstx_domain_include`. If both are configured, only `mstx_domain_include` takes effect.

    `mstx` must be set to `True`.

- **`aic_metrics`** (`Enum`): Optional. AI Core performance metric collection items. The collected result data is displayed in Kernel View. For details about the values and meanings, see [torch_npu.profiler.AiCMetrics](torch_npu-profiler-AiCMetrics.md).

- **`l2_cache`** (`bool`): Optional. Specifies whether to enable L2 cache data collection. This item generates the `l2_cache.csv` file in `ASCEND_PROFILER_OUTPUT`. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `False`.

- **`op_attr`** (`bool`): Optional. Specifies whether to enable operator attribute information collection. Currently, only `aclnn` operators are supported. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `False`.

     This parameter does not take effect when `torch_npu.profiler.ProfilerLevel.Level_none` is enabled.

- **`data_simplification`** (`bool`): Optional. Specifies whether to delete redundant data after exporting profile data. If enabled, only `profiler_*.json` files, the `ASCEND_PROFILER_OUTPUT` directory, raw profile data under the `PROF_XXX` directory, the `FRAMEWORK` directory, and the `logs` directory are retained to save storage space. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `True`.

- **`record_op_args`** (`bool`): Optional. Specifies whether to enable the operator information statistics recording function. If enabled, collected operator information files are output to the `{worker_name}_{timestamp}_ascend_pt_op_args` directory. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `False`.

- **`gc_detect_threshold`** (`float`): Optional. GC detection threshold. The value must be greater than or equal to 0, in milliseconds (ms). When a numeric value is specified, GC detection is enabled and only GC events whose duration exceeds the threshold are collected. Configuring this parameter to `0` collects all GC events. This can cause excessive collection data volume, use with caution. The recommended value is `1` ms.

    The default value is `None`, which disables the GC detection function.

    - GC is used by the Python process to reclaim the memory of destroyed objects.
    - The GC layer generates records in `trace_view.json` or generates the `GC_RECORD` table in `ascend_pytorch_profiler_{Rank_ID}.db`.

- **`host_sys`** (`list`): Optional. Switch for host-side system data collection. If omitted, host-side system data collection is disabled. Valid values are:

    - **`torch_npu.profiler.HostSystem.CPU`**: process-level CPU utilization
    - **`torch_npu.profiler.HostSystem.MEM`**: process-level memory usage
    - **`torch_npu.profiler.HostSystem.DISK`**: process-level drive I/O usage
    - **`torch_npu.profiler.HostSystem.NETWORK`**: system-level network I/O usage
    - **`torch_npu.profiler.HostSystem.OSRT`**: process-level syscall and pthreadcall

- **`sys_io`** (`bool`): Optional. Switch for NIC, RoCE, or MAC data collection. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `False`.

- **`sys_interconnection`** (`bool`): Optional. Switch for collective communication bandwidth data (HCCS), PCIe data, or inter-chip transmission bandwidth information collection. Valid values are:

    - `True`: enabled
    - `False`: disabled

    The default value is `False`.

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
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

with torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config=experimental_config) as prof:
        for step in range(steps): # Training function
                train_one_step() # Training function
                prof.step()
```

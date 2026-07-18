# torch.npu.set_stream_limit

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

- Sets the device resource limits for a specified stream.
- This API must be called after the `torch.npu.set_device_limit` API is called and before operator execution. If the device resource limits are set multiple times for the same stream, the last setting takes effect.
- After this API is set, the stream can be passed across threads for use.

## Prototype

```python
torch.npu.set_stream_limit(stream, cube_num=-1, vector_num=-1) -> None
```

## Parameters

- **`stream`** (`torch_npu.npu.Stream`): Required. Stream for resource limit control.
- **`cube_num`** (`int`): Optional. Number of Cube cores. The default value is `-1` (no core splitting limit is configured).
- **`vector_num`** (`int`): Optional. Number of Vector cores. The default value is `-1` (no core splitting limit is configured).

## Return Values

`None`

No value is returned.

## Constraints

- This API supports core control only for operators developed using Ascend C. Core control is currently not supported for non-Ascend C operators. Using this API in micro-batch multi-stream parallel scenarios may cause deadlocks or other adverse effects, and its use is not recommended. For non-Ascend C operators, you can refer to [CANN Ascend C Operator Development](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html) to add operator implementations before using this API to implement operator core control.
- This API is primarily intended for micro-batch multi-stream parallelism. When it is used together with operators that do not support core control, the effectiveness of multi-stream parallelism may be affected.
- This API must be used together with `torch_npu.npu.config.allow_internal_format = False` (proprietary formats are not allowed).
- This API does not support concurrent multi-threaded configuration of core limits on the same stream. Therefore, the effective core limits during operator execution cannot be guaranteed.

## Example

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.set_stream_limit(torch.npu.Stream(), 13, 23)
 ```

## Effective Core Control

1. The Ascend PyTorch Profiler is used to collect performance data. This includes information about PyTorch layer operators, CANN layer operators, underlying NPU operators, and operator memory footprints.
   > **Note**: Ascend PyTorch Profiler is a performance analysis tool developed by CANN for the PyTorch framework. You can add an Ascend PyTorch Profiler interface (`torch_npu.profiler.profile` is recommended) to your PyTorch script to collect specified metrics. During model execution, performance data is collected synchronously. For details about its usage and the resulting files, see section "Ascend PyTorch Profiler" in the [CANN Performance Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0001.html) documentation.

     ```python
     >>> import torch
     >>> import torch_npu
     >>> stream1 = torch.npu.current_stream()
     >>> stream2 = torch.npu.Stream()
     >>> x1 = torch.randn(1024, 1960).npu()
     >>> experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
     >>> with torch_npu.profiler.profile(
     >>>    with_stack=False,                # Specifies whether to collect the operator function call stack. Default value: False.
     >>>    record_shapes=False,             # Specifies whether to collect operator input shapes and input types. Default value: False.
     >>>    profile_memory=False,            # Specifies whether to collect memory-related data. Default value: False.
     >>>    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1),           # warmup defaults to 0; required in early torch_npu versions.
     >>>    experimental_config=experimental_config,                                   
     >>>    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir") # Exports TensorBoard visualization data.
     >>>    ) as prof:
     >>>    torch.npu.set_stream_limit(stream1, 12, 22)
     >>>    torch.npu.set_stream_limit(stream2, 13, 23)
     >>>    output = torch_npu.npu_swiglu(x1, dim=-1)
     >>>    with torch.npu.stream(stream2):
     >>>       output = torch_npu.npu_swiglu(x1, dim=-1)
     ```

   When the following message is displayed, data collection is successful. The message "Start parsing profiling data" indicates the output path of the profiling results.

     ```shell
     2025-07-01 08:50:41 [INFO] [367681] profiler.py: Start parsing profiling data: /home/prof/${hostname}_${pid}_${timestamp}_ascend_pt
     2025-07-01 08:50:44 [INFO] [367725] profiler.py: CANN profiling data parsed in a total time of 0:00:03.169691
     2025-07-01 08:50:45 [INFO] [367681] profiler.py: All profiling data parsed in a total time of 0:00:04.654659
     ......
     ```

   The key outputs are as follows:

     ```shell
     |-- /home/prof/${hostname}_${pid}_${timestamp}_ascend_pt   
       |-- ASCEND_PROFILER_OUTPUT           // Directory for collected and parsed performance data
         |-- api_statistic.csv             // Records CANN layer API execution duration statistics; generated when profiler_level is set to Level 1 or Level 2
         |-- kernel_details.csv            // Generated when activities is set to NPU type
         |-- op_statistic.csv              // Call count and execution duration for AI Core and AI CPU operators
         |-- operator_details.csv          // Generated when activities is set to the CPU type and record_shapes is set to True
         |-- step_trace_time.csv           // Computation and communication duration statistics per iteration
         |-- trace_view.json               // Timeline information for the entire AI task
         |-- ......
       |-- FRAMEWORK                        // Raw performance data on the framework side (can be ignored)
       |-- logs                             // Logs generated during parsing
       ......
   ```

2. View the collected result files (in formats such as JSON and CSV).

   For single-operator core control, you can check the `kernel_details.csv` file and check the **BlockDim** column (the number of cores used for operator computation). If the `BlockDim` value is less than or equal to the core limit configured by the user, the core control is successful.
   In this example, the `BlockDim` values of the SwiGlu operator are `22` and `23`, which comply with the core control logic.

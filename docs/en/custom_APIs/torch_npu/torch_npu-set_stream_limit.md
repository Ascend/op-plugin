# torch.npu.set_stream_limit

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

## Function

- Sets the device resource limit for a specified stream.
- Call this API after `torch.npu.set_device` and before operator execution. If the limit is set multiple times for the same stream, the last setting takes effect.
- After this API is called, the stream can be passed across threads.

## Prototype

```python
torch.npu.set_stream_limit(stream, cube_num=-1, vector_num=-1) -> None
```

## Parameters

- **`stream`** (`torch_npu.npu.Stream`): Required. Stream for resource limit control.
- **`cube_num`** (`int`): Optional. Number of Cube cores. The default value is `-1` (no core control limit is configured).
- **`vector_num`** (`int`): Optional. Number of Vector cores. The default value is `-1` (no core control limit is configured).

## Return Values

`None`

Represents no return value.

## Constraints

- This API supports core control only for operators developed for Ascend C. Core control is not supported for operators not developed for Ascend C, and using this API in micro-batch multi-stream parallel scenarios may cause a deadlock or other impacts, so it is not recommended. For non-Ascend C operators, you can refer to *[Ascend C Operator Development](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/en/guide/index.md)* to add an operator implementation, and then use this API to control the cores of the operator.
- This API is mainly used in micro-batch multi-stream parallelism. If an operator does not support core control, the multi-stream parallel effect may be affected.
- This API must be used together with torch_npu.npu.config.allow_internal_format = False (private formats not allowed).
- This API does not support setting the number of controlled cores on the same stream concurrently in multiple threads, and the effective core control value during operator execution cannot be guaranteed.

## Example

 ```python
>>> import torch
>>> import torch_npu
>>> torch.npu.set_device(0)
>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.set_stream_limit(torch.npu.Stream(), 13, 23)
 ```

**Core control example**

1. Use the Ascend PyTorch Profiler API to collect performance data, including PyTorch layer operator information, CANN layer operator information, underlying NPU operator information, and operator memory usage information.
   > **Note**: Ascend PyTorch Profiler is a performance analysis tool developed by CANN for the PyTorch framework. By adding the Ascend PyTorch Profiler API (the torch_npu.profiler.profile API is recommended) to a PyTorch script, the specified metric data is collected, and performance data is collected synchronously during model execution. For details about the usage and the result files, see <a href="https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/developer_notes/ascend_pytorch_profiler_user_guide.md">Ascend PyTorch Tuning Tool</a>.

     ```python
     >>> import torch
     >>> import torch_npu
     >>> torch.npu.set_device(0)
     >>> stream1 = torch.npu.current_stream()
     >>> stream2 = torch.npu.Stream()
     >>> x1 = torch.randn(1024, 1960).npu()
     >>> experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
     >>> with torch_npu.profiler.profile(
     >>>    with_stack=False,                # Switch for collecting the function call stack of operators. Disabled by default.
     >>>    record_shapes=False,             # Switch for collecting the input shape and input type of operators. Disabled by default.
     >>>    profile_memory=False,            # Switch for collecting memory-related data. Disabled by default.
     >>>    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1),           # The default value of warmup is 0. In earlier TorchNPU packages, this parameter is mandatory.
     >>>    experimental_config=experimental_config,                                   
     >>>    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir") # Export TensorBoard visualization data
     >>>    ) as prof:
     >>>    torch.npu.set_stream_limit(stream1, 12, 22)
     >>>    torch.npu.set_stream_limit(stream2, 13, 23)
     >>>    output = torch_npu.npu_swiglu(x1, dim=-1)
     >>>    with torch.npu.stream(stream2):
     >>>       output = torch_npu.npu_swiglu(x1, dim=-1)
     ```

   When the following print information is displayed, the collection is normal. The "Start parsing profiling data" information indicates the path of the collection result.

     ```shell
     2025-07-01 08:50:41 [INFO] [367681] profiler.py: Start parsing profiling data: /home/prof/${hostname}_${pid}_${timestamp}_ascend_pt
     2025-07-01 08:50:44 [INFO] [367725] profiler.py: CANN profiling data parsed in a total time of 0:00:03.169691
     2025-07-01 08:50:45 [INFO] [367681] profiler.py: All profiling data parsed in a total time of 0:00:04.654659
     ......
     ```

   The key artifacts are as follows:

     ```shell
     |-- /home/prof/${hostname}_${pid}_${timestamp}_ascend_pt   
       |-- ASCEND_PROFILER_OUTPUT           // Directory of the collected and parsed performance data
         |-- api_statistic.csv             // Generated when profiler_level is set to Level1 or Level2. Statistics about the CANN layer API execution duration.
         |-- kernel_details.csv            // Generated when activities is set to the NPU type
         |-- op_statistic.csv              // Statistics about the number of AI Core and AICPU operator calls and the duration
         |-- operator_details.csv          // Generated when activities is set to the CPU type and record_shapes is set to True
         |-- step_trace_time.csv           // Statistics about the computation and communication time in an iteration
         |-- trace_view.json               // Records the time information of the entire AI task
         |-- ......
       |-- FRAMEWORK                        // Raw performance data on the framework side. This directory can be ignored.
       |-- logs                             // Parsing process logs
       ......
   ```

2. View the collected result files (in json, csv, and other formats).

   For single-operator core control, view the **kernel_details.csv** file and check the **BlockDim** column (the number of cores used by the operator for computation). If BlockDim is less than or equal to the number of controlled cores set by the user, core control is considered successful.
   In the example, the BlockDim of the SwiGlu operator is 22 and 23, which complies with the core control logic.

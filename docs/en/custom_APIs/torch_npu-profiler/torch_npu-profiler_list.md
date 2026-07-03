# torch_npu.profiler APIs

This section describes the custom APIs related to profiling, which provide data required for performance optimization.

| API                                                     | Description                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [torch_npu.profiler.profile](./torch_npu-profiler-profile.md) | Collects profile data during PyTorch training.                   |
| [torch_npu.profiler._KinetoProfile](./torch_npu-profiler-_KinetoProfile.md) | Collects profile data during PyTorch training.                   |
| [torch_npu.profiler.ProfilerActivity](./torch_npu-profiler-ProfilerActivity.md) | Defines the event collection list, Enum type. It is used to assign a value to the `activities` parameter of `torch_npu.profiler.profile`.|
| [torch_npu.profiler.tensorboard_trace_handler](./torch_npu-profiler-tensorboard_trace_handler.md) | Exports the collected profile data to a format supported by the TensorBoard tool. This function acts as the execution operation for the `on_trace_ready` parameter of `torch_npu.profiler.profile`.|
| [torch_npu.profiler.schedule](./torch_npu-profiler-schedule.md) | Sets the action for each step. It constructs the `schedule` parameter of `torch_npu.profiler.profile`.|
| [torch_npu.profiler.ProfilerAction](./torch_npu-profiler-ProfilerAction.md) | Controls the profiler action state, Enum type.                                    |
| [torch_npu.profiler._ExperimentalConfig](./torch_npu-profiler-_ExperimentalConfig.md) | Configures the extended profile data collection parameters. This API is used to construct the `experimental_config` parameter of `torch_npu.profiler.profile`.|
| [torch_npu.profiler.ExportType](./torch_npu-profiler-ExportType.md) | Sets the file format of exported profile data result files, List type. It serves as the `export_type` parameter of the `_ExperimentalConfig` class.|
| [torch_npu.profiler.ProfilerLevel](./torch_npu-profiler-ProfilerLevel.md) | Defines the profile data collection level. It serves as the `profiler_level` parameter of the `_ExperimentalConfig` class.  |
| [torch_npu.profiler.AiCMetrics](./torch_npu-profiler-AiCMetrics.md) | Defines AI Core performance metric collection items, used as the `aic_metrics` parameter of the `_ExperimentalConfig` class.|
| [torch_npu.profiler.supported_activities](./torch_npu-profiler-supported_activities.md) | Queries the CPU or NPU events of the `activities` parameter supported for collection.            |
| [torch_npu.profiler.supported_profiler_level](./torch_npu-profiler-supported_profiler_level.md) | Queries the supported levels of `torch_npu.profiler.ProfilerLevel`.        |
| [torch_npu.profiler.supported_ai_core_metrics](./torch_npu-profiler-supported_ai_core_metrics.md) | Queries the supported AI Core performance metric collection items of `torch_npu.profiler.AiCMetrics`.|
| [torch_npu.profiler.supported_export_type](./torch_npu-profiler-supported_export_type.md) | Queries the supported profile data result file formats of `torch_npu.profiler.ExportType`.|
| [torch_npu.profiler.dynamic_profile.init](./torch_npu-profiler-dynamic_profile-init.md) | Initializes `dynamic_profile`.                             |
| [torch_npu.profiler.dynamic_profile.step](./torch_npu-profiler-dynamic_profile-step.md) | Divides steps during `dynamic_profile` data collection.                           |
| [torch_npu.profiler.dynamic_profile.start](./torch_npu-profiler-dynamic_profile-start.md) | Triggers a `dynamic_profile` data collection.                           |
| [torch_npu.profiler.profiler.analyse](./torch_npu-profiler-profiler-analyse.md) | Parses the profile data collected by Ascend PyTorch Profiler offline.                   |
| [torch_npu.profiler.profile.enable_profiler_in_child_thread](./torch_npu-profiler-profile-enable_profiler_in_child_thread.md) | Registers the profiler collection callback function.                                  |
| [torch_npu.profiler.profile.disable_profiler_in_child_thread](./torch_npu-profiler-profile-disable_profiler_in_child_thread.md) | Deregisters the profiler collection callback function.                                  |

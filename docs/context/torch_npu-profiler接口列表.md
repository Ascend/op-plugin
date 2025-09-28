# torch_npu.profiler接口列表

本章节包含采集profiling相关的自定义接口，提供性能优化所需要的数据。

| API名称                                                      | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [torch_npu.profiler.profile](./torch_npu.profiler.profile.md) | 提供PyTorch训练过程中的性能数据采集功能。                    |
| [torch_npu.profiler._KinetoProfile](./torch_npu.profiler._KinetoProfile.md) | 提供PyTorch训练过程中的性能数据采集功能。                    |
| [torch_npu.profiler.ProfilerActivity](./torch_npu.profiler.ProfilerActivity.md) | 事件采集列表，枚举类。用于赋值给torch_npu.profiler.profile的activities参数。 |
| [torch_npu.profiler.tensorboard_trace_handler](./torch_npu.profiler.tensorboard_trace_handler.md) | 将采集到的性能数据导出为TensorBoard工具支持的格式。作为torch_npu.profiler.profile on_trace_ready参数的执行操作。 |
| [torch_npu.profiler.schedule](./torch_npu.profiler.schedule.md) | 设置不同step的行为。用于构造torch_npu.profiler.profile的schedule参数。 |
| [torch_npu.profiler.ProfilerAction](./torch_npu.profiler.ProfilerAction.md) | Profiler状态，Enum类型。                                     |
| [torch_npu.profiler._ExperimentalConfig](./torch_npu.profiler._ExperimentalConfig.md) | 性能数据采集扩展参数。用于构造torch_npu.profiler.profile的experimental_config参数。 |
| [torch_npu.profiler.ExportType](./torch_npu.profiler.ExportType.md) | 设置导出的性能数据结果文件格式，作为 _ExperimentalConfig类的export_type参数。 |
| [torch_npu.profiler.ProfilerLevel](./torch_npu.profiler.ProfilerLevel.md) | 采集等级，作为 _ExperimentalConfig类的profiler_level参数。   |
| [torch_npu.profiler.AiCMetrics](./torch_npu.profiler.AiCMetrics.md) | AI Core的性能指标采集项，作为 _ExperimentalConfig类的aic_metrics参数。 |
| [torch_npu.profiler.supported_activities](./torch_npu.profiler.supported_activities.md) | 查询当前支持采集的activities参数的CPU、NPU事件。             |
| [torch_npu.profiler.supported_profiler_level](./torch_npu.profiler.supported_profiler_level.md) | 查询当前支持的torch_npu.profiler.ProfilerLevel级别。         |
| [torch_npu.profiler.supported_ai_core_metrics](./torch_npu.profiler.supported_ai_core_metrics.md) | 查询当前支持的torch_npu.profiler. AiCMetrics的AI Core性能指标采集项。 |
| [torch_npu.profiler.supported_export_type](./torch_npu.profiler.supported_export_type.md) | 查询当前支持的torch_npu.profiler.ExportType的性能数据结果文件类型。 |
| [torch_npu.profiler.dynamic_profile.init](./torch_npu.profiler.dynamic_profile.init.md) | 初始化dynamic_profile动态采集。                              |
| [torch_npu.profiler.dynamic_profile.step](./torch_npu.profiler.dynamic_profile.step.md) | dynamic_profile动态采集划分step。                            |
| [torch_npu.profiler.dynamic_profile.start](./torch_npu.profiler.dynamic_profile.start.md) | 触发一次dynamic_profile动态采集。                            |
| [torch_npu.profiler.profiler.analyse](./torch_npu.profiler.profiler.analyse.md) | Ascend PyTorch Profiler性能数据离线解析。                    |
| [torch_npu.profiler.profile.enable_profiler_in_child_thread](./torch_npu.profiler.profile.enable_profiler_in_child_thread.md) | 注册Profiler采集回调函数。                                   |
| [torch_npu.profiler.profile.disable_profiler_in_child_thread](./torch_npu.profiler.profile.disable_profiler_in_child_thread.md) | 注销Profiler采集回调函数。                                   |
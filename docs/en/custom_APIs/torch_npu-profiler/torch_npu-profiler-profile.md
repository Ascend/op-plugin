# torch_npu.profiler.profile

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Collects profile data during PyTorch training.

## Prototype

```python
torch_npu.profiler.profile(activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_modules=False, with_flops=False, experimental_config=None, custom_trace_id_callback=None)
```

## Parameters

- **`activities`** (`enum`): Optional. CPU or NPU event collection list. For details about the values and meanings, see [torch_npu.profiler.ProfilerActivity](./torch_npu-profiler-ProfilerActivity.md).

- **`schedule`** (`callable`): Optional. Sets the action for each step. It is controlled by the [schedule class](./torch_npu-profiler-schedule.md). If omitted, no operation is performed.

- **`on_trace_ready`** (`callable`): Optional. Automatically executes an operation when collection ends. This parameter supports only operations of the [tensorboard_trace_handler](./torch_npu-profiler-tensorboard_trace_handler.md) function. If omitted, no operation is performed.

- **`record_shapes`** (`bool`): Optional. Specifies whether to record input shapes and input types of operators. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`profile_memory`** (`bool`): Optional. Specifies whether to record memory allocation status of operators. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  > [!NOTE]
  > Collecting profile data in environments where glibc is earlier than 2.34 can trigger a known glibc bug ([ID: 19329](https://sourceware.org)). Upgrading the environment glibc version resolves this issue.

- **`with_stack`** (`bool`): Optional. Specifies whether to record call stack information. This includes call information of the framework layer or CPU operator layer. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`with_modules`** (`bool`): Optional. Specifies whether to record the module-level Python call stack, which is call information at the framework layer. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`with_flops`** (`bool`): Optional. Specifies whether to record floating-point operations of operators. Profile data analysis for this parameter is not supported. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`experimental_config`**: Optional. Configures common collection items of the performance profiling tool through extended configurations. For details about the supported collection items, see [torch_npu.profiler._ExperimentalConfig](./torch_npu-profiler-_ExperimentalConfig.md).

- **`custom_trace_id_callback`** (`Callable`): Optional. Generates a trace_id to identify each piece of profile data. The trace_id is output to the `profiler_metadata.json` file.

## Return Values

None

## Examples

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

- Complete parameter example for profile data collection

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
      activities=[
          torch_npu.profiler.ProfilerActivity.CPU,
          torch_npu.profiler.ProfilerActivity.NPU
          ],
      schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1, skip_first_wait=1),
      on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
      record_shapes=False,
      profile_memory=False,
      with_stack=False,
      with_modules=False,
      with_flops=False,
      experimental_config=experimental_config) as prof:
      for step in range(steps): # Training function
          train_one_step() # Training function
          prof.step()
  ```

- Generate a trace ID to identify profiler data

  ```python
  import torch
  import torch_npu
  ...
  
  # Define a trace ID generator
  class RepeatTraceIdGenerator:
      def __init__(self):
          self.repeat_count = 0    # Count from 0
  
      def __call__(self) -> str:
          # The count increments by 1 each time profile data collection is started.
          current_id = self.repeat_count
          self.repeat_count += 1
          return str(current_id)
  
  # Create a trace ID generator
  trace_id_gen = RepeatTraceIdGenerator()
  
  if __name__ == "__main__":
      device = torch.device('npu:1')
      torch.npu.set_device(device)
      x0 =torch.rand(3, 4).npu()
      x1 =torch.rand(3, 4).npu()
  
  
      stream = torch.npu.current_stream()
      stream.synchronize()
  
  # Add basic profiling configuration parameters
  with torch_npu.profiler.profile(
      activities=[
          torch_npu.profiler.ProfilerActivity.CPU,
          torch_npu.profiler.ProfilerActivity.NPU
      ],
      schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=1),
      on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
      custom_trace_id_callback=trace_id_gen
  ) as prof:
      for i in range(12):
          add(x0, x1)  # Training function
          prof.step()
  ```

# torch_npu.npu.mstx.range_push

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>|    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

Marks the beginning of instrumentation.

Use it together with [torch_npu.npu.mstx.range_pop](./torch_npu-npu-mstx-range_pop.md).

Use it only within a single thread. Multiple nested calls are supported. `torch_npu.npu.mstx.range_pop` automatically matches the most recent `torch_npu.npu.mstx.range_push`.

## Prototype

```python
torch_npu.npu.mstx.range_push(message: str, stream=None, domain: str='default') -> int
```

## Parameters

- **`message`** (`str`): Required. String carrying information for the instrumentation point.
  
  Length of the `message` string must not exceed 255 bytes in msPTI scenarios.

- **`stream`** (`torch_npu.npu.Stream`): Optional. Stream used to execute the instrumentation task (default value: `None`).

  - If set to `None`, only instantaneous events on the host are marked.
  - If set to a valid stream, the instantaneous events on the host and device are marked.
  
- **`domain`** (`str`): Optional. Name of the domain where instantaneous events are marked. The default value is 'default', indicating the default domain. If omitted, the default domain is used.

## Return Values

Nesting level of the range instrumentation recorded by this API within the current thread, starting from `0`. Returns `-1` if the API call fails.

## Examples

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

- Non-nested call

  ```python
  torch_npu.npu.mstx.range_push("dataloader")
  dataloader()
  torch_npu.npu.mstx.range_pop()
  ```

- Nested call

  ```python
  torch_npu.npu.mstx.range_push("dataloader1", cur_stream)
  dataloader()    # Event 1
  torch_npu.npu.mstx.range_push("dataloader2", cur_stream)
  dataloader()    # Event 2
  torch_npu.npu.mstx.range_pop()
  dataloader()    # Event 3
  torch_npu.npu.mstx.range_pop()
  ```

  In the preceding example, the duration of Event 2 is collected by the innermost pair of `push` and `pop` calls, while Events 1, 2, and 3 are enclosed by the outermost pair of `push` and `pop` calls.

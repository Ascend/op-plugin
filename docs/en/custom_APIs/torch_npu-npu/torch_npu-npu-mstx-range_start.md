# torch_npu.npu.mstx.range_start

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

Use it together with [torch_npu.npu.mstx.range_end](./torch_npu-npu-mstx-range_end.md).

It supports cross-thread usage and multiple nested calls. `torch_npu.npu.mstx.range_end` automatically matches the most recent `torch_npu.npu.mstx.range_start`.

## Prototype

```python
torch_npu.npu.mstx.range_start(message: str='None', stream=None, domain: str='default') -> int
```

## Parameters

- **`message`** (`str`): Optional. String carrying information for the instrumentation point (default value: `'None'`).

  Length of the `message` string must not exceed 255 bytes in msPTI scenarios.

- **`stream`** (`torch_npu.npu.Stream`): Optional. Stream used to execute the instrumentation task (default value: `None`).

  - If set to `None` or not specified, only instantaneous events on the host are marked.
  - If set to a valid stream, the instantaneous events on the host and device are marked.

- **`domain`** (`str`): Optional. Name of the domain where instantaneous events are marked. The default value is 'default', indicating the default domain. If omitted, the default domain is used.

## Return Values

**`range_id`**: ID of the range. If API execution fails, `0` is returned.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
id = torch_npu.npu.mstx.range_start("dataloader", None)
dataloader()
torch_npu.npu.mstx.range_end(id)
```

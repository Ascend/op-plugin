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

## Prototype

```python
torch_npu.npu.mstx.range_start(message: str='None', stream=None, domain: str='default') -> int:
```

## Parameters

- **`message`** (`str`): Optional. String pointer carrying instrumentation message data. The default value is `None`. The length of the passed message string must satisfy the following constraints:
  - MSPTI scenarios: No more than 255 bytes.
  - Non-MSPTI scenarios: No more than 156 bytes.
- **`stream`** (`int`): Optional. Stream used to execute the instrumentation task. The default value is `None`.
  - If set to `None`, only instantaneous events on the host are marked.
  - If set to a valid stream, the instantaneous events on the host and device are marked.
- **`domain`** (`str`): Optional. Name of the domain where instantaneous events are marked. The default value is `default`, indicating the default domain. If omitted, the default domain is used.

## Return Values

**`range_id`**: ID of the range. If API execution fails, `0` is returned.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
id = torch_npu.npu.mstx.range_start("dataloader", None)    # If the second input parameter is set to None or not set, only the range duration on the host is recorded.
dataloader()
torch_npu.npu.mstx.range_end(id)
```

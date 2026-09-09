# torch_npu.npu.mstx.range_end

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

Marks the end of instrumentation.

Use it together with [torch_npu.npu.mstx.range_start](./torch_npu-npu-mstx-range_start.md).

## Prototype

```python
torch_npu.npu.mstx.range_end(range_id: int, domain: str='default') -> None
```

## Parameters

- **`range_id`** (`int`): Required. Unique identifier returned by the `torch_npu.npu.mstx.range_start` API.
- **`domain`** (`str`): Optional. Name of the domain where the end of a time segment event is marked. This parameter must match the `domain` configuration specified in the [torch_npu.npu.mstx.range_start](./torch_npu-npu-mstx-range_start.md) API.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
id = torch_npu.npu.mstx.range_start("dataloader", None)
dataloader()
torch_npu.npu.mstx.range_end(id)
```

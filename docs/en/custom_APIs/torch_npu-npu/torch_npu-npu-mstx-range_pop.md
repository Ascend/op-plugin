# torch_npu.npu.mstx.range_pop

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

Use it together with [torch_npu.npu.mstx.range_push](./torch_npu-npu-mstx-range_push.md).

## Prototype

```python
torch_npu.npu.mstx.range_pop(domain: str='default') -> int
```

## Parameters

**`domain`** (`str`): Optional. Name of the domain where the end of a time segment event is marked. This parameter must match the `domain` configuration specified in the `torch_npu.npu.mstx.range_push` API.

## Return Values

Nesting level of the range instrumentation recorded by the paired `torch_npu.npu.mstx.range_push` call within the current thread. If there is no paired `torch_npu.npu.mstx.range_push` call, the API call fails and returns `-1`.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
torch_npu.npu.mstx.range_push("dataloader")
dataloader()
torch_npu.npu.mstx.range_pop()
```

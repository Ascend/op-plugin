# get_trace_id

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Queries the current trace ID.

## Prototype

```python
get_trace_id(self)
```

## Parameters

None

## Return Values

Returns `trace_id`.

## Example

This API is not directly called. It provides the `trace_id` for the `set_custom_trace_id_callback` API. For details, see [set_custom_trace_id_callback](set_custom_trace_id_callback.md).

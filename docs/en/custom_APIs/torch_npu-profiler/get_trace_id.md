# get_trace_id

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas training products</term>: Supported
<!-- end id3 -->

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

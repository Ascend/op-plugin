# (beta) torch.distributed.ProcessGroupHCCL

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

Creates and returns a `ProcessGroupHCCL` object.

## Prototype

```python
torch.distributed.ProcessGroupHCCL(store, rank, size, timeout) -> ProcessGroup
```

## Parameters

- **`store`** (`torch.distributed.distributed_c10d.PrefixStore`): `PrefixStore` object created using its constructor.
- **`rank`**: Rank ID of the current node.
- **`size`**: Total number of communication nodes.
- **`timeout`**: Communication timeout used to detect node disconnection. The default value is `1800` (seconds).

## Return Values

`ProcessGroup`

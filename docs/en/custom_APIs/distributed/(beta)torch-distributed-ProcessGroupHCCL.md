# (beta) torch.distributed.ProcessGroupHCCL

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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

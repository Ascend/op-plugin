# (beta) torch_npu.contrib.Prefetcher

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Prototype

```python
torch_npu.contrib.Prefetcher(loader, stream=None)
```

## Function

Provides a data prefetcher on NPU devices, primarily used to optimize data loading workflows and improve training efficiency.

## Parameters

- **`loader`** (`torch.utils.data.DataLoader` or DataLoader-like iterator): Required. Preprocessed input data.
- **`stream`** (`torch.npu.Stream`): Optional. The default value is `None`. Due to NPU memory management constraints, you must specify a stream to prevent memory leaks if the prefetcher is initialized multiple times during training. If the prefetcher is initialized only once during training, you do not need to specify a stream, as a stream is created automatically.

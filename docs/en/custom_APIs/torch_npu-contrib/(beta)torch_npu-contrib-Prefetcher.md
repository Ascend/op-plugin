# (beta) torch_npu.contrib.Prefetcher

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Prototype

```python
torch_npu.contrib.Prefetcher(loader, stream=None)
```

## Function

Provides a data prefetcher on NPU devices, primarily used to optimize data loading workflows and improve training efficiency.

## Parameters

- **`loader`** (`torch.utils.data.DataLoader` or DataLoader-like iterator): Required. Preprocessed input data.
- **`stream`** (`torch.npu.Stream`): Optional. The default value is `None`. Due to NPU memory management constraints, you must specify a stream to prevent memory leaks if the prefetcher is initialized multiple times during training. If the prefetcher is initialized only once during training, you do not need to specify a stream, as a stream is created automatically.

## Example

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.contrib import Prefetcher
>>> # Create a DataLoader
>>> dataset = torch.utils.data.TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,)))
>>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
>>> # Initialize Prefetcher (initialize only once; no stream needs to be specified)
>>> prefetcher = Prefetcher(loader)
>>> # Retrieve data iteratively
>>> input, target = prefetcher.next()
>>> while input is not None:
...     # Perform training operations on input and target
...     input, target = prefetcher.next()
>>> # When initializing Prefetcher repeatedly (for example, when training for multiple epochs), specify a stream to prevent memory leaks
>>> stream = torch.npu.Stream()
>>> for epoch in range(10):
...     prefetcher = Prefetcher(loader, stream=stream)
...     input, target = prefetcher.next()
...     while input is not None:
...         # Perform training operations on input and target
...         input, target = prefetcher.next()
```

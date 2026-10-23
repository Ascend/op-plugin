# (beta) torch_npu.contrib.module.NpuCachedDropout

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

## Function

Executes a `FairseqDropout` operation on NPU devices.

## Prototype

```python
torch_npu.contrib.module.NpuCachedDropout(p, module_name=None)
```

## Parameters

- **`p`** (`float`): Probability that elements are zeroed.
- **`module_name`** (`string`): Module name.

## Constraints

Dynamic shapes are not supported.

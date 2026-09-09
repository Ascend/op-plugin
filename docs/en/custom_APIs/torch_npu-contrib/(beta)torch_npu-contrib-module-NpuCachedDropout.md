# (beta) torch_npu.contrib.module.NpuCachedDropout

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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

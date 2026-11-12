# (beta) torch_npu.contrib.NpuFairseqDropout

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Executes a `FairseqDropout` operation on NPU devices.

## Prototype

```python
torch_npu.contrib.NpuFairseqDropout(p, module_name=None)
```

## Parameters

- **`p`** (`float`): Probability that elements are zeroed.
- **`module_name`** (`str`): Optional. Module name used to identify the module to which the current Dropout belongs. The default value is `None`. This parameter is used only as an identifier and does not participate in the actual computation logic.

## Constraints

Dynamic shapes are not supported.

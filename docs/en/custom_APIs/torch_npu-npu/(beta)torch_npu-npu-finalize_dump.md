# (beta) torch_npu.npu.finalize_dump

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Ends dumping. Before setting this function, ensure that the environment variable `NPU_DUMP_ENABLE=1` has been set.

## Prototype

```python
torch_npu.npu.finalize_dump()
```

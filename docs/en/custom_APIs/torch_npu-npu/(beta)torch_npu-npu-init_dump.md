# (beta) torch_npu.npu.init_dump

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Initializes the `dump` configuration. Before setting this function, ensure that the environment variable `NPU_DUMP_ENABLE=1` has been set and dump is configured using `torch_npu.npu.set_dump_config(path="/tmp/dump", mode="all")`.

## Prototype

```python
torch_npu.npu.init_dump()
```

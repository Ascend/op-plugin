# (beta) torch_npu.npu.set_dump

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Configures `dump` parameters using a configuration file.

## Prototype

```python
torch_npu.npu.set_dump(path_to_json)
```

## Parameters

 **`path_to_json`**: Path to the configuration file, including the file name. Configure this parameter as needed. For details, see <a href="https://www.hiascend.com/document/detail/en/canncommercial/900/API/ascendgraphapi/aclpythondevg_01_0155.html">set_dump</a> in <i>CANN GE APIs</i>.

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
```

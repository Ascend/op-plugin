# (beta) torch_npu.npu.aclnn.version

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Queries the ACLNN operator version information. For details about the ACLNN operators, see "<a href="https://www.hiascend.com/document/detail/en/canncommercial/900/API/aolapi/operatorlist_00001.html">Overview</a>" in the <i>CANN Operator Library</i>.

## Prototype

```python
torch_npu.npu.aclnn.version(): -> None
```

## Constraints

Currently, ACLNN operators do not support version queries. The default value is `None`. Correct version information will be provided once ACLNN support is available.

## Example

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.version()
```

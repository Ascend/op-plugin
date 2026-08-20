# (beta) torch_npu.npu.finalize_dump

> [!NOTICE]  
> This API is updated in this version. For details about the specific changes, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950DT</term>            |    √     |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Ends the dump process and triggers a write of the dump data to the drive.

## Prototype

```python
torch_npu.npu.finalize_dump()
```

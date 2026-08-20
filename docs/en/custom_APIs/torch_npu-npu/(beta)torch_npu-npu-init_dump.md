# (beta) torch_npu.npu.init_dump

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

Initializes the dump configuration as the entry point of the dump workflow.

The correct call sequence is: `init_dump()` → `set_dump(cfg_file)` → execute the model → `finalize_dump()`. If this API is not called first, `set_dump` and `finalize_dump` will report an error because the dump has not been initialized.

## Prototype

```python
torch_npu.npu.init_dump()
```

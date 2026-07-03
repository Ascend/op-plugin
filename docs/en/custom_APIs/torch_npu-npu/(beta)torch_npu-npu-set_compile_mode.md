# (beta) torch_npu.npu.set_compile_mode

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Controls whether to enable binary mode.

## Prototype

```python
torch_npu.npu.set_compile_mode(jit_compile = bool)
```

## Parameters

**`jit_compile`** (`bool`): Valid values are `True` (non-binary mode) or `False` (binary mode).

> [!NOTE]  
>
>- For Atlas training products/Atlas inference products, the default configuration is `jit_compile=True`, indicating non-binary mode.
>- For Atlas A2 training products/Atlas A3 training products, the default configuration is `jit_compile=False`, indicating binary mode.

## Example

```python
>>> torch_npu.npu.set_compile_mode(jit_compile=False)
```

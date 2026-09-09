# (beta) torch_npu.npu.set_compile_mode

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Sets whether to enable JIT compilation.

## Prototype

```python
torch_npu.npu.set_compile_mode(jit_compile = bool)
```

## Parameters

**`jit_compile`** (`bool`): Valid values are `True` (enables JIT compilation) or `False` (disables JIT compilation).

> [!NOTE]  
>
>- For Atlas training products/Atlas inference products, the default configuration is `jit_compile=True`, which enables JIT compilation.
>- For Atlas A2 training products/Atlas A3 training products, the default configuration is `jit_compile=False`, which disables JIT compilation.

## Example

```python
>>> torch_npu.npu.set_compile_mode(jit_compile=False)
```

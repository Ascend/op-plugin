# (beta) torch_npu.npu.set_compile_mode

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Specifies whether to enable JIT compilation.

## Prototype

```python
torch_npu.npu.set_compile_mode(jit_compile = bool)
```

## Parameters

**`jit_compile`** (`bool`): If the value is True, JIT compilation is enabled. If the value is False, JIT compilation is disabled.

> [!NOTE]  
>
><!-- npu="910,310p" id5 -->
>- For Atlas training products and Atlas inference products, the default value is `jit_compile=True`, that is, JIT compilation is enabled.
><!-- end id5 -->
><!-- npu="A3,910b" id6 -->
>- For Atlas A2 training products and Atlas A3 training products, the default value is `jit_compile=False`, that is, JIT compilation is disabled.
><!-- end id6 -->

## Examples

```python
>>> torch_npu.npu.set_compile_mode(jit_compile=False)
```

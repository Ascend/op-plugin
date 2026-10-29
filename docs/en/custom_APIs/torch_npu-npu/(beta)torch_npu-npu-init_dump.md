# (beta) torch_npu.npu.init_dump

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

Initializes the dump configuration as the entry point of the dump workflow.

The correct call sequence is: `init_dump()` → `set_dump(cfg_file)` → execute the model → `finalize_dump()`. If this API is not called first, `set_dump` and `finalize_dump` will report an error because the dump has not been initialized.

## Prototype

```python
torch_npu.npu.init_dump()
```

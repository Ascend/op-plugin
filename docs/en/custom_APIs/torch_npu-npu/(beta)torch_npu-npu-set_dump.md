# (beta) torch_npu.npu.set_dump

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

Configures `dump` parameters using a configuration file.

## Prototype

```python
torch_npu.npu.set_dump(path_to_json)
```

## Parameters

 **`path_to_json`**: Path to the configuration file, including the file name. Configure this parameter as required. For details about the configuration, see the "<a href="https://gitcode.com/cann/runtime/blob/9.2.0/python/docs/en/api_ref/runtime/dump/function-set_dump.md">Function: set_dump</a>" section in the *CANN Runtime API Reference*.

## Example

```python
>>> import torch
>>> import torch_npu
>>>
>>> # 1. Initialize the dump process
>>> torch_npu.npu.init_dump()
>>>
>>> # 2. Specify the dump configuration file path
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. Run model inference (example)
>>> # output = model(input_data)
>>>
>>> # 4. Finalize the dump process
>>> torch_npu.npu.finalize_dump()
```

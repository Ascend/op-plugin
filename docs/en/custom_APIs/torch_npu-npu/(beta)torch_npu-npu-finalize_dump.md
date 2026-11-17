# (beta) torch_npu.npu.finalize_dump

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

Ends the dump process and triggers a write of the dump data to the drive.

## Prototype

```python
torch_npu.npu.finalize_dump()
```

## Example

```python
>>> import torch
>>> import torch_npu
>>>
>>> # 1. Initialize dump.
>>> torch_npu.npu.init_dump()
>>>
>>> # 2. Specify the dump configuration file path. The actual path of dump.json is user-defined and is used here only as an example.
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. Run model inference (example).
>>> # output = model(input_data)
>>>
>>> # 4. Finalize dump.
>>> torch_npu.npu.finalize_dump()
```

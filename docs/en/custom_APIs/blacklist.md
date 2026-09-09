# Example of Adding a Binary Blocklist

See the following examples to set the `NPU_FUZZY_COMPILE_BLACKLIST` option to add a binary blocklist.

Example for a single operator:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"       # Configure based on the actual situation
torch.npu.set_option(option)
```

Example for multiple operators:

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"          # Configure based on the actual situation
torch.npu.set_option(option)
```

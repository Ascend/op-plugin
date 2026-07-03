# (beta) torch_npu.utils.save_async

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Asynchronously saves an object to a file on a drive.

## Prototype

```python
torch_npu.utils.save_async(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True, _disable_byteorder_record=False, model=None)
```

## Parameters

- **`obj`** (`object`): Object to be saved.
- **`f`** (`Union[str, PathLike, BinaryIO, IO[bytes]]`): Path or handle of the target file to save.
- **`pickle_module`** (`Any`): Module used for pickle serialization. The default value is `pickle`.
- **`pickle_protocol`** (`int`): Pickle protocol version. The default value is `DEFAULT_PROTOCOL`.
- **`_use_new_zipfile_serialization`** (`bool`): Specifies whether to enable new zipfile-based serialization. The default value is `True`.
- **`_disable_byteorder_record`** (`bool`): Specifies whether to disable byte-order recording. The default value is `False`.
- **`model`**: Model object. A reverse hook will be registered on the model (`torch.nn.Module`). The default value is `None`.

## Example

```python
import torch
import torch.nn as nn
import os
import torch_npu

input = torch.tensor([1.,2.,3.,4.]).npu()
torch_npu.utils.save_async(input, "save_tensor.pt")
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU()
)
model = model.npu()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(3):
    for step in range(3):
        input_data = torch.ones(6400, 100).npu()
        labels = torch.randint(0, 5, (6400,)).npu()
        outputs = model(input_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_path = os.path.join(f"model_{epoch}_{step}.path")
        torch_npu.utils.save_async(model, save_path, model=model)
```

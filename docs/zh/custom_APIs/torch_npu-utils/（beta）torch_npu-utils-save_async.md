# （beta）torch\_npu.utils.save\_async

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

异步将对象保存到指定文件。

## 函数原型

```python
torch_npu.utils.save_async(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True, _disable_byteorder_record=False, model=None)
```

## 参数说明

- **obj** (`object`)：必选参数，要保存的对象。
- **f** (`Union[str, PathLike, BinaryIO, IO[bytes]]`)：必选参数，保存的目标文件路径或文件句柄。
- **pickle_module** (`Any`)：可选参数，用于Pickle序列化的模块，默认值为pickle。
- **pickle_protocol** (`int`)：可选参数，Pickle协议版本，默认值为DEFAULT\_PROTOCOL。
- **_use_new_zipfile_serialization** (`bool`)：可选参数，是否使用新的ZIP文件序列化，默认值为True。
- **_disable_byteorder_record** (`bool`)：可选参数，是否禁用字节顺序记录，默认值为False。
- **model** (`torch.nn.Module`)：可选参数，模型对象，默认值为None；传入torch.nn.Module时，接口会在首次为当前设备创建异步保存流时注册反向hook。

## 调用示例

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

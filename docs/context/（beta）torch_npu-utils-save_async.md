# （beta）torch\_npu.utils.save\_async
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

异步保存一个对象到一个硬盘文件上。

## 函数原型

```
torch_npu.utils.save_async(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True, _disable_byteorder_record=False, model=None)
```

## 参数说明

- **obj** (`object`)：要保存的对象。
- **f** \(Union\[str, PathLike, BinaryIO, IO\[bytes\]\]\)：保存的目标文件路径或文件句柄。
- **pickle_module** (Any)：用于Pickle序列化的模块，默认值为pickle。
- **pickle_protocol** (`int`)：Pickle协议版本，默认值为DEFAULT\_PROTOCOL。
- **_use_new_zipfile_serialization** (`bool`)：是否使用新的Zip文件序列化，默认值为True。
- **_disable_byteorder_record** (`bool`)：是否禁用字节顺序记录，默认值为False。
- **model**：模型对象，需要对模型反向注册hook，torch.nn.Module，默认值为None。


## 约束说明
该接口仅支持PyTorch 2.1.0版本。

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


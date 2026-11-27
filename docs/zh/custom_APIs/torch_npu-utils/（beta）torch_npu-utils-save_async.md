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
- **f** (`Union[str, BinaryIO, IO[bytes]]`)：必选参数，保存的目标文件路径或支持写入二进制数据的文件对象。传入`pathlib.Path`时，先用`str(f)`转为字符串。
- **pickle_module** (`Any`)：可选参数，用于Pickle序列化的模块，默认值为pickle。
- **pickle_protocol** (`int`)：可选参数，Pickle协议版本，默认值为DEFAULT\_PROTOCOL。
- **_use_new_zipfile_serialization** (`bool`)：可选参数，默认值为True。异步保存要求使用ZIP序列化，传入False会抛出`RuntimeError`。
- **_disable_byteorder_record** (`bool`)：可选参数，是否禁用字节顺序记录，默认值为False。
- **model** (`torch.nn.Module`)：可选参数，模型对象，默认值为None；传入torch.nn.Module时，接口会在首次为当前设备创建异步保存流时注册反向hook。

## 调用示例

以下示例把模型的参数状态异步保存到指定路径。请在当前设备首次调用时传入模型，以注册反向钩子。

```python
import torch
import torch_npu

model = torch.nn.Linear(4, 2).npu()
torch_npu.utils.save_async(model.state_dict(), "model.pt", model=model)
```

示例在后台生成`model.pt`。该接口启动保存线程后即返回。

# （beta）torch_npu.npu.set_dump

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950DT</term>            |    √     |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

传入配置文件来配置dump参数。

## 函数原型

```python
torch_npu.npu.set_dump(path_to_json)
```

## 参数说明

 **path_to_json**：配置文件所在的路径，包含文件名，用户需根据实际情况配置。具体配置请参考《CANN Runtime运行时API》中函数：“set_dump”章节。
 <!-- 《CANN Runtime运行时 API》中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclpythondevg_01_0155.html">函数：set_dump</a>” -->

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>>
>>> # 1. 初始化Dump
>>> torch_npu.npu.init_dump()
>>>
>>> # 2. 指定Dump 配置文件路径
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. 执行模型推理（示例）
>>> # output = model(input_data)
>>>
>>> # 4. 结束Dump
>>> torch_npu.npu.finalize_dump()
```

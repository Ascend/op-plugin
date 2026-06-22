# （beta）torch_npu.npu.set_dump

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas 350 加速卡</term>            |    √     |
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

 **path_to_json**：配置文件所在的路径，包含文件名，用户需根据实际情况配置。具体配置请参考《CANN Runtime运行时 API》中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclpythondevg_01_0155.html">函数：set_dump</a>”章节。

## 调用示例

```python
>>> import os
>>> import torch
>>> import torch_npu
>>>
>>> # 1. 设置环境变量（需在启动脚本中设置，或代码中提前设置）
>>> os.environ["NPU_DUMP_ENABLE"] = "1"
>>>
>>> # 2. 指定Dump 配置文件路径
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. 启用Dump
>>> torch_npu.npu.init_dump()
>>>
>>> # 4. 执行模型推理（示例）
>>> # output = model(input_data)
>>>
>>> # 5. 结束Dump
>>> torch_npu.npu.finalize_dump()
```

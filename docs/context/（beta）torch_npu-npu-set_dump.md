# （beta）torch_npu.npu.set_dump

## 函数原型

```
torch_npu.npu.set_dump(path_to_json)
```

## 功能说明

传入配置文件来配置dump参数。

## 参数说明

path_to_json：配置文件所在的路径，包含文件名，用户需根据实际情况配置。具体配置请参考《CANN AscendCL应用软件开发指南 (Python)》中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0155.html">函数：set_dump</a>”章节。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>>torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
```


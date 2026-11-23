# （beta）torch_npu.npu.set_aoe

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

启用AOE（Ascend Optimization Engine，昇腾调优引擎）调优。

## 函数原型

```python
torch_npu.npu.set_aoe(dump_path)
```

## 参数说明

**dump_path**（`str`）：必填参数，指定保存Dump算子图的目录，如`./aoe_dump`。目录不存在时，接口尝试创建目录。请确保具备相应的创建和写入权限。

## 调用示例

```python
import torch
import torch_npu
import os

os.mkdir("./aoe_dump")
dump_path = "./aoe_dump"
torch_npu.npu.set_aoe(dump_path)

```

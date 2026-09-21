# （beta）torch.distributed.ProcessGroupHCCL

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

创建一个ProcessGroupHCCL对象并返回。

## 函数原型

```python
torch.distributed.ProcessGroupHCCL(store, rank, size, timeout) -> ProcessGroup
```

## 参数说明

- **store**：`torch.distributed.distributed_c10d.PrefixStore`对象，可以通过构造函数构造。
- **rank**：当前节点的rank序号。
- **size**：全部通讯节点的数量。
- **timeout**：超时时间，用于判断节点断连，默认值为1800s。

## 返回值说明

`ProcessGroup`

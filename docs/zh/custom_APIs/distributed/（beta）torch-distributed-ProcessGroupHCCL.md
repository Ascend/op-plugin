# （beta）torch.distributed.ProcessGroupHCCL

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

创建一个ProcessGroupHCCL对象并返回。

## 函数原型

```python
torch_npu._C._distributed_c10d.ProcessGroupHCCL(store, rank, size, timeout=datetime.timedelta(seconds=1800)) -> ProcessGroupHCCL
```

## 参数说明

- **store** (`torch.distributed.Store`)：必选参数，用于进程间交换通信初始化信息，例如`PrefixStore`对象。
- **rank** (`int`)：必选参数，当前进程在组内的编号，范围为`0`至`size - 1`。
- **size** (`int`)：必选参数，组内进程总数，为正整数。
- **timeout** (`datetime.timedelta`)：可选参数，通信超时时间，默认值为`datetime.timedelta(seconds=1800)`。

## 返回值说明

返回HCCL后端对象，类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL`，继承自`torch.distributed.Backend`。

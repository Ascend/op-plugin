# （beta）torch_npu.contrib.NpuFairseqDropout

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

在NPU设备上使用FairseqDropout。

## 函数原型

```python
torch_npu.contrib.NpuFairseqDropout(p, module_name=None)
```

## 参数说明

- **p** (`float`)：元素归零的概率。
- **module_name** (`str`)：可选参数，模块名称，用于标识当前Dropout所属的模块，默认为`None`。该参数仅作标识用途，不参与实际计算逻辑。

## 约束说明

不支持动态shape。

# （beta）torch_npu.contrib.module.NpuCachedDropout

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

在NPU设备上使用FairseqDropout。

## 函数原型

```python
torch_npu.contrib.module.NpuCachedDropout(p, module_name=None)
```

## 参数说明

- **p** (`float`)：元素归零的概率。
- **module_name** (`string`)：模块名称。

## 约束说明

不支持动态shape。

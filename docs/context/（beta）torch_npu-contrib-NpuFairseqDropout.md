# （beta）torch_npu.contrib.NpuFairseqDropout

## 函数原型

```
torch_npu.contrib.NpuFairseqDropout(p, module_name=None)
```

## 功能说明

在NPU设备上使用FairseqDropout。

## 参数说明

- input_size：对输入期望的特征数量。
- hidden_size：hidden state中的特征数量。

## 约束说明

不支持动态shape。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


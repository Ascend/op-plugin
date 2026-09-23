# （beta）torch_npu.contrib.BiLSTM

> [!NOTICE]  
>该接口计划废弃，可以参考[小算子拼接方案](https://gitee.com/ascend/ModelZoo-PyTorch/blob/732cb7fc5ab59249ae62a905c0d43400a8250da7/PyTorch/contrib/audio/deepspeech/deepspeech_pytorch/bidirectional_lstm.py#L18)进行替换。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

将NPU兼容的双向LSTM操作应用于输入序列。

## 函数原型

```python
torch_npu.contrib.BiLSTM(input_size, hidden_size)
```

## 参数说明

- **input_size**：对输入期望的特征数量。
- **hidden_size**：hidden state中的特征数量。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> r = torch_npu.contrib.BiLSTM(512, 256).npu()
>>> input_tensor = torch.randn(26, 2560, 512).npu()
>>> output = r(input_tensor)
```

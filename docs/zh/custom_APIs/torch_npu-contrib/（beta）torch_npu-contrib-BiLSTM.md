# （beta）torch_npu.contrib.BiLSTM

> [!NOTICE]  
>该接口计划废弃，可以参考[小算子拼接方案](https://gitee.com/ascend/ModelZoo-PyTorch/blob/732cb7fc5ab59249ae62a905c0d43400a8250da7/PyTorch/contrib/audio/deepspeech/deepspeech_pytorch/bidirectional_lstm.py#L18)进行替换。底层算子内核已停止维护，性能和精度无法保证，建议使用替代方案。

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

- **input_size** (`int`)：必选参数，输入每个时间步的特征数，为正整数，须等于输入末维大小。
- **hidden_size** (`int`)：必选参数，每个方向的隐藏状态特征数，为正整数。

## 调用示例

输入形状按序列长度、批大小、特征数排列。示例的输入特征数为512，每个方向的隐藏状态特征数为256。

```python
import torch
import torch_npu
r = torch_npu.contrib.BiLSTM(512, 256).npu()
input_tensor = torch.randn(26, 2560, 512).npu()
output = r(input_tensor)
```

双向输出在最后一维拼接，得到形状 `(26, 2560, 512)`的张量。

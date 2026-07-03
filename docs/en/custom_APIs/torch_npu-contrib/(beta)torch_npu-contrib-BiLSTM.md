# (beta) torch_npu.contrib.BiLSTM

> [!NOTICE]  
>This API is planned for deprecation. For details about the replacement, see [Small Operator Concatenation Solution](https://gitee.com/ascend/ModelZoo-PyTorch/blob/732cb7fc5ab59249ae62a905c0d43400a8250da7/PyTorch/contrib/audio/deepspeech/deepspeech_pytorch/bidirectional_lstm.py#L18).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies NPU-compatible bidirectional LSTM operations on input sequences.

## Prototype

```python
torch_npu.contrib.BiLSTM(input_size, hidden_size)
```

## Parameters

- **`input_size`**: Expected number of features in the input.
- **`hidden_size`**: Number of features in the hidden state.

## Example

```python
>>> import torch
>>> import torch_npu
>>> r = torch_npu.contrib.BiLSTM(512, 256).npu()
>>> input_tensor = torch.randn(26, 2560, 512).npu()
>>> output = r(input_tensor)
```

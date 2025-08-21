

# torch_npu.npu.obfuscation_finalize

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>           |    √     |
|<term>Atlas 推理系列产品 </term> | √   |

## 功能说明

该接口用于完成PMCC模型混淆引擎的资源释放。

## 函数原型

```
torch_npu.obfuscation_finalize(Tensor fd_to_close) -> Tensor
```

## 参数说明

- **fd_to_close**（`Tensor`）：填写调用obfuscation_initialize返回的fd。数据类型为int32。

## 调用示例

```python
import torch
import torch_npu

device = "npu:0"
hidden_size = int(3584)
cmd = 1
data_type = torch.bfloat16
model_obf_seed = 0
data_obf_seed = 0
thread_num = 4
tp_rank = 0
i = 0
hidden_states = torch.randn((1024,3584), dtype=torch.bfloat16, device=device)
obf_cft = 1.0
fd = torch_npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num= thread_num, obf_coefficient=obf_cft)
torch_npu.obfuscation_finalize(fd)
```
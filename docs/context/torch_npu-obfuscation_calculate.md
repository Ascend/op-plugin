

# torch_npu.npu.obfuscation_calculate

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>           |    √     |
|<term>Atlas 推理系列产品 </term> | √   |

## 功能说明

该接口用于实现矩阵乘计算输入和输出的transpose操作。

## 函数原型

```
torch_npu.obfuscation_calculate(Tensor fd, Tensor x, Tensor param, float obf_coefficient) -> Tensor
```

## 参数说明

- **x**（`Tensor`）：必选参数,待混淆处理的Tensor输入,数据类型如下,对Tensor维度不作限制,Shape为( , *, ... , hiddenSize),即最后一维的size是hiddenSize。数据格式支持ND。
    * <term>Atlas 推理系列产品</term>: Tensor数据类型支持torch.float16 / torch.float32 / torch.int8
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>: Tensor数据类型支持torch.float16 / torch.float32 / torch.bfloat16 / torch.int8
- **param**（`Tensor`）：必选参数,预留的参数字段,Tensor数据类型为int32。
- **obf_coefficient**（`Float`）：可选参数,混淆系数,数据类型为float,支持输入范围为0-1,默认值1.0。

## 返回值

- **y**（`Tensor`）：混淆处理后的张量,输出数据类型及Shape与x相同。

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
param = torch.tensor([3584], device=device)
x_obf_out = torch_npu.obfuscation_calculate(fd, hidden_states, param, obf_coefficient=obf_cft)
```
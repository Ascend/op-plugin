

# torch_npu.npu.obfuscation_calculate

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>           |    √     |
|<term>Atlas 推理系列产品 </term> | √   |

## 功能说明

该接口用于将张量x和配置参数（如param）发送至PMCC（Privacy&Model Confidential Computing）混淆引擎。引擎的CA（普通OS中的Client Application）模块调用TA（TEE OS中的Trusted Application）模块，进行张量混淆处理，最终返回混淆结果。

## 函数原型

```
torch_npu.npu.obfuscation_calculate(fd, x, param, obf_coefficient) -> Tensor
```

## 参数说明

- **fd**（`Tensor`）：必选参数，socket连接符，数据类型为`int32`，填写调用[obfuscation_initialize](./torch_npu-npu-obfuscation_initialize.md)接口的返回值。
- **x**（`Tensor`）：必选参数，待混淆处理的`Tensor`输入，对`Tensor`维度不作限制，shape为( , *, ... , hiddenSize)，即最后一维的size是[obfuscation_initialize](./torch_npu-npu-obfuscation_initialize.md)的入参`hiddenSize`。数据格式支持ND。
    * <term>Atlas 推理系列产品</term>: `Tensor`数据类型支持`float16` 、`float32`、`int8`。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>: `Tensor`数据类型支持`float16`、`float32`、`bfloat16`、`int8`。
- **param**（`Tensor`）：必选参数，张量`x`的最后一维的维度，数据类型为`int32`。
- **obf_coefficient**（`float`）：可选参数，混淆系数，支持输入范围为(0.0，1.0]，默认值1.0。

## 返回值说明
`Tensor`

代表`obfuscation_calculate`的计算结果，输出数据类型及shape与`x`相同。

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
fd = torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num= thread_num, obf_coefficient=obf_cft)
param = torch.tensor([3584], device=device)
x_obf_out = torch_npu.npu.obfuscation_calculate(fd, hidden_states, param, obf_coefficient=obf_cft)
```


# torch_npu.npu.obfuscation_initialize

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>           |    √     |
|<term>Atlas 推理系列产品 </term> | √   |

## 功能说明

该接口用于完成PMCC（Privacy&Model Confidential Computing）模型混淆引擎的资源初始化，即与PMCC混淆引擎CA（普通OS中的Client Application）建立socket连接、对CA、TA（TEE OS中的Trusted Application）进行初始化，并返回socket连接符。

## 函数原型

```
torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type, model_obf_seed_id, data_obf_seed_id, thread_num, obf_coefficient) -> Tensor
```

## 参数说明

- **hidden_size**（`int`）：必选参数，隐藏层的维度，数据类型为`int32`，支持输入范围为1-10000，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
- **tp_rank**（`int`）：必选参数， 张量并行TP Rank，数据类型为`int32`，支持输入范围为0-1024，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
- **cmd**（`int`）：必选参数，资源初始化的指令编号，数据类型为`int32`，取值范围为{1, 2, 3}。
    * 1：进行浮点推理模式资源初始化。
    * 2：进行量化推理模式资源初始化。
    * 3：进行资源释放。
- **data_type**（`int`）：可选参数， 代表Tensor数据类型的编号，数据类型为`int32`，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
    * <term>Atlas 推理系列产品</term>: Tensor数据类型支持`float16` 、`float32`、`int8`。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>: Tensor数据类型支持`float16`、`float32`、`bfloat16`、`int8`。
- **model_obf_seed_id**（`int`）：可选参数， 模型混淆因子id，用于`TA`从`TEE KMC`查询模型混淆因子，数据类型为`int32`，仅在`cmd`设置为1或2时需要填写已注册的有效混淆因子id，否则填0。
- **data_obf_seed_id**（`int`）：必选参数， 数据混淆因子id，用于`TA`从`TEE KMC`查询数据混淆因子，数据类型为`int32`，仅在`cmd`设置为1或2时需要填写已注册的有效混淆因子id，否则填0。
- **thread_num**（`int`）：可选参数， `CA`/`TA`进行混淆处理使用的线程数，数据类型为`int32`，取值范围为{1, 2, 3, 4, 5, 6}，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
- **obf_coefficient**（`float`）：可选参数，混淆系数，支持输入范围为0-1，默认值1.0。

## 返回值说明
`Tensor`

代表socket连接符，1D，shape为(1)，数据类型为`int32`。

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
```
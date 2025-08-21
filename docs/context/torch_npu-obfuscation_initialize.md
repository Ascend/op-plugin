

# torch_npu.npu.obfuscation_initialize

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>           |    √     |
|<term>Atlas 推理系列产品 </term> | √   |

## 功能说明

该接口用于完成PMCC模型混淆引擎的资源初始化。

## 函数原型

```
torch_npu.obfuscation_initialize(int hidden_size, int tp_rank, int cmd, int data_type, int model_obf_seed_id, int data_obf_seed_id, int thread_num, float obf_coefficient) -> Tensor
```

## 参数说明

- **hidden_size**（`Int`）：必选参数,隐藏层的维度,数据类型为int32,支持输入范围为1-10000,仅在cmd设置为1或2时需要填写有效值,否则填0。
- **tp_rank**（`Int`）：必选参数, 张量并行TP Rank,数据类型为int32,支持输入范围为0-1024,仅在cmd设置为1或2时需要填写有效值,否则填0。
- **cmd**（`Int`）：必选参数, setup指令编号,在{1, 2, 3}中选择,设置为1时进行浮点推理模式资源初始化、为2时进行量化推理模式资源初始化,设置为3时进行资源释放。数据类型为int32。
- **data_type**（`Int`）：可选参数, 代表Tensor数据类型的编号,数据类型为INT32,仅在cmd设置为1或2时需要填写有效值,否则填0。
    * <term>Atlas 推理系列产品</term>: Tensor数据类型支持torch.float16 / torch.float32 / torch.int8
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>: Tensor数据类型支持torch.float16 / torch.float32 / torch.bfloat16 / torch.int8
- **model_obf_seed_id**（`Int`）：可选参数, 模型混淆因子id,用于TA从TEE KMC查询模型混淆因子,数据类型为int32,仅在cmd设置为1或2时需要填写有效值,否则填0。
- **data_obf_seed_id**（`Int`）：必选参数, 数据混淆因子id,用于TA从TEE KMC查询数据混淆因子,数据类型为int32,仅在cmd设置为1或2时需要填写有效值,否则填0。
- **thread_num**（`Int`）：可选参数, CA/TA进行混淆处理使用的线程数。在{1, 2, 3, 4, 5, 6}中选择,数据类型为int32,仅在cmd设置为1或2时需要填写有效值,否则填0。
- **obf_coefficient**（`Float`）：可选参数,混淆系数,数据类型为float,支持输入范围为0-1,默认值1.0。

## 返回值

- **fd**（`Tensor`）：socket连接符,1D,shape为(1),int32。

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
```
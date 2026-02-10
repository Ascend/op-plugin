# torch_npu.npu.use_compatible_impl

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term>                        |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

该接口用于控制算子API的实现是否与PyTorch社区完全对齐。该接口仅用于切换算子API所调用的底层算子。
该功能是否开启，可通过`torch_npu.npu.are_compatible_impl_enabled`进行查询。

## 函数原型

```
torch_npu.npu.use_compatible_impl(is_enable)
```

## 参数说明

**is_enable**(`bool`)：控制是否开启兼容性实现。

 - True: 开启兼容性算子替换。
 - False: 关闭兼容性算子替换。

## 返回值说明

无

## 约束说明

目前仅支持`torch.nn.functional.gelu`。

## 调用示例

```python
import torch
import torch_npu

torch_npu.npu.use_compatible_impl(True)
shape = [100, 400]
mode = "none"
input = torch.rand(shape, dtype=torch.float16).npu()
output = torch.nn.functional.gelu(input, approximate=mode)
```
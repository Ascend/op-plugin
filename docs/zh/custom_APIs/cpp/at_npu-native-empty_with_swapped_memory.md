# at_npu::native::empty_with_swapped_memory

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

申请一个device信息为NPU且实际内存在host侧的特殊tensor。

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```cpp
at::Tensor empty_with_swapped_memory(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<c10::Device> device_opt)
```

## 参数说明

- **size** (`c10::IntArrayRef`)：必选参数，表示生成Tensor的shape。
- **dtype_opt** (`c10::optional<at::ScalarType>`)：可选参数，表示生成Tensor的数据类型，若为`c10::nullopt`，则表示使用dtype全局默认值。
- **device_opt** (`c10::optional<c10::Device>`)：可选参数，表示生成Tensor的设备信息，若为`c10::nullopt`，则表示使用当前默认device。

## 返回值说明

`at::Tensor`

代表生成的特殊Tensor。

## 约束说明

- 该接口暂不支持图模式。

- 该接口申请的特殊Tensor当前仅支持如下算子：<br>
`aten::fill_`<br>
`aten::zero_`<br>
`aten::mul_`<br>
`npu_apply_adam_w`<br>
`npu_hans_encode`<br>
`npu_hans_decode`<br>

- 当安装CANN版本8.5.0及以上，且Ascend HDK版本26.0.rc1及以上时，该接口申请的特殊Tensor支持直接打印。
- 当安装CANN版本小于8.5.0或者Ascend HDK版本小于26.0.rc1时，该接口申请的特殊Tensor不支持直接打印，此时会打印warning日志。若要查看值，请创建一个与该特殊张量具有相同shape、dtype和device的常规NPU张量，将其填充为1，以该特殊张量作为操作数对其调用`mul_`，然后打印该常规张量即可。

## 调用示例

```cpp
#include <iostream>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

int main() {
    // 创建指定shape和dtype的特殊Tensor（swapped memory格式）
    at::Tensor tensor = at_npu::native::empty_with_swapped_memory({2, 3}, at::kFloat, c10::nullopt);
    tensor.fill_(3.14);

    // 创建与特殊Tensor同shape、同dtype的普通NPU张量，并填充为1
    auto out = at::empty(tensor.sizes(), tensor.options());
    out.fill_(1.0);

    // 以特殊Tensor为操作数做原地乘法，结果写入普通NPU张量out
    out.mul_(tensor);

    std::cout << out << '\n';
    return 0;
}
```

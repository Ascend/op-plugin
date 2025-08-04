# at_npu::native::empty_with_swapped_memory
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

申请一个device信息为NPU且实际内存在host侧的特殊Tensor。

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```
at::Tensor empty_with_swapped_memory(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<c10::Device> device_opt)
```

## 参数说明

- **size** (`c10::IntArrayRef`)：必选参数，表示生成Tensor的shape。
- **dtype_opt** (`c10::optional<at::ScalarType>`)：必选参数，表示生成Tensor的数据类型，若为`c10::nullopt`，则表示使用dtype全局默认值。
- **device_opt** (`c10::optional<c10::Device>`)：必选参数，表示生成Tensor的设备信息，若为`c10::nullopt`，则表示使用当前默认device。



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

- 该接口申请的特殊Tensor不支持直接打印，需要查看值时要先通过`mul_`转为普通Tensor再打印。

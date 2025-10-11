# torch_npu.utils.get_cann_version
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>   | √   |

## 功能说明

获取当前环境CANN或相关组件的版本号。

## 函数原型

```
torch_npu.utils.get_cann_version(module="CANN")
```

## 参数说明

**module**(`str`)：可选参数，指定需要获取版本号的组件，默认值为"CANN"。支持的可选参数为["CANN", "RUNTIME", "COMPILER", "HCCL", "TOOLKIT", "OPP", "OPP_KERNEL", "DRIVER"]。

可选参数说明：

- "CANN"：CANN(Compute Architecture for Neural Networks)是昇腾针对AI场景推出的异构计算架构。

- "RUNTIME"：runtime组件。

- "COMPILER"：编译器。

- "HCCL"：集合通信库。

- "TOOLKIT"：开发工具包。

- "OPP"：算子包。

- "OPP_KERNEL"：二进制算子包。

- "DRIVER"：驱动。


## 返回值说明
`str`

代表具体组件的版本号。

当传入的module无效时，会返回空字符串。

## 约束说明

1. DRIVER的版本号获取是根据/etc/ascend_install.info; /usr/local/Ascend/driver/version.info中的信息获取的。在容器中获取该版本号时，需要保证容器里映射了这两个文件。
2. CANN版本号小于8.1.RC1时，该功能不支持，会返回空字符串。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.utils import get_cann_version
>>> version = get_cann_version(module="CANN")
>>> version
'8.3.RC1'
```
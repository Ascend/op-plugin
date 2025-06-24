# c10_npu::NPUStreamGuard::original_device

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

返回guard构造时的设备。

## 定义文件

torch_npu\csrc\core\npu\NPUGuard.h

## 函数原型

```
c10::Device c10_npu::NPUStreamGuard::original_device() const
```

## 参数说明

无

## 返回值说明

`c10::Device`

表示guard构造时的设备。

## 约束说明

无

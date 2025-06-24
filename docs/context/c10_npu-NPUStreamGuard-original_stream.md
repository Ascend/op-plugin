# c10_npu::NPUStreamGuard::original_stream

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

返回guard构造时设置的流。

## 定义文件

torch_npu\csrc\core\npu\NPUGuard.h

## 函数原型

```
c10_npu::NPUStream c10_npu::NPUStreamGuard::original_stream() const
```

## 参数说明

无

## 返回值说明

`c10_npu::NPUStream`

表示构造时设置的流。

## 约束说明

无
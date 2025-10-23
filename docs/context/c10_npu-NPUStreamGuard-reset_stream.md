# c10_npu::NPUStreamGuard::reset_stream

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

给guard重新设置新的流。

## 定义文件

torch_npu\csrc\core\npu\NPUGuard.h

## 函数原型

```
void c10_npu::NPUStreamGuard::reset_stream(c10::Stream stream)
```

## 参数说明

**stream** (`c10::Stream`)：必选参数，表示guard准备保障的流。

## 返回值说明

无

## 约束说明

无
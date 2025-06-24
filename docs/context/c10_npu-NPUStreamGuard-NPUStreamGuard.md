# c10_npu::NPUStreamGuard::NPUStreamGuard

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

构造函数，创建一个流guard。

## 定义文件

torch_npu\csrc\core\npu\NPUGuard.h

## 函数原型

```
c10_npu::NPUStreamGuard::NPUStreamGuard(c10::Stream stream)
```

## 参数说明

**stream** (`c10::Stream`)：必选参数，表示guard保障的流。

## 返回值说明

无

## 约束说明

无
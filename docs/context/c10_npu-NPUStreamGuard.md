# c10_npu::NPUStreamGuard

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

NPU设备流guard，保障作用域内的设备流，与`c10::cuda::CUDAStreamGuard`相同。

## 定义文件

torch_npu\csrc\core\npu\NPUGuard.h

## 函数原型

```
struct c10_npu::NPUStreamGuard
```


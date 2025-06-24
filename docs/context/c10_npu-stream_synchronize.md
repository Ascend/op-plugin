# c10_npu::stream_synchronize
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term> Atlas A3 训练系列产品</term>             |    √     |
|<term> Atlas A2 训练系列产品</term>   | √   |


## 功能说明

NPU设备流同步，与`c10::cuda::stream_synchronize`相同。

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```
void stream_synchronize(aclrtStream stream)
```

## 参数说明

**stream** (`aclrtStream`)：必选参数，表示需要同步的流。

## 返回值说明

无

## 约束说明

无
# c10_npu::NPUStreamGuard::reset_stream

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

给guard重新设置新的NPU流。将当前设置的流重置为原始流，并将当前设置的设备重置为原始设备。然后，将当前设备设置为与传入流关联的设备，并将该设备上的当前NPU流设置为传入流。

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

'stream'必须是NPU流（即由NPU设备创建的c10::Stream），否则行为未定义。
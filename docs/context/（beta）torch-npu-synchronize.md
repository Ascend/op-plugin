# （beta）torch::npu::synchronize

## 定义文件

torch_npu\csrc\libs\init_npu.h

## 函数原型

```
void torch::npu::synchronize(int64_t device_index = -1)
```

## 功能说明

NPU设备同步接口，与void torch::cuda::synchronize(int64_t device_index = -1)相同。

## 参数说明

device_index：int64_t类型，用来同步设备的index，默认-1，即同步当前设备。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


# （beta）torch::npu::synchronize

## 定义文件

torch_npu\csrc\libs\init_npu.h

## 函数原型

```cpp
void torch::npu::synchronize(int64_t device_index = -1)
```

## 功能说明

NPU设备同步接口，该接口会阻塞当前线程，直到所有已提交给NPU设备的计算任务执行完毕，与void torch::cuda::synchronize(int64_t device_index = -1)相同。

## 参数说明

device_index：int64_t类型，用来同步设备的index，默认-1，即同步当前设备。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>
<!-- end id4 -->

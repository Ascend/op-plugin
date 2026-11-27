# c10_npu::stream_synchronize

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->

## 功能说明

NPU设备流同步，与`c10::cuda::stream_synchronize`相同。

## 定义文件

torch_npu/csrc/core/npu/NPUFunctions.h

## 函数原型

```cpp
void stream_synchronize(aclrtStream stream)
```

## 参数说明

**stream** (`aclrtStream`)：必选参数，传入需要同步的有效NPU流句柄。例如，可通过`c10_npu::getCurrentNPUStream().stream()`获取当前设备的流句柄。

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
auto stream = c10_npu::getCurrentNPUStream();
c10_npu::stream_synchronize(stream.stream());
```

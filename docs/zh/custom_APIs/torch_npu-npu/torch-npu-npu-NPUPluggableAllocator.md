# torch_npu.npu.NPUPluggableAllocator

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

从so文件加载的NPU内存分配器。

## 定义文件

torch_npu/npu/memory.py

## 函数原型

```python
torch_npu.npu.memory.NPUPluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)
```

## 参数说明

- **path_to_so_file**（`str`）：so文件路径。
- **alloc_fn_name**（`str`）：内存申请函数名（与C/C++文件中函数名一致）。
- **free_fn_name**（`str`）：内存释放函数名（与C/C++文件中函数名一致）。

## 约束说明

`alloc_fn_name`内存申请函数名必须与C/C++文件中函数名一致。

`free_fn_name`内存释放函数名必须与C/C++文件中函数名一致。

## 调用示例

完整调用示例可参考[LINK](https://gitcode.com/Ascend/pytorch/blob/master/test/allocator/test_pluggable_allocator_extensions.py)。

**Python代码示例**：

```python
>>> import torch
>>> import torch_npu
# Load the allocator
>>> new_alloc = torch_npu.npu.memory.NPUPluggableAllocator('pluggable_allocator_extensions.so', 'my_malloc', 'my_free')
# Swap the current allocator
>>> torch_npu.npu.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
>>> npu_tensor = torch.zeros(10, device='npu')
```

风险提示：自定义的so建议严格参照安全代码示例，内存申请、释放函数必须正确实现。错误的so文件可能出现内存申请失败、内存泄漏等问题。

建议处理方式：用户可在内存申请、释放等内存相关操作函数中，增加日志记录内存行为，方便后续定位问题。

**C++安全代码示例**：

```cpp
#include <sys/types.h>
#include <iostream>
#include <torch/extension.h>
 
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
 
extern "C" {
 
void* my_malloc(ssize_t size, int device, aclrtStream stream)
{
    void *ptr;
    aclrtMallocAlign32(&ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    std::cout<<"alloc ptr = "<<ptr<<", size = "<<size<<std::endl;
    return ptr;
}
 
void my_free(void* ptr, ssize_t size, int device, aclrtStream stream)
{
    std::cout<<"free ptr = "<<ptr<<std::endl;
    aclrtFree(ptr);
}
}
 
```

**日志记录示例**：可使用TorchNPU自带的debug级别日志打印。日志需注意存放至安全路径。

```cpp
#include "torch_npu/csrc/core/npu/npu_log.h"
 
ASCEND_LOGD("Pluggable Allocator malloc: malloc = %zu", size);
 
ASCEND_LOGD("Pluggable Allocator free: free= %zu", size);
```

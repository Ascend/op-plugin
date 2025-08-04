# torch_npu.npu.change_current_allocator

## 定义文件

torch_npu/npu/memory.py

## 函数原型

```
torch_npu.npu.change_current_allocator(allocator) -> None
```

## 功能说明

更改使用的内存分配器。

## 参数说明

“allocator”(torch_npu.npu.memory._NPUAllocator) - 要设置为使用的内存分配器。

## 异常说明

如果内存分配器已被初始化，函数调用失败。

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term> 
- <term> Atlas 推理系列产品</term>

## 调用示例

完整调用示例可参考[LINK](https://gitee.com/ascend/pytorch/blob/v2.1.0-7.1.0/test/allocator/test_pluggable_allocator_extensions.py)。

**Python代码示例**：

```python
import torch
import torch_npu
# Load the allocator
new_alloc = torch_npu.npu.memory.NPUPluggableAllocator('pluggable_allocator_extensions.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch.npu.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
npu_tensor = torch.zeros(10, device='npu')
```

风险提示：自定义的so建议严格参照安全代码示例，内存申请，释放函数必须正确实现。错误的so文件可能出现内存申请失败，内存泄漏等问题。

建议处理方式：用户可在内存申请，释放等内存相关操作函数中，增加日志记录内存行为，方便后续定位问题。

**c++安全代码示例**：

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
    useflag = true;
    return ptr;
}
 
void my_free(void* ptr, ssize_t size, int device, aclrtStream stream)
{
    std::cout<<"free ptr = "<<ptr<<std::endl;
    aclrtFree(ptr);
}
}
 
```

**日志记录示例**：可使用Ascend Extension for PyTorch自带的debug级别日志打印。日志需注意存放至安全路径。

```cpp
#include "torch_npu/csrc/core/npu/npu_log.h"
 
ASCEND_LOGD("Pluggable Allocator malloc: malloc = %zu", size);
 
ASCEND_LOGD("Pluggable Allocator free: free= %zu", size);
```


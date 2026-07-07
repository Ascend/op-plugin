# torch_npu.npu.change_current_allocator

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Changes the current memory allocator.

## Definition File

torch_npu/npu/memory.py

## Prototype

```python
torch_npu.npu.change_current_allocator(allocator) -> None
```

## Parameters

 **`allocator`** (`torch_npu.npu.memory._NPUAllocator`): Memory allocator instance to be used.

## Constraints

This function fails if the memory allocator has already been initialized.

## Example

For details about the complete call example, see [LINK](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/test/allocator/test_pluggable_allocator_extensions.py).

**Python Code Sample**

```python
>>> import torch
>>> import torch_npu
# Load the allocator
>>> new_alloc = torch_npu.npu.memory.NPUPluggableAllocator('pluggable_allocator_extensions.so', 'my_malloc', 'my_free')
# Swap the current allocator
>>> torch.npu.change_current_allocator(new_alloc)
#This will allocate memory in the device using the new allocator
>>> npu_tensor = torch.zeros(10, device='npu')
```

Custom .so implementations must strictly adhere to secure coding requirements. Memory allocation and deallocation functions must be correctly implemented. Invalid library files may cause issues such as memory allocation failures and memory leaks.

You are advised to add logging in custom memory allocation and deallocation functions to track memory behavior and facilitate subsequent fault locating.

**C++ Security Code Sample**

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

**Logging**: You can use the built-in `DEBUG`-level logging feature provided by Ascend Extension for PyTorch. Ensure that all log files are stored in a secure path.

```cpp
#include "torch_npu/csrc/core/npu/npu_log.h"
 
ASCEND_LOGD("Pluggable Allocator malloc: malloc = %zu", size);
 
ASCEND_LOGD("Pluggable Allocator free: free= %zu", size);
```

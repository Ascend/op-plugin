# torch_npu.npu.NPUPluggableAllocator

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Creates an NPU memory allocator from a custom so file.

## Definition File

torch_npu/npu/memory.py

## Prototype

```python
torch_npu.npu.NPUPluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)
```

## Parameters

- **`path_to_so_file`** (`str`): Path to the `.so` library file.
- **`alloc_fn_name`** (`str`): Memory allocation function name. This name must match the function name in the C/C++ implementation.
- **`free_fn_name`** (`str`): Memory deallocation function name. This name must match the function name in the C/C++ implementation.

## Constraints

`alloc_fn_name` must match the corresponding allocation function defined in the C/C++ source code.

`free_fn_name` must match the corresponding deallocation function defined in the C/C++ source code.

## Example

For details about the complete call example, see [LINK](https://gitcode.com/ascend/pytorch/blob/v2.7.1-26.0.0/test/allocator/test_pluggable_allocator_extensions.py).

**Python Code Sample**

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

# torch_npu.erase_stream

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
| <term>Atlas A2 training products</term> | √   |
| <term>Atlas training products</term>                                      |    √     |

## Function

- Removes the marker indicating that a `Tensor` has been used by a stream, which was added to the memory pool by using `record_stream`.

- Memory can be reused across multiple streams. By default, the memory pool is marked by using `record_stream` to prevent the reused memory from being returned to the memory pool prematurely, which causes memory corruption. During each memory allocation, the memory pool determines whether the operator has finished execution and can be safely released by querying events on the device. However, this coordination mechanism between the host and the device introduces a side effect: when the host issues commands much faster than the device executes them, it may drive up the peak memory because the device has not finished execution when the host performs the query.

- This API provides an `erase_stream` capability to return memory to the memory pool ahead of time by actively erasing and freeing the memory after an event wait. Since subsequent operators must be executed after the event wait, this memory returned to the memory pool in advance will not be corrupted by subsequent operators.

## Prototype

```python
torch_npu.erase_stream(tensor, stream) -> None
```

## Parameters

- **`tensor`** (`Tensor`): Required. Tensor from which the marker needs to be removed.
- **`stream`** (`torch_npu.npu.Stream`): Required. Stream to which the removed marker belongs.

## Return Values

`None`

No value is returned.

## Constraints

This API must be used in combination with an event wait to ensure that the marker is removed only after the operator execution is complete, thereby avoiding memory corruption.

## Example

```python
>>> import torch
>>> import torch_npu
>>> stream1 = torch_npu.npu.Stream()
>>> stream2 = torch_npu.npu.Stream()
>>> with torch_npu.npu.stream(stream2):
...     matrix1 = torch.ones(1000, 1000, device='npu')
...     matrix2 = torch.ones(1000, 1000, device='npu')
...     tensor1 = torch.matmul(matrix1, matrix2)
...     data_ptr1 = tensor1.data_ptr()
...     print(data_ptr1)
...     tensor1.record_stream(stream1)
...     torch_npu.erase_stream(tensor1, stream1)
...     del tensor1
...     tensor2 = torch.ones(1000, 1000, device='npu')
...     print(tensor2.data_ptr())
...
20616943637504
20616943637504
```

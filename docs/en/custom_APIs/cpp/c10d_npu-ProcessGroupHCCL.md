# c10d_npu::ProcessGroupHCCL

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term>  | √   |     

## Function

Inherits from `c10d::Backend` and implements the `HCCL` backend APIs for communication operators.

## Definition File

torch_npu\csrc\distributed\ProcessGroupHCCL.hpp

## Prototype

```cpp
class c10d_npu::ProcessGroupHCCL
```

## Constraints

- To achieve optimal concurrency and performance, all HCCL communication functions provided by this class are asynchronous. You must call `WorkHCCL::wait()` or `WorkHCCL::synchronize()` to ensure that the operations are complete.
- This class inherits from the native `c10d::Backend`. For details about the basic communication operator APIs, see the [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html). They are not repeated in this document. The supported basic communication operators are as follows:<br>
broadcast<br>
allreduce<br>
allreduce_coalesced<br>
reduce<br>
allgather<br>
allgather_togather<br>
allgather_into_tensor_coalesced<br>
reduce_scatter<br>
reduce_scatter_tensor_coalesced<br>
barrier<br>
gather<br>
scatter<br>
send<br>
recv<br>
recv_anysource<br>
alltoall_base<br>
alltoall<br>

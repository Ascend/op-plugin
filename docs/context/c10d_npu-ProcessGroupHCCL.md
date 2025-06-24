# c10d_npu::ProcessGroupHCCL

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term> Atlas A3 训练系列产品 </term>            |    √     |
|<term> Atlas A2 训练系列产品</term>   | √   |     

## 功能说明

ProcessGroupHCCL继承自`c10d::Backend`，实现`HCCL`后端的相关接口，用于通信算子调用。

## 定义文件

torch_npu\csrc\distributed\ProcessGroupHCCL.hpp

## 函数原型

```
class c10d_npu::ProcessGroupHCCL
```

## 约束说明

- 为了更好的并发和性能，该类提供的所有HCCL通信都是异步函数，用户需要确保通过WorkHCCL::wait()或WorkHCCL::synchronize()来保证任务完成。
- 该类继承自原生`c10d::Backend`，实现的基本通信算子相关接口资料可参考[原生文档](https://docs.pytorch.org/docs/stable/distributed.html)，不在本文档额外补充。基本通信算子列表如下：
broadcast
allreduce
allreduce_coalesced
reduce
allgather
allgather_togather
allgather_into_tensor_coalesced
reduce_scatter
reduce_scatter_tensor_coalesced
barrier
gather
scatter
send
recv
recvAnysource
alltoall_base
alltoall



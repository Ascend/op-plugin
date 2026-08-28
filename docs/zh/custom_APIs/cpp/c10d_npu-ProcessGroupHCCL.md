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

```cpp
class c10d_npu::ProcessGroupHCCL : public c10d::Backend {
public:
    struct Options : c10d::Backend::Options {
        explicit Options(bool is_high_priority_stream = false);
        static c10::intrusive_ptr<Options> create(
            bool _is_high_priority_stream = false,
            std::chrono::milliseconds timeout = kNoTimeout);
        std::chrono::milliseconds opTimeout;
        bool is_high_priority_stream;
    };

    ProcessGroupHCCL(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        c10::intrusive_ptr<Options> options = Options::create());

    ~ProcessGroupHCCL() override;

    // 基本通信算子
    c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;
    c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;
    c10::intrusive_ptr<c10d::Work> allreduce_coalesced(std::vector<at::Tensor>& tensors,
        const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override;
    c10::intrusive_ptr<c10d::Work> reduce(std::vector<at::Tensor>& tensors,
        const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;
    c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
    c10::intrusive_ptr<c10d::Work> reduce_scatter(std::vector<at::Tensor>& outputTensors,
        std::vector<std::vector<at::Tensor>>& inputTensors,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;
    c10::intrusive_ptr<c10d::Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;
    c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override;
    c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override;
    c10::intrusive_ptr<c10d::Work> alltoall_base(at::Tensor& outputTensor, at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes, std::vector<int64_t>& inputSplitSizes,
        const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;
};
```

## 约束说明

- 为了更好的并发和性能，该类提供的所有HCCL通信操作均为异步函数，用户需要通过调用WorkHCCL::wait()或WorkHCCL::synchronize()来确保任务执行完成。
- 该类继承自原生`c10d::Backend`，实现的基本通信算子相关接口资料可参考[原生文档](https://docs.pytorch.org/docs/stable/distributed.html)，本文档不作额外补充。基本通信算子列表如下：<br>
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

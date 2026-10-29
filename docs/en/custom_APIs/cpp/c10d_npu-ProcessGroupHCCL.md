# c10d_npu::ProcessGroupHCCL

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

ProcessGroupHCCL inherits from `c10d::Backend` and implements the related interfaces of the `HCCL` backend, which are used to invoke communication operators.

## Definition File

torch_npu\csrc\distributed\ProcessGroupHCCL.hpp

## Prototype

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

    // Basic communication operators
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

## Constraints

- To achieve better concurrency and performance, all the HCCL communication operations provided by this class are asynchronous functions. You need to call WorkHCCL::wait() or WorkHCCL::synchronize() to ensure that the task is completed.
- This class inherits from the native `c10d::Backend`. For details about the related interfaces of the basic communication operators it implements, see the [native documentation](https://docs.pytorch.org/docs/stable/distributed.html); no additional description is provided in this document. The list of basic communication operators is as follows:<br>
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

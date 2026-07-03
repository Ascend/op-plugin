# c10d_npu::ProcessGroupHCCL::batch_isend_irecv

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>            |    √     |
|<term>Atlas A2 training products</term>  | √   |     

## Function

Sends or receives a batch of tensors. Asynchronously processes each point-to-point (P2P) operation in the sequence. Returns the corresponding request handle.

## Definition File

torch_npu\csrc\distributed\ProcessGroupHCCL.hpp

## Prototype

```cpp
c10::intrusive_ptr<c10d::Work> batch_isend_irecv(std::vector<std::string>& op_type, std::vector<at::Tensor>& tensors, std::vector<uint32_t> remote_rank_list)
```

## Parameters

- **`op_type`** (`std::vector<std::string>&`): Required. Operation sequence. Valid values are `"isend"` or `"irecv"`.
- **`tensors`** (`std::vector<at::Tensor>&`): Required. Tensors to send or receive. The number of tensors must match the number of operations in `op_type`.
- **`remote_rank_list`** (`std::vector<uint32_t>`): Required. Global rank IDs of the peer processes. The number of rank IDs must match the number of operations in `op_type`.

## Return Values

`c10::intrusive_ptr<c10d::Work>`

Asynchronous `isend` and `irecv` operations. Because these operations execute asynchronously, a work handle is returned. Before reading any tensor data, call `work.wait()` to ensure that `batch_isend_irecv` is complete.

## Constraints

None

# (beta) torch_npu.distributed.all_to_all_vc

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products/Atlas 800I A2 inference server</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id3 -->

## Function

Variable-length all-to-all communication implemented based on the HCCL `HcclAlltoAllVC` API. This API is driven by a global `[rankSize][rankSize]` count matrix `send_count_matrix`, where `send_count_matrix[i][j]` indicates the number of elements sent from rank i to rank j. All ranks pass the same complete matrix. The sent and received data is arranged contiguously in the buffer by destination/source rank.

## Prototype

```python
torch_npu.distributed.all_to_all_vc(output, input, send_count_matrix, group=None, async_op=False) -> torch.distributed.distributed_c10d.Work
```

## Parameters

- **`output`** (`Tensor`): Required. Output Tensor used to receive data. The number of elements must be equal to the sum of the receive column of the current rank (the sum of elements in column i of `send_count_matrix`, where i is the current rank).
- **`input`** (`Tensor`): Required. Input Tensor used to send data. The number of elements must be equal to the sum of the send row of the current rank (the row corresponding to the current rank in `send_count_matrix`). The sent data is arranged contiguously in `input` by destination rank.
- **`send_count_matrix`** (`List[List[int]]`): Required. Global `[group size][group size]` square matrix. `send_count_matrix[i][j]` indicates the number of elements sent from rank i to rank j. All ranks must pass the same complete matrix.
    - If the number of rows is not equal to the group size: `ValueError: send_count_matrix must have {world_size} rows (group size), got {actual row count}`.
    - If it is not a square matrix (the length of a row is not equal to the group size): `ValueError: send_count_matrix must be a square [{world_size}][{world_size}] matrix`.
    - If a negative element exists: `RuntimeError: sendCountMatrix[i][j] is negative, counts must be non-negative`.
    - If the sum of the send row of the current rank is inconsistent with the number of elements in `input`: `RuntimeError: input numel (...) must equal send row sum (...)`.
    - If the sum of the receive column of the current rank is inconsistent with the number of elements in `output`: `RuntimeError: output numel (...) must equal recv col sum (...)`.
- **`group`** (`torch.distributed.distributed_c10d.ProcessGroup`): Optional. Distributed process group. The default value is None, which indicates the default process group (the global process group containing all processes).
- **`async_op`** (`bool`): Optional. Whether to call asynchronously. The default value is False. If set to True, the API returns immediately and communication is performed in the background. The caller can wait for communication to complete by calling `wait()` on the returned work handle. If set to False, the API blocks until communication is complete.

## Return Values

When `async_op=True`, a work handle for the computation is returned, and the actual computation result is written to `output`. When `async_op=False`, the API blocks until communication is complete and returns None.

## Constraints

- `input` and `output` must be contiguous Tensors on the NPU and must have the same dtype.

- In-place operations are not supported: `input` and `output` cannot point to the same memory address (HCCL requires that the addresses configured for recvBuf and sendBuf be different).

- `send_count_matrix` must be a `[group size][group size]` square matrix with non-negative elements. The sum of the send row of the current rank must be equal to the number of elements in `input`, and the sum of the receive column must be equal to the number of elements in `output`.

## Examples

Create and save the following file `test.py`.

```python
import os
import torch
import torch_npu
import torch.distributed as dist
dist.init_process_group(backend="hccl")
rank = int(os.getenv('LOCAL_RANK'))
torch.npu.set_device(rank)
send_count_matrix = [[1, 3], [2, 1]]
send_vals = []
for j in range(2):
    send_vals += [rank * 10 + j] * send_count_matrix[rank][j]
recv_total = sum(send_count_matrix[i][rank] for i in range(2))
input_tensor = torch.tensor(send_vals, dtype=torch.float32).npu()
output_tensor = torch.zeros(recv_total, dtype=torch.float32).npu()
torch_npu.distributed.all_to_all_vc(
    output_tensor,
    input_tensor,
    send_count_matrix,
    async_op=False
)
```

Run the following command.

```bash
torchrun --nproc-per-node=2 test.py
```

# (beta) torch_npu.distributed.all_gather_into_tensor_uneven

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas 800I A2 inference products</term>           |    √     |
|<term>Atlas inference products</term>                                      |    √     |

## Function

Extends the native [torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor) API by supporting zero-copy and uneven tensor splitting in `torch_npu.distributed.all_gather_into_tensor_uneven`.

## Prototype

```python
torch_npu.distributed.all_gather_into_tensor_uneven(output, input, output_split_sizes =None, group=None, async_op=False) -> torch.distributed.distributed_c10d.Work
```

## Parameters

- **`output`** (`Tensor`): Output tensor used to receive the computation result.
- **`input`** (`Tensor`): Input tensor used to provide computation data. No special constraints are imposed on the shape of `input`.
- **`output_split_sizes`** (`List[int]`): Split sizes along dimension 0 of the `output` tensor. The default value is `None`. The number of elements must match the size of the `group`, and the sum must equal the size of dimension 0 of `output`.
    - If the sum of `output_split_sizes` does not match the size of dimension 0 of `output`, a `RuntimeError` is raised: "Split sizes doesn't match total dim 0 size."
    - If the number of elements in `output_split_sizes` does not match the size of `group`, a `RuntimeError` is raised: "Number of tensor splits not equal to group size."
- **`group`** (`torch.distributed.distributed_c10d.ProcessGroup`): The process group for distributed communication. The default value is `None`.
- **`async_op`** (`bool`): Specifies whether to execute the operation asynchronously. The default value is `False`.

## Return Values

The `output` tensor shape is the concatenation of the `input` tensor shapes across all ranks. 

## Constraints

- This API can be used only in single-server scenarios.

- The sum of `output_split_sizes` must equal the size of dimension 0 of `output`. The number of elements in `output_split_sizes` must match the size of the `group`.

## Example

Create the `test.py` file and save it.

```python
import os
import torch
import torch_npu
import torch.distributed as dist
 
dist.init_process_group(backend="hccl")
rank = int(os.getenv('LOCAL_RANK'))
torch.npu.set_device(rank)
 
output_split_sizes = [2, 3]
input_tensors = [torch.tensor([1, 2], dtype=torch.int32).npu(), torch.tensor([4, 5, 6], dtype=torch.int32).npu()]
output_tensor = torch.zeros(sum(output_split_sizes), dtype=torch.int32).npu()
 
torch_npu.distributed.all_gather_into_tensor_uneven(
    output_tensor,
    input_tensors[rank],
    output_split_sizes=output_split_sizes,
    async_op=False
)
```

Run the following command:

```bash
torchrun --nproc-per-node=2 test.py
```

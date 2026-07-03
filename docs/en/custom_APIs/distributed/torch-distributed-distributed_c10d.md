# torch.distributed.distributed_c10d._world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Obtains the name of the collective communication domain from the initialized domain.

## Prototype

```python
torch.distributed.distributed_c10d._world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rankid->int,init_comm=True) -> string
```

Note: This method belongs to the PyTorch `ProcessGroup` class using the NPU backend. The `ProcessGroup` can be `default_pg` or a non-default group created by `torch.distributed.distributed_c10d.new_group`.

> [!NOTICE]  
> When calling this API, ensure that the current device is correctly configured.

## Parameters

- **`rankid`** (`int`): Required. Rank ID of the device in the collective communication domain. The input is a global rank ID, which uniquely identifies a device across multiple servers.

- **`init_comm`** (`bool`): Optional. Specifies whether to force initialization if `HCCL` is uninitialized. The default value is `True`. When set to `True`, calling `get_hccl_comm_name` initializes `HCCL` if it has not yet been initialized and returns the group name. When set to `False`, the API does not initialize `HCCL`. Instead, it returns an empty string if `HCCL` is not already initialized.

>**Note**:<br>
>`HCCL` initialization allocates memory resources, increasing memory usage. By default, 200 MB is allocated for the `send` buffer and 200 MB for the `receive` buffer, totaling 400 MB. The buffer size is controlled by the `HCCL_BUFFSIZE` environment variable.

## Return Values

`string`

A `string` value representing the name of the collective communication domain.

## Constraints

- Ensure `init_process_group` has been called before using this API, and that the backend is set to `HCCL`.
- The calling method differs between PyTorch version `2.1.0` (and later) and earlier versions. For details, see [Example](#section14459801435).

## Example<a name="section14459801435"></a>

```python
import torch
import torch_npu
import torch.multiprocessing as mp
import os
from torch.distributed.distributed_c10d import _get_default_group
import torch.distributed as dist
def example(rank, world_size):
    torch.npu.set_device("npu:" + str(rank))
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    default_pg = _get_default_group()
    if torch.__version__ > '2.0':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    print(hcomm_info)
def main():
    world_size = 2
    mp.spawn(example,
            args=(world_size, ),
            nprocs=world_size,
            join=True)
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    main()

group_name_0
group_name_0
```

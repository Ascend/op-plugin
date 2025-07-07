# torch.distributed.distributed_c10d._world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name

## 功能说明

从初始化完成的集合通信域中获取集合通信域名字。

## 函数原型

```
torch.distributed.distributed_c10d._world.default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rankid->int,init_comm=True) -> String
```

注：接口为PyTorch的ProcessGroup类，backend为NPU backend的方法。ProcessGroup可以是default_pg，也可以是torch.distributed.distributed_c10d.new_group创建的非default_pg。

>**须知：**<br>
>调用该接口时，需要保证当前current device被设置为正确。

## 参数说明

rankid：集合通信对应device的rankid。传入的rankid为全局的rankid，多机间device具有唯一的rankid。

init_comm：可选入参，默认值为True。值为True时，表示调用get_hccl_comm_name时，若hccl还未完成初始化时，则完成初始化，并返回group name。值为False时，表示调用get_hccl_comm_name时，若hccl还未完成初始化，申请内存资源等操作时，则不进行初始化，并返回空字符串。

>**说明：** <br>
>hccl初始化会申请内存资源，造成内存升高，默认申请内存大小为Send buffer与Recv buffer各200M，共400M。buffer大小受环境变量HCCL_BUFFSIZE控制。

## 输出说明

string类型的集合通信域的名字。

## 约束说明

1. 使用该接口前确保init_process_group已被调用，且初始化的backend为hccl。
2. PyTorch2.1及以后版本与PyTorch2.1之前的版本对该接口调用方式不同，见[调用示例](#section14459801435)。

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term> 
- <term> Atlas 推理系列产品</term> 

## 调用示例<a name="section14459801435"></a>

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
```


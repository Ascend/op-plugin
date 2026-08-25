# （beta）torch_npu.distributed.all_to_all_vc

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term>            |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Ascend 950PR/Ascend 950DT</term>                              |    √     |

## 功能说明

基于HCCL `HcclAlltoAllVC`接口实现的变长all-to-all通信。该接口由一个全局`[rankSize][rankSize]`计数矩阵`send_count_matrix`驱动，`send_count_matrix[i][j]`表示rank i向rank j发送的元素个数，所有rank传入同一份完整矩阵，收发数据按目标/来源rank在缓冲区内连续排布。

## 函数原型

```python
torch_npu.distributed.all_to_all_vc(output, input, send_count_matrix, group=None, async_op=False) -> torch.distributed.distributed_c10d.Work
```

## 参数说明

- **output** (`Tensor`)：必选参数，输出Tensor，用于接收数据，元素个数须等于本rank接收列（`send_count_matrix`第i列中i为本rank的元素之和）之和。
- **input** (`Tensor`)：必选参数，输入Tensor，用于发送数据，元素个数须等于本rank发送行（`send_count_matrix`中本rank对应行）之和；发送数据按目标rank在`input`内连续排布。
- **send_count_matrix** (`List[List[int]]`)：必选参数，全局`[group size][group size]`方阵，`send_count_matrix[i][j]`表示rank i向rank j发送的元素个数，所有rank须传入同一份完整矩阵。
    - 行数不等于group size时报错：`ValueError: send_count_matrix must have {world_size} rows (group size), got {实际行数}`。
    - 不是方阵（某行长度不等于group size）时报错：`ValueError: send_count_matrix must be a square [{world_size}][{world_size}] matrix`。
    - 存在负数元素时报错：`RuntimeError: sendCountMatrix[i][j]为负数，counts must be non-negative`。
    - 本rank发送行之和与`input`元素个数不一致时报错：`RuntimeError: input numel (...) must equal send row sum (...)`。
    - 本rank接收列之和与`output`元素个数不一致时报错：`RuntimeError: output numel (...) must equal recv col sum (...)`。
- **group** (`torch.distributed.distributed_c10d.ProcessGroup`)：可选参数，分布式进程组，默认值None，表示使用默认进程组（即包含所有进程的全局进程组）。
- **async_op** (`bool`)：可选参数，是否异步调用，默认值False。取值为True表示异步调用，接口立即返回，通信在后台进行，调用方可通过返回的工作句柄调用wait()等待通信完成；取值为False表示同步调用，接口阻塞直至通信完成后返回。

## 返回值说明

`async_op=True`时返回进行计算的工作句柄，实际计算结果写入output；`async_op=False`时阻塞至通信完成后返回None。

## 约束说明

- `input`与`output`须为NPU上的连续Tensor，且dtype一致。

- 不支持原地操作：`input`与`output`不能指向同一内存地址（HCCL要求recvBuf与sendBuf配置的地址不能相同）。

- `send_count_matrix`为`[group size][group size]`方阵且元素非负；本rank发送行之和等于`input`元素个数，接收列之和等于`output`元素个数。

## 调用示例

创建以下文件test.py并保存。

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

执行如下命令。

```bash
torchrun --nproc-per-node=2 test.py
```

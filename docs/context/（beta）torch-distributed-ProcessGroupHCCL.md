# （beta）torch.distributed.ProcessGroupHCCL
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

创建一个ProcessGroupHCCL对象并返回。

## 函数原型
```
torch.distributed.ProcessGroupHCCL(store, rank, size, timeout) -> ProcessGroup
```

## 参数说明

- **store**：`torch.distributed.distributed_c10d.PrefixStore`对象，可以通过构造函数构造。
- **rank**：当前节点的rank序号。
- **size**：全部通讯节点的数量。
- **timeout**：通讯中断时间，判断节点断连，默认值为1800s。

## 返回值说明
`ProcessGroup`
# （beta）torch.distributed.ProcessGroupHCCL

## 函数原型
```
torch.distributed.ProcessGroupHCCL(store, rank, size, timeout); -> ProcessGroup
```

## 功能说明

创建一个ProcessGroupHCCL对象并返回。

## 参数说明

-   store：torch.distributed.distributed\_c10d.PrefixStore对象，可以通过构造函数构造。
-   rank：当前节点的rank序号。
-   size：全部通讯节点的数量。
-   timeout：通讯中断时间，判断节点断连，默认值为1800s。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>


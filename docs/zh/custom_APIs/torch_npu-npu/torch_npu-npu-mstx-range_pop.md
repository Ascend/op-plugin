# torch_npu.npu.mstx.range_pop

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

标识打点结束。

与[torch_npu.npu.mstx.range_push](./torch_npu-npu-mstx-range_push.md)成对使用。

## 函数原型

```python
torch_npu.npu.mstx.range_pop(domain: str='default') -> int
```

## 参数说明

**domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内，标识时间段事件的结束。需要与torch_npu.npu.mstx.range_push接口的domain配置一致。

## 返回值说明

返回线程内配对的torch_npu.npu.mstx.range_push接口记录range打点的层级；无配对的torch_npu.npu.mstx.range_push接口时，接口执行失败，返回-1。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
torch_npu.npu.mstx.range_push("dataloader")
dataloader()
torch_npu.npu.mstx.range_pop()
```

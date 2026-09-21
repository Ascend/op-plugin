# torch_npu.npu.mstx.range_pop

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

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

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
torch_npu.npu.mstx.range_push("dataloader")
dataloader()
torch_npu.npu.mstx.range_pop()
```

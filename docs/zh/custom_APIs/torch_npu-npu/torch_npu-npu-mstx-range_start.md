# torch_npu.npu.mstx.range_start

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

标识打点开始。

与[torch_npu.npu.mstx.range_end](./torch_npu-npu-mstx-range_end.md)成对使用。

支持跨线程使用，支持多次嵌套调用，torch_npu.npu.mstx.range_end自动匹配最近的torch_npu.npu.mstx.range_start。

## 函数原型

```python
torch_npu.npu.mstx.range_start(message: str, stream=None, domain: str='default') -> int
```

## 参数说明

- **message** (`str`)：必选参数，打点携带信息的字符串。

  传入的message字符串长度要求：msPTI场景不能超过255字节。

- **stream** (`torch_npu.npu.Stream`)：可选参数，用于执行打点任务的stream，默认为None。

  - 配置为None或不配置时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
  
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内标记瞬时事件。默认为'default'，表示默认domain，不设置也为默认domain。

## 返回值说明

range_id：用于标识该range；如果接口执行失败，返回0。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
id = torch_npu.npu.mstx.range_start("dataloader", None)
dataloader()
torch_npu.npu.mstx.range_end(id)
```

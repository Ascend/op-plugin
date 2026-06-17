# torch_npu.npu.mstx.range_push

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

标识打点开始。

与[torch_npu.npu.mstx.range_pop](./torch_npu-npu-mstx-range_pop.md)成对使用。

仅支持在单线程内使用，支持多次嵌套调用，torch_npu.npu.mstx.range_pop自动匹配最近的torch_npu.npu.mstx.range_push。

## 函数原型

```python
torch_npu.npu.mstx.range_push(message: str, stream=None, domain: str='default') -> int
```

## 参数说明

- **message** (`str`)：必选参数，打点携带信息的字符串。传入的message字符串长度要求：
  - msPTI场景：不能超过255字节。
  - msProf或torch_npu.profiler采集场景：不能超过156字节。
- **stream** (`torch_npu.npu.Stream`)：可选参数，用于执行打点任务的stream，默认为None。
  - 配置为None或不配置时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内标记瞬时事件。默认为'default'，表示默认domain，不设置也为默认domain。

## 返回值说明

返回线程内该接口记录range打点的层级，从0开始计数；如果接口执行失败，返回-1。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

- 无嵌套调用

  ```python
  torch_npu.npu.mstx.range_push("dataloader")
  dataloader()
  torch_npu.npu.mstx.range_pop()
  ```

- 嵌套调用

  ```python
  torch_npu.npu.mstx.range_push("dataloader1", cur_stream)
  dataloader()    # 事件1
  torch_npu.npu.mstx.range_push("dataloader2", cur_stream)
  dataloader()    # 事件2
  torch_npu.npu.mstx.range_pop()
  dataloader()    # 事件3
  torch_npu.npu.mstx.range_pop()
  ```

  如上示例中，事件2的耗时由中间一对push和pop接口采集，事件1、2、3则由最外层push和pop接口包含。

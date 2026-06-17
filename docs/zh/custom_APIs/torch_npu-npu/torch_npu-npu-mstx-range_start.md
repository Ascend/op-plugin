# torch_npu.npu.mstx.range_start

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

与[torch_npu.npu.mstx.range_end](./torch_npu-npu-mstx-range_end.md)成对使用。

支持跨线程使用，支持多次嵌套调用，torch_npu.npu.mstx.range_end自动匹配最近的torch_npu.npu.mstx.range_start。

## 函数原型

```python
torch_npu.npu.mstx.range_start(message: str='None', stream=None, domain: str='default') -> int
```

## 参数说明

- **message** (`str`)：可选参数，打点携带信息的字符串，默认为'None'。传入的message字符串长度要求：
  - msPTI场景：不能超过255字节。
  - msProf或torch_npu.profiler采集场景：不能超过156字节。
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

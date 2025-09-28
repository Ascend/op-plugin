# torch_npu.npu.mstx.mark

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

标记瞬时事件。

## 函数原型

```
torch_npu.npu.mstx.mark(message: str, stream=None, domain: str='default') -> none:
```

## 参数说明


- **message** (`str`)：可选参数，打点携带信息字符串指针。传入的message字符串长度要求：
  - MSPTI场景：不能超过255字节。
  - 非MSPTI场景：不能超过156字节。
- **stream** (`int`)：可选参数，用于执行打点任务的stream。
  - 配置为None时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内标记瞬时事件。默认为default，表示默认domain，不设置也为默认domain。

## 返回值说明

无
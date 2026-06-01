# torch_npu.npu.mstx.annotate

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

API级的打点，可自定义选择目标代码段或目标函数进行耗时采集。

## 函数原型

```python
torch_npu.npu.mstx.annotate(message: str = '', stream=None, domain: str = 'default')
```

## 参数说明

- **message** (`str`)：打点携带信息的字符串。

  使用with语句调用本接口时，该参数必选；使用装饰器调用本接口时，该参数可选，默认使用函数名作为message。

  传入的message字符串长度要求：

  - msPTI场景：不能超过255字节。
  - msProf或torch_npu.profiler采集场景：不能超过156字节。

- **stream** (`torch_npu.npu.Stream`)：可选参数，用于执行打点任务的stream，默认为None。
  
  - 配置为None或不配置时，只标记Host侧的采集耗时。
  - 配置为有效的stream时，标记Host侧和对应Device侧的采集耗时。
  
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内标记瞬时事件。默认为'default'，表示默认domain，不设置也为默认domain。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

- with语句调用

  ```python
  with torch_npu.npu.mstx.annotate('my_code_range', cur_stream):
      my_code()
  ```

  如上示例中，with语句会对其作用域内的接口执行打点操作，采集该接口耗时。

- 装饰器调用

  ```python
  @torch_npu.npu.mstx.annotate()
  def my_code():
      print("my_code start")
      my_code()
      print("my_code end")
  ```
  
  如上示例中，默认使用装饰的函数名作为该打点任务的message。

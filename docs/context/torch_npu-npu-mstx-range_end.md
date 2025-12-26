# torch_npu.npu.mstx.range_end

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

## 函数原型

```
torch_npu.npu.mstx.range_end(range_id: int, domain: str='default') -> int:
```

## 参数说明

- **range_id** (`int`)：必选参数，传入由[torch_npu.npu.mstx.range_start](./torch_npu-npu-mstx-range_start.md)接口返回的ID。
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内，标识时间段事件的结束。需要与[torch_npu.npu.mstx.range_start](./torch_npu-npu-mstx-range_start.md)接口的domain配置一致。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
id = torch_npu.npu.mstx.range_start("dataloader", None)    # 第二个入参设置None或者不设置，只记录Host侧range耗时
dataloader()
torch_npu.npu.mstx.range_end(id)
```
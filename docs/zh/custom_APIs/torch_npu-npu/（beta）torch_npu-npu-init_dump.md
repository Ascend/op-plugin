# （beta）torch_npu.npu.init_dump

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950DT</term>            |    √     |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

初始化dump配置，是dump流程的起始接口。

正确的调用顺序为：`init_dump()` → `set_dump(cfg_file)` → 执行模型 → `finalize_dump()`。若未先调用本接口，`set_dump`与`finalize_dump`将因dump未初始化而报错。

## 函数原型

```python
torch_npu.npu.init_dump()
```

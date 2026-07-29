# （beta）torch_npu.npu.init_dump

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

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

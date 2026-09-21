# （beta）torch_npu.npu.init_dump

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

初始化dump配置，是dump流程的起始接口。

正确的调用顺序为：`init_dump()` → `set_dump(cfg_file)` → 执行模型 → `finalize_dump()`。若未先调用本接口，`set_dump`与`finalize_dump`将因dump未初始化而报错。

## 函数原型

```python
torch_npu.npu.init_dump()
```

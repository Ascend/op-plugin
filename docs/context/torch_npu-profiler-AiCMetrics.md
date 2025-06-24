# torch_npu.profiler.AiCMetrics

## 函数原型

```
torch_npu.profiler.AiCMetrics
```

## 功能说明

AI Core的性能指标采集项，Enum类型。用于作为_ExperimentalConfig类的aic_metrics参数。

## 参数说明

- AiCoreNone：关闭AI Core的性能指标采集。

- PipeUtilization：计算单元和搬运单元耗时占比，包括采集项vec_ratio、mac_ratio、scalar_ratio、mte1_ratio、mte2_ratio、mte3_ratio、icache_miss_rate、fixpipe_ratio。
- ArithmeticUtilization：各种计算类指标占比统计，包括采集项mac_fp16_ratio、mac_int8_ratio、vec_fp32_ratio、vec_fp16_ratio、vec_int32_ratio、vec_misc_ratio。
- Memory：外部内存读写类指令占比，包括采集项ub_read_bw、ub_write_bw、l1_read_bw、l1_write_bw、l2_read_bw、l2_write_bw、main_mem_read_bw、main_mem_write_bw。
- MemoryL0：内部内存读写类指令占比，包括采集项scalar_ld_ratio、scalar_st_ratio、l0a_read_bw、l0a_write_bw、l0b_read_bw、l0b_write_bw、l0c_read_bw、l0c_write_bw、l0c_read_bw_cube、l0c_write_bw_cube。
- ResourceConflictRatio：流水线队列类指令占比，包括采集项vec_bankgroup_cflt_ratio、vec_bank_cflt_ratio、vec_resc_cflt_ratio、mte1_iq_full_ratio、mte2_iq_full_ratio、mte3_iq_full_ratio、cube_iq_full_ratio、vec_iq_full_ratio、iq_full_ratio。
- MemoryUB：内部内存读写指令占比，包括采集项vec_bankgroup_cflt_ratio、vec_bank_cflt_ratio、vec_resc_cflt_ratio、mte1_iq_full_ratio、mte2_iq_full_ratio、mte3_iq_full_ratio、cube_iq_full_ratio、vec_iq_full_ratio、iq_full_ratio。
- L2Cache：读写cache命中次数和缺失后重新分配次数，包括采集项ai\*_write_cache_hit、ai\*_write_cache_miss_allocate、ai\*_r\*_read_cache_hit、ai\*_r\*_read_cache_miss_allocate。

默认值为AiCoreNone。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       aic_metrics = torch_npu.profiler.AiCMetrics.AiCoreNone
       )
with torch_npu.profiler.profile(
        on_trace_ready = torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config = experimental_config) as prof:
        for step in range(steps): # 训练函数
                 train_one_step() # 训练函数
                 prof.step()
```


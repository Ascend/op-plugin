# torch_npu.npu.mstx.mark

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

标记瞬时事件。

## 函数原型

```python
torch_npu.npu.mstx.mark(message: str, stream=None, domain: str='default') -> None
```

## 参数说明

- **message** (`str`)：必选参数，打点携带信息的字符串。

  传入的message字符串长度要求：msPTI场景不能超过255字节。

- **stream** (`torch_npu.npu.Stream`)：可选参数，用于执行打点任务的stream，默认为None。
  
  - 配置为None时，只标记Host侧的瞬时事件。
  - 配置为有效的stream时，标记Host侧和对应Device侧的瞬时事件。
  
- **domain** (`str`)：可选参数，指定的domain名称，表示在指定的domain内标记瞬时事件。默认为'default'，表示默认domain，不设置也为默认domain。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu
import time
experimental_config = torch_npu.profiler._ExperimentalConfig(
    data_simplification=False,
    # 开启mstx，并设置mstx_domain_include或mstx_domain_exclude的domain过滤属性
    mstx=True,
    mstx_domain_include=['default','domain1']    # 配置采集'default'和'domain1'打点数据
    # mstx_domain_exclude=['domain2']    # 配置不采集'domain2'的打点数据，与mstx_domain_include不同时配置
)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    experimental_config=experimental_config) as prof:
    for i in range(5):
        # 标记默认domain的mstx打点
        torch_npu.npu.mstx.mark("mark_with_default_domain")
        range_id = torch_npu.npu.mstx.range_start("range_with_default_domain")
        time.sleep(1)    # 模拟用户代码
        torch_npu.npu.mstx.range_end(range_id)
        ...    # 用户代码
        # 标记用户自定义domain1的mstx打点
        torch_npu.npu.mstx.mark("mark_with_domain1", domain = "domain1")
        range_id1 = torch_npu.npu.mstx.range_start("range_with_domain1", domain="domain1")
        time.sleep(1)    # 模拟用户代码
        torch_npu.npu.mstx.range_end(range_id1, domain="domain1")
        ...    # 用户代码
        # 标记用户自定义domain2的mstx打点
        torch_npu.npu.mstx.mark("mark_with_domain2", domain = "domain2")
        range_id2 = torch_npu.npu.mstx.range_start("range_with_domain2", domain="domain2")
        time.sleep(1)    # 模拟用户代码
        torch_npu.npu.mstx.range_end(range_id2, domain="domain2")
        prof.step()

```

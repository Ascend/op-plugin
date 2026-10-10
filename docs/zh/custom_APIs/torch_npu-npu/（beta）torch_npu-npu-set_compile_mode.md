# （beta）torch_npu.npu.set_compile_mode

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

设置是否开启JIT编译。

## 函数原型

```python
torch_npu.npu.set_compile_mode(jit_compile = bool)
```

## 参数说明

**jit_compile**（`bool`）：设置为True时表示开启JIT编译，设置为False时表示关闭JIT编译。

> [!NOTE]  
>
<!-- npu="910,310p" id5 -->
>- Atlas训练系列产品/Atlas推理系列产品默认为jit_compile=True，即开启JIT编译。
<!-- end id5 -->
<!-- npu="A3,910b" id6 -->
>- Atlas A2训练系列产品/Atlas A3训练系列产品默认为jit_compile=False，即关闭JIT编译。
<!-- end id6 -->

## 调用示例

```python
>>> torch_npu.npu.set_compile_mode(jit_compile=False)
```

# （beta）torch_npu.npu.set_compile_mode

## 函数原型

```
torch_npu.npu.set_compile_mode(jit_compile = bool)
```

## 功能说明

设置是否开启二进制。

## 参数说明

jit_compile：jit_compile=True时是非二进制模式，jit_compile=False时是二进制模式。

>**说明：**<br>
>- Atlas 训练系列产品/Atlas 推理系列产品默认为jit_compile=True，即非二进制模式。
>- Atlas A2 训练系列产品/Atlas A3 训练系列产品默认为jit_compile=False，即二进制模式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>>torch_npu.npu.set_compile_mode(jit_compile=False)
```


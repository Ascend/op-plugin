# （beta）torch_npu.npu.is_jit_compile_false

## 函数原型

```
torch_npu.npu.is_jit_compile_false()
```

## 功能说明

确认算子计算是否采用二进制，如果是二进制计算，返回True，否则返回False。

## 输出说明

bool型。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


## 调用示例

```python
>>>torch_npu.npu.set_compile_mode(jit_compile=False)
>>>torch_npu.npu.is_jit_compile_false()
True
```


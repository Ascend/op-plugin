# （beta）torch_npu.npu_sign_bits_pack

## 函数原型

```
torch_npu.npu_sign_bits_pack(Tensor self, int size) -> Tensor
```

## 功能说明

将float类型1位Adam打包为uint8。

## 参数说明

- self(Tensor) - 1D float张量。
- size(Int) - reshape时输出张量的第一个维度。

## 约束说明

Size可被float打包的输出整除。如果self的size可被8整除，则输出的size为(size of self)/8；否则，输出的size为(size of self // 8) + 1。将在小端位置添加-1浮点值以填充可整除性。<term>Atlas 训练系列产品</term>支持float32和float16类型输入。<term>Atlas 推理系列产品</term>支持float32和float16类型输入。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
>>> b = torch_npu.npu_sign_bits_pack(a, 2)
>>> b
tensor([[159],
        [ 15]], device='npu:0', dtype=torch.uint8)
```


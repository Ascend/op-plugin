# （beta）torch_npu.npu_format_cast

## 函数原型

```
torch_npu.npu_format_cast(self, acl_format) -> Tensor
```

## 功能说明

修改NPU张量的格式。

## 参数说明

- self (Tensor) - 输入张量。
- acl_format (Int) - 目标格式。数据格式具体可参考《CANN AscendCL应用软件开发指南 (Python)》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0914.html">aclFormat</a>”章节。此处需输入数字，例如2，即代表ACL_FORMAT_ND格式。数据排布格式具体可参考《CANN Ascend C算子开发指南》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html">数据排布格式”</a>章节。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(2, 3, 4, 5).npu()
>>> torch_npu.get_npu_format(x)
0
>>> x1 = torch_npu.npu_format_cast(x, 2)
>>> torch_npu.get_npu_format(x1)
2
```


## 简介
op_plugin/config 目录主要包含API的ATEN IR定义和前反向绑定配置。

## 结构化适配介绍
结构化适配指通过在`op_plugin_functions.yaml`中进行配置，可自动生成算子实现Kernel。仅支持op_api对应的算子。

如何判断是否可结构化：opapi对应的Aclnn算子与Aten IR的语义对齐，适配层除申请output tensor，无其他适配逻辑。

### Yaml配置说明
每个结构化适配的函数必须在`op_plugin_functions.yaml`中配置, 具有如下格式：
```
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  op_api: v2.1
  gen_opapi:
    out:
      size: arg0
      dtype: arg1.scalar_type()
    exec: aclnnFuncName
```
各个字段的含义如下：

- `gen_opapi`: 表示对应API可结构化，其他字段需要配置在此字段下

- `out`: 表示函数的输出，此字段下面包含size和dtype字段，如果包含多个输出，可配置成out0、out1等。对于out类接口，此字段不可自定义，需要与Aten IR定义的输出参数名相同。对于inplace类接口，不需要配置此字段。

- `size`: 配置输出tensor的shape大小，如果大小和schema中的某个参数相同，可以配置成输入参数的名字。也可配置成自定义infershape函数，infershape函数需在`KernelNpuOutputSize.h`中实现。对于out类接口，如果输出shape不变，可省略此字段。配置方式主要包含以下几种：
```
Aten IR定义：
- func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
方式一: 和输入参数相同
  size: arg0
方式二：枚举每个维度的值
  size: '{4, arg0.size(0), arg0.size(1), arg1.size(0)}'
方式三：条件表达式
  size: 'arg1 == 1? arg0.sizes(): at::ArrayRef<int64_t>()'
方式四：在KernelNpuOutputSize.h中自定义infershape函数, 例如broadcast_ops_npu_output_size
  size: broadcast_ops_npu_output_size(arg0, arg1)
```

- `dtype`: 配置输出tensor的dtype大小，如果大小和schema中的某个参数相同，可以配置成输入参数的名字。也可配置成自定义inferdtype函数，inferdtype函数需在`KernelNpuOutputDtype.h`中实现。对于out类接口，如果输出dtype不需要check，可省略此字段。配置方式主要包含以下几种：
```
Aten IR定义：
- func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
方式一: 和输入参数相同
  dtype: arg0
方式二：配置成已知的dtype类型
  dtype: at::kFloat
方式三：条件表达式
  dtype: 'isIntegralType(arg0.scalar_type(), true) ? at::kFloat : arg0.scalar_type()'
方式四：在KernelNpuOutputDtype.h中自定义inferdtype函数。
  dtype: inferdtype(arg0, arg1)
```

- `name`: 输出结果涉及named tensor逻辑，可配置此字段，当前仅支持name和输入参数相同的配置。不涉及可忽略。

- `exec`: 配置`EXEC_NPU_CMD`对应的参数，如果除aclnnname，其它参数顺序和Aten IR的顺序相同，可只配置aclnnname。以`abs`为例，`exec`字段可以配置成下面两种方式
```
    - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
      方式一：
      exec: aclnnAbs, self, out
      方式二：
      exec: aclnnAbs
```

- `structured_inherit`：如果原函数或inplace类接口的字段配置与out类接口的字段配置相同，可通过此字段继承对应的out类接口。
以`abs`为例，原函数和out类函数的out属性和`exec`相同，可通过`structured_inherit`字段继承。
```
  - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    op_api: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5
    gen_opapi:
      out:
        size: self
        dtype: self
        name: self
      exec: aclnnAbs, self, out
  - func: abs(Tensor self) -> Tensor
    op_api: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5
    gen_opapi:
      structured_inherit: abs.out  
```

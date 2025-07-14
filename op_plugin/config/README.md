# API适配开发流程

## 适配文件结构

```
├── op_plugin
│   ├── config  # 算子配置文件目录
│   │   ├── derivatives.yaml # 算子前反向绑定配置文件
│   │   └── op_plugin_functions.yaml # 算子对外接口配置文件
│   ├── ops # 算子适配文件目录
│   │   ├── aclops # aclop算子适配目录
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi # aclnn算子适配目录
│   │       ├── sparse # sparse相关算子适配目录
│   │       │   └── SparseTensorUtils.h
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
│   └── ...
└──...
```


## NPU适配算子开发

### 算子Yaml配置

算子的aten ir定义位于op_plugin/config/op_plugin_functions.yaml文件中，所有版本的定义都在这个文件里面，通过配置不同版本来区分。
```yaml
# op_plugin_functions.yaml
all_version: [v1.11, v2.0, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8]
# 官方算子
official:
  - func: abs(Tensor self) -> Tensor
    acl_op: all_version
    op_api: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8
    gen_opapi:
      structured_inherit: abs.out
# 自定义算子
custom:
  - func: my_abs(Tensor self) -> Tensor
    acl_op: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8
    op_api: all_version
symint:
  - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    acl_op: [v2.1, newest]
```

其中：

- `all_version`表示当前pytorch支持的所有版本
- `official`和`custom`分别表示该字段下的算子为原生和自定义算子；`symint`字段表明该算子支持symint类型的入参，该种算子后面详细介绍。
- `func`定义了算子的schema，主要有名称、入参和返回参数，具体规则可参考原生定义。
- `acl_op`字段后面填版本名称，表示在该版本支持acl_op调用。如果支持的版本与`all_version`表示的版本一致，则可以用"all_version"表示；也可以用一个左闭右闭的区间表示，如`acl_op: [v2.1, newest]`或者`acl_op: [v2.1, v2.4]`，`newest`表示最新版本，具体可查看`all_version`。可选字段。
- `op_api`字段后面填版本名称，表示在该版本支持op_api调用。使用方式参考`acl_op`字段。可选字段。
- `gen_opapi`对于支持op_api调用的算子，如果适配代码简单，可以直接调用底层算子，不需要额外的适配，则可以考虑用结构化适配的方式自动生成适配代码，详见章节[结构化适配介绍](#结构化适配介绍)

如果存在某个Aten IR有两个版本不一致，则需要两个都加上，如std.correction在1.11和2.1及以上的入参名称不同，则需要分开写成两个，通过`version`区分。

```yaml
  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
    acl_op: v1.11
    op_api: v1.11

  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    acl_op: [v2.1, newest]
    op_api: [v2.1, newest]
```



### 适配代码实现

当前支持适配`ACLOP`算子和`ACLNN`算子两类算子，`ACLOP`算子适配文件位于`op_plugin/ops/aclops`，`ACLNN`算子适配文件位于`op_plugin/ops/opapi`目录。一个算子所有版本的适配代码都在一个文件中，通过编译宏`VERSION_BETWEEN`来区分不同版本。

#### ACLOP算子适配

如果所有版本的适配代码一致，则不需要额外添加编译宏，适配文件路径为：`op_plugin/ops/aclops/AbsKernelNpu.cpp`，文件命名规范为算子名称+`KernelNpu`，算子名称首字母大写。

```c++
// 算子适配实现文件路径 op_plugin/ops/aclops/AbsKernelNpu.cpp
// 1. 引入依赖头文件
// 对外接口头文件，包含op_plugin所有aclop算子对外的函数原型
#include "op_plugin/AclOpsInterface.h"
// torch调用ACLOP算子时，所依赖的基础函数对应的头文件
#include "op_plugin/utils/OpAdapter.h"

// 2. 算子接口适配实现
// opplugin内适配的算子对外接口都定义在op_plugin命名空间中，外部调用方式为op_plugin::abs、op_plugin::abs_out；内部不同类型的算子适配采用不同的命名空间
// CANN算子定义在acl_op命名空间中，
namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
// 不对外暴露的接口，都定义在匿名空间中。常见为xx_nocheck等，直调ACLOP算子，不做内存、shape校验的函数。
namespace{
at::Tensor& abs_out_nocheck(at::Tensor& result, const at::Tensor& self) {
    at_npu::native::OpCommand cmd;
    cmd.Name("Abs")
       .Input(self)
       .Output(result)
       .Run();
    return result;
}
} // namespace

// abs_out api实现函数，参数与torch api一致。
at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result) {
    // CheckOut作用：校验result的size、dtype等是否符合预期。若dtype不符合预期，则抛错。若size不符合则进行resize操作
    npu_preparation::CheckOut({self}, result, self);
    // check_match作用：校验result是否为连续。因ACLOP算子无法支持非连续输出，result非连续时，需要单独处理。
    if (!npu_utils::check_match(&result)) {
      // 若result非连续，创建连续tensor(contig_tensor)，接收ACLOP算子(abs)的输出。再将contig_tensor拷贝到原始输出result。
      at::Tensor contiguous_result = npu_utils::format_contiguous(result);
      abs_out_nocheck(contigTensor, self);
      npu_utils::format_fresh_view(result, contiguous_result);
    } else {
     // 若result连续，直接调用ACLOP算子。
      abs_out_nocheck(result, self);
  }
    return result;
}

// abs api实现函数，参数与torch api一致。
at::Tensor abs(const at::Tensor& self) {
    // 构造输出tensor，调用ACLOP算子。
    auto output_size = op_infer::infershape_for_elewise(self);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    abs_out_nocheck(result, self);
    return result;
}

// abs_ api实现函数，参数与torch api一致。该接口为inplace操作，即输出结果存放在输入tensor中。
at::Tensor& abs_(at::Tensor& self) {
    // 调用out接口，避免因self作为输出时，非连续场景下，直调ACLOP算子结果出错。
    acl_op::abs_out(self, self);
    return self;
}
} // namespace acl_op
```

不同版本间适配代码有差异的，所有代码均放在同一个文件中，用编译宏来区分

```c++
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
// 1.11的函数入参和2.0及以上版本有区别，因此用宏来控制
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
    return embedding_common_nocheck(weight, indices);
}
#endif
// 2.0及以上版本的代码都一致
#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor embedding_symint(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
    return embedding_common_nocheck(weight, indices);
}
#endif

} // namespace acl_op
```

#### ACLNN算子适配

ACLNN算子适配与ACLOP类似，也是如果所有版本的适配代码一致，则不需要额外添加编译宏，适配文件路径为：`op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp`，文件命名规范为算子名称+`KernelNpuOpApi`，算子名称首字母大写。

```c++
//算子适配实现路径/op_plugin/ops/base_ops/opapi/AbsKernelNpuOpApi.cpp
// 1. 引入依赖头文件
// 对外接口头文件，包含op_plugin所有ACLNN算子对外的函数原型
#include "op_plugin/OpApiInterface.h"
// 引用 ACLOP 算子声明头文件
#include "op_plugin/AclOpsInterface.h"
// torch调用ACLNN算子时，所依赖的基础函数对应的头文件
#include "op_plugin/utils/op_api_common.h"

// 2. 算子接口适配实现
// ACLNN算子定义在op_api命名空间中，
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// abs_out api实现函数，参数与torch api一致。
at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result) {
    // 查找ACLNN算子实现，查找失败则使用ACLOP算子实现
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, result));
    npu_preparation::check_tensor({self}, result, self);
    // 异步调用npu执行
    EXEC_NPU_CMD(aclnnAbs, self, result);
    return result;
}

// abs api实现函数，参数与torch api一致。
at::Tensor abs(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self));

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAbs, self, result);
    return result;
}

// abs_ api实现函数，参数与torch api一致。该接口为inplace操作，即输出结果存放在输入
at::Tensor& abs_(at::Tensor& self) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
    op_api::abs_out(self, self);
    return self;
}
}  // namespace op_api
```

不同版本间适配代码有差异的，所有代码均放在同一个文件中，用编译宏来区分

```c++
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// 1.11的函数入参和2.0及以上版本有区别，需要单独实现，因此用宏来控制
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq,
                     bool sparse)
{
  DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
  // calculate the output size
  auto output_size = op_infer::array_to_small_vector(indices.sizes());
  output_size.emplace_back(weight.size(weight.dim() - 1));
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
  // calculate the output resugt of the NPU
  EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
  return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor embedding_symint(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse)
{
    DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse));
    // calculate the output size
    auto output_size = op_infer::array_to_small_vector(indices.sizes());
    output_size.emplace_back(weight.size(weight.dim() - 1));
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
    // calculate the output resugt of the NPU
    EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
    return result;
}
#endif

} // namespace op_api
```

### 自动前反向绑定算子配置

Pytorch的算子自动反向微分依赖于算子的前反向绑定，即前向函数和反向函数的绑定。对于原生的算子，官方已有前反向绑定逻辑，插件侧有对应前向算子和反向算子适配即可(只需要在`op_plugin_functions.yaml`里面配置)。对于自定义算子，则需要在插件侧配置前反向自动绑定。

针对需要绑定前反向的算子（包括自定义算子和前反向绑定逻辑与原生不一致的原生算子）提供自动绑定前向算子和反向算子的功能。

- 适配前向和反向算子： 与上节算子适配开发中一致，分别适配前向算子和反向算子，并在op_plugin_functions.yaml中配置前向和反向算子。
- 配置前反向绑定，将前向和反向算子进行绑定：
  Op-plugin与原生PyTorch一致，通过derivatives.yaml，配置算子的前反向绑定关系，如下所示：

```yaml
# derivatives.yaml
- name: l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  self: l1_loss_backward(grad, self, target, reduction)
  target: l1_loss_backward(grad, self, target, reduction) * -1
  version: [v2.0, newest]
```

所有版本的算子前反向绑定都在同一个derivatives.yaml里面，通过`version`字段来区分版本。


## 结构化适配介绍
结构化适配指通过在`op_plugin_functions.yaml`中进行配置，可自动生成算子实现Kernel。仅支持op_api对应的算子。

如何判断是否可结构化：opapi对应的Aclnn算子与Aten IR的语义对齐，适配层除申请output tensor，无其他适配逻辑。

自动生成的适配文件位于`op_plugin/ops/opapi/StructKernelNpuOpApi.cpp`。

### Yaml配置说明
每个结构化适配的函数必须在`op_plugin_functions.yaml`中配置, 具有如下格式：
方式一（常规场景）：
```yaml
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  op_api: v2.1
  gen_opapi:
    out:
      size: arg0
      dtype: arg1.scalar_type()
      name: arg0
    new_params:
      arg3: arg0.value_or(0)
    exec: aclnnFuncName, arg0, arg1, out, arg3
```
各个字段的含义如下：

- `gen_opapi`: 表示对应API可结构化，其他字段需要配置在此字段下

- `out`: 表示函数的输出，此字段下面包含size和dtype字段，如果包含多个输出，可配置成out0、out1等。对于out类接口，此字段不可自定义，需要与Aten IR定义的输出参数名相同。对于inplace类接口，不需要配置此字段。

- `size`: 配置输出tensor的shape大小，如果大小和schema中的某个参数相同，可以配置成输入参数的名字。也可配置成自定义infershape函数，infershape函数需在`KernelNpuOutputSize.h`中实现。对于out类接口，如果输出shape不变，可省略此字段。配置方式主要包含以下几种：
```yaml
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
```yaml
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

- `new_params`: 可选字段，支持新增自定义变量，配置格式如下：
```yaml
    new_params:
      arg0: func(arg1)
```

- `exec`: 配置`EXEC_NPU_CMD`对应的参数。如果除aclnnname（原函数可排除out参数），其它参数顺序和Aten IR的顺序相同，可只配置aclnnname。以`abs`为例，`exec`字段可以配置成下面两种方式
```yaml
    - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
      方式一：
      exec: aclnnAbs, self, out
      方式二：
      exec: aclnnAbs
```
方式二（继承场景）:
```yaml
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  op_api: v2.1
  gen_opapi:
    structured_inherit: func_name.out
```

- `structured_inherit`：如果原函数或inplace类接口的字段配置与out类接口的字段配置相同，可通过此字段继承对应的out类接口。
以`abs`为例，原函数和out类函数的out属性和`exec`相同，可通过`structured_inherit`字段继承。
```yaml
  - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    op_api: [v1.11, newest]
    gen_opapi:
      out:
        size: self
        dtype: self
        name: self
      exec: aclnnAbs, self, out
  - func: abs(Tensor self) -> Tensor
    op_api: [v1.11, newest]
    gen_opapi:
      structured_inherit: abs.out
```

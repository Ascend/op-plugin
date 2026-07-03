# API Adaptation Development Workflow

## Adaptation File Structure

```text
├── op_plugin
│   ├── config                       # Operator configuration directory
│   │   ├── deprecated.yaml          # Configuration file for generating deprecation warnings
│   │   ├── derivatives.yaml         # Configuration file for operator forward and backward bindings
│   │   └── op_plugin_functions.yaml # Configuration file for public operator APIs
│   ├── ops                          # Operator adaptation directory
│   │   ├── aclops                   # Directory for ACLOP operator adaptations
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                    # Directory for ACLNN operator adaptations
│   │       ├── sparse               # Directory for sparse operator adaptations
│   │       │   └── SparseTensorUtils.h
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
│   └── ...
└──...
```

## NPU Operator Adaptation Development

### Operator YAML Configuration

The ATen IR definitions of operators are located in `op_plugin/config/op_plugin_functions.yaml`. Definitions for all supported versions are maintained in this file and distinguished by version-specific configuration.

```yaml
# op_plugin_functions.yaml
all_version: [v1.11, v2.0, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10, v2.11, v2.12, v2.13]
# Official operators
official:
  - func: abs(Tensor self) -> Tensor
    acl_op: all_version
    op_api: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10, v2.11, v2.12, v2.13
    gen_opapi:
      structured_inherit: abs.out
# Custom operators
custom:
  - func: my_abs(Tensor self) -> Tensor
    acl_op: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10, v2.11, v2.12, v2.13
    op_api: all_version
symint:
  - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    acl_op: [v2.1, newest]
```

The fields are described as follows:

- `all_version`: All versions currently supported by PyTorch.
- `official` and `custom`: Native and custom operators. The `symint` field indicates that the operator supports `SymInt` input parameters. This type of operator is described in detail later.
- `func`: Operator schema, including the operator name, input parameters, and return values. For the specific syntax, refer to the native PyTorch definitions.
- `acl_op`: Versions that support `acl_op` calls. If the supported versions are identical to those specified by `all_version`, use `all_version`. A closed interval can also be used, such as `acl_op: [v2.1, newest]` or `acl_op: [v2.1, v2.4]`, where `newest` represents the latest version defined by `all_version`. This field is optional.
- `op_api`: Versions that support `op_api` calls.  It is configured in the same way as `acl_op`. This field is optional.
- `gen_opapi`: For operators that support `op_api` calls, if the adaptation logic is simple and the underlying operator can be called directly without requiring additional adaptation logic, structured adaptation can be used to automatically generate the adaptation code. For details, see the section [Introduction to Structured Adaptation](#introduction-to-structured-adaptation).

If an ATen IR definition differs between versions, both definitions must be included. For example, because the input parameter names of `std.correction` differ between PyTorch 1.11 and PyTorch 2.1 or later, two separate definitions are required and distinguished by the `version` field.

```yaml
  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
    acl_op: v1.11
    op_api: v1.11

  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    acl_op: [v2.1, newest]
    op_api: [v2.1, newest]
```

### Adaptation Code Implementation

Currently, two types of operators are supported: `ACLOP` operators and `ACLNN` operators. The adaptation files for `ACLOP` operators are located in `op_plugin/ops/aclops`, while those for `ACLNN` operators are located in `op_plugin/ops/opapi`. The adaptation code for all versions of the same operator is maintained in a single file and differentiated using the `VERSION_BETWEEN` preprocessor macro.

#### ACLOP Operator Adaptation

If the adaptation code is identical across all versions, no additional compilation macros are required. The adaptation file path is `op_plugin/ops/aclops/AbsKernelNpu.cpp`. The file naming convention is *OperatorName* + `KernelNpu`, with the operator name capitalized.

```c++
// Operator adaptation implementation file: op_plugin/ops/aclops/AbsKernelNpu.cpp
// 1. Include dependency headers
// Public API header containing the function prototypes for all ACLOP operators in op_plugin
#include "op_plugin/AclOpsInterface.h"
// Header containing the utility functions required when PyTorch calls ACLOP operators
#include "op_plugin/utils/OpAdapter.h"

// 2. Implement operator API adaptation
// The Public APIs of adapted operators are defined in the op_plugin namespace and are called as op_plugin::abs and op_plugin::abs_out. Different categories of operator adaptations use different internal namespaces.
// CANN operators are defined in the acl_op namespace
namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
// Functions that are not exposed publicly are defined in anonymous namespaces. Typical examples include `xx_nocheck` functions, which call ACLOP operators directly without performing memory or shape validation.
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

// abs_out implementation. The parameters are identical to those of the torch API.
at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result) {
    // CheckOut validates whether the size, dtype, and other attributes of result meet expectations. If the dtype is invalid, an error is raised. If the size does not match, result is resized.
    npu_preparation::CheckOut({self}, result, self);
    // check_match verifies whether result is contiguous. Because ACLOP operators do not support non-contiguous outputs, non-contiguous outputs must be handled separately.
    if (!npu_utils::check_match(&result)) {
      // If result is non-contiguous, create a contiguous tensor (contig_tensor) to receive the output of the ACLOP operator, then copy contig_tensor back to result.
      at::Tensor contiguous_result = npu_utils::format_contiguous(result);
      abs_out_nocheck(contigTensor, self);
      npu_utils::format_fresh_view(result, contiguous_result);
    } else {
     // If result is contiguous, call the ACLOP operator directly.
      abs_out_nocheck(result, self);
  }
    return result;
}

// abs implementation. The parameters are identical to those of the torch API.
at::Tensor abs(const at::Tensor& self) {
    // Construct the output tensor and call the ACLOP operator.
    auto output_size = op_infer::infershape_for_elewise(self);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    abs_out_nocheck(result, self);
    return result;
}

// abs_ implementation function. Its parameters match those of the PyTorch API. This is an in-place operation, where the output is stored in the input tensor.
at::Tensor& abs_(at::Tensor& self) {
    // Call the out API to avoid incorrect results when self is used as the output in non-contiguous scenarios
    acl_op::abs_out(self, self);
    return self;
}
} // namespace acl_op
```

If the adaptation code differs across versions, all code is placed in the same file and differentiated using compilation macros.

```c++
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
// The function parameters differ between PyTorch 1.11 and PyTorch 2.0 or later, so compilation macros are used to distinguish them.
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
// The implementation is identical for PyTorch 2.0 and later.
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

#### ACLNN Operator Adaptation

ACLNN operator adaptation follows the same process as ACLOP operator adaptation. If the adaptation code is identical across all versions, no additional compilation macros are required. The adaptation file path is `op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp`. The file naming convention is *OperatorName* + `KernelNpuOpApi`, with the operator name capitalized.

```c++
// Operator adaptation implementation file: op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp
// 1. Include dependency headers
// Public API header containing the function prototypes for all ACLNN operators in op_plugin
#include "op_plugin/OpApiInterface.h"
// Header containing ACLOP operator declarations
#include "op_plugin/AclOpsInterface.h"
// Header containing the utility functions required when PyTorch calls ACLNN operators
#include "op_plugin/utils/op_api_common.h"

// 2. Implement operator API adaptation
// ACLNN operators are defined in the op_api namespace.
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// abs_out implementation. The parameters are identical to those of the torch API.
at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result) {
    // Search for the ACLNN implementation. If it is unavailable, fall back to the ACLOP implementation.
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, result));
    npu_preparation::check_tensor({self}, result, self);
    // Launch asynchronous execution on the NPU
    EXEC_NPU_CMD(aclnnAbs, self, result);
    return result;
}

// abs implementation. The parameters are identical to those of the torch API.
at::Tensor abs(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self));

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAbs, self, result);
    return result;
}

// abs_ implementation function. Its parameters match those of the PyTorch API. This is an in-place operation, where the output is stored in the input.
at::Tensor& abs_(at::Tensor& self) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
    op_api::abs_out(self, self);
    return self;
}
}  // namespace op_api
```

If the adaptation code differs across versions, all code is placed in the same file and differentiated using compilation macros.

```c++
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// The function parameters differ between PyTorch 1.11 and PyTorch 2.0 or later, so a separate implementation is required and distinguished using compilation macros.
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

### Automatic Forward and Backward Binding Configuration

The automatic differentiation mechanism of PyTorch depends on the forward and backward binding of operators (the mapping between a forward function and its corresponding backward function). For native operators, the official PyTorch framework already provides forward and backward binding logic; the plugin side only needs to adapt the corresponding forward and backward operators and configure them in `op_plugin_functions.yaml`. For custom operators, automatic forward and backward binding must be explicitly configured on the plugin side.

This feature automatically binds forward and backward operators for those requiring forward and backward bindings, including custom operators and native operators whose binding logic differs from the native implementation.

- **Adapt the forward and backward operators**. Adapt the forward and backward operators as described in the previous section and configure them in `op_plugin_functions.yaml`.
- Configure forward and backward bindings.
  Bind the forward and backward operators through `derivatives.yaml`, following the same approach used by native PyTorch.

```yaml
# derivatives.yaml
- name: l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  self: l1_loss_backward(grad, self, target, reduction)
  target: l1_loss_backward(grad, self, target, reduction) * -1
  version: [v2.0, newest]
```

The forward and backward bindings for all operator versions are maintained in a single `derivatives.yaml` file and distinguished by the `version` field.

## Introduction to Structured Adaptation

Structured adaptation automatically generates the operator implementation kernel based on the configuration in `op_plugin_functions.yaml`. It is supported only for `op_api` operators.

An operator is eligible for structured adaptation when the corresponding ACLNN operator has semantics consistent with the ATen IR and the adaptation layer performs no logic other than output tensor allocation.

The generated adaptation file is located at: `op_plugin/ops/opapi/StructKernelNpuOpApi.cpp`.

### YAML Configuration

Each structured adaptation function must be configured in `op_plugin_functions.yaml`.
Method 1 (common scenario):

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

The fields are described as follows:

- **`gen_opapi`**: Indicates that the API supports structured adaptation. All related fields are configured under this field.

- **`out`**: Function outputs. It contains `size` and `dtype` subfields. Multiple outputs can be configured as `out0`, `out1`, and so on. For `out` APIs, this field name must match the output parameter name defined in the ATen IR. It is not required for in-place APIs.

- **`size`**: Output tensor shape during shape deduction. If it is identical to one of the input parameters, specify the input parameter name directly. A custom shape deduction function implemented in `KernelNpuOutputSize.h` can also be specified. For `out` APIs, this field can be omitted if the output shape remains unchanged. The configuration methods are as follows:

```yaml
Aten IR definition:
- func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
Method 1: Same as the input parameter
  size: arg0
Method 2: Enumerate the value of each dimension
  size: '{4, arg0.size(0), arg0.size(1), arg1.size(0)}'
Method 3: Conditional expression
  size: 'arg1 == 1? arg0.sizes(): at::ArrayRef<int64_t>()'
Method 4: Custom shape deduction function implemented in KernelNpuOutputSize.h, such as broadcast_ops_npu_output_size
  size: broadcast_ops_npu_output_size(arg0, arg1)
```

- `dtype`: Output tensor data type. If it matches one of the input parameters, specify the input parameter name directly. A custom data type deduction function implemented in `KernelNpuOutputDtype.h` can also be specified. For `out` APIs, this field can be omitted if data type deduction validation is unnecessary. The configuration methods are as follows:

```yaml
Aten IR definition:
- func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
Method 1: Same as the input parameter
  dtype: arg0
Method 2: Specify a predefined dtype
  dtype: at::kFloat
Method 3: Conditional expression
  dtype: 'isIntegralType(arg0.scalar_type(), true) ? at::kFloat : arg0.scalar_type()'
Method 4: Custom `inferdtype` function implemented in KernelNpuOutputDtype.h
  dtype: inferdtype(arg0, arg1)
```

- `name`: Used when the output involves named tensor logic. Currently, only configurations identical to an input parameter are supported. Ignore this field if named tensors are not involved.

- `new_params`: Optional field for defining custom variables. The configuration format is as follows:

```yaml
    new_params:
      arg0: func(arg1)
```

- `exec`: Parameters passed to `EXEC_NPU_CMD`. If all arguments other than `aclnnname` follow the same order as in the ATen IR (excluding the `out` parameter of the original function), only `aclnnname` needs to be configured. For example, the `exec` field of `abs` can be configured in either of the following ways.

```yaml
    - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
      Method 1:
      exec: aclnnAbs, self, out
      Method 2:
      exec: aclnnAbs
```

Method 2 (inheritance scenario):

```yaml
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  op_api: v2.1
  gen_opapi:
    structured_inherit: func_name.out
```

- **`structured_inherit`**: If the configuration of an original API or an in-place API is identical to that of the corresponding `out` API, the configuration can be inherited through this field.
For example, for `abs`, the original API and the in-place API have the same `out` configuration and `exec` definition as the `out` API, so they can inherit these settings through `structured_inherit`.

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

## Adding Deprecation Warnings for Custom NPU APIs

### YAML Configuration

When a custom ATen API needs to be deprecated, deprecation warnings can be generated automatically by configuring `op_plugin/config/deprecated.yaml`.

```yaml
deprecated:
  # Deprecated without replacement
  - name: npu_nms_rotated
  # Deprecated with replacement
  - name: npu_broadcast
    replace: 'torch.broadcast_to'
  - name: npu_broadcast.out
    replace: 'torch.broadcast_to'
```

The fields are described as follows:

- **`name`**: ATen API to be deprecated. This name must include the overload name. For example, `npu_broadcast.out` represents the `out` overload of `npu_broadcast`.
- `replace`: Recommended replacement API, such as `torch.broadcast_to` replaces `npu_broadcast`.

If `replace` is specified, the generated warning follows this format:

```python
f'TORCH_WARN_ONCE("{name} is deprecated and will be removed in future version. Use {replace} instead.");'
```

`name` does not include the overload name. For example, for `npu_broadcast.out`, the generated warning is:

```python
'TORCH_WARN_ONCE("npu_broadcast is deprecated and will be removed in future version. Use torch.broadcast_to instead.");'
```

If `replace` is not specified, the generated warning follows this format:

```python
f'TORCH_WARN_ONCE("{name} is deprecated and will be removed in future version.");'
```

Likewise, the overload name is omitted from `name`. For example, for `npu_nms_rotated`, the generated warning is:

```python
'TORCH_WARN_ONCE("npu_nms_rotated is deprecated and will be removed in future version.");'
```

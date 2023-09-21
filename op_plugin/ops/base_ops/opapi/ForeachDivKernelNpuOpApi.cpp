// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;

std::vector<at::Tensor> _foreach_div(at::TensorList tensors1, at::TensorList tensors2) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2);
  if (!at::native::can_use_fast_route(tensors1, tensors2, true)) {
    return at::native::foreach_tensor_div_list_kernel_slow(tensors1, tensors2);
  }
  // construct the output tensorlist of the NPU
  auto scalar_type = tensors1[0].scalar_type();
  std::vector<at::Tensor> result;
  for (const at::Tensor &tensor : tensors1) {
    auto output_size = op_infer::input_same_output_size(tensor);
    result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                  tensor.options().dtype(scalar_type)));
  }
  at::TensorList result_ = at::TensorList(result);

  EXEC_NPU_CMD(aclnnForeachDivList, tensors1, tensors2, result_);
  return result;
}

void _foreach_div_(at::TensorList tensors1, at::TensorList tensors2) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2);
  if (!at::native::can_use_fast_route(tensors1, tensors2, true)) {
    return at::native::foreach_tensor_div_list_kernel_slow_(tensors1, tensors2);
  }

  EXEC_NPU_CMD(aclnnForeachDivList, tensors1, tensors2, tensors1);
  return;
}

std::vector<at::Tensor> _foreach_div(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  // 暂时默认走slow路径，待后续ascendc算子aclnn框架支持scalarlist类型处理
  at::native::check_foreach_api_restrictions(tensors, scalars);
  return at::native::foreach_tensor_div_scalarlist_kernel_slow(tensors, scalars);
}

void _foreach_div_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  // 暂时默认走slow路径，待后续ascendc算子aclnn框架支持scalarlist类型处理
  at::native::check_foreach_api_restrictions(tensors, scalars);
  return at::native::foreach_tensor_div_scalarlist_kernel_slow_(tensors, scalars);
}

}  // namespace op_api

// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& isin_out(const at::Tensor& elements,
                     const at::Scalar& test_elements,
                     bool assume_unique,
                     bool invert,
                     at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIsInTensorScalar, acl_op::isin_out(elements, test_elements, assume_unique, invert,
      result));
  at_npu::native::OpPreparation::check_tensor({elements}, result, at::ScalarType::Bool, elements.sizes());
  EXEC_NPU_CMD(aclnnIsInTensorScalar, elements, test_elements, assume_unique, invert, result);
  return result;
}

at::Tensor isin(const at::Tensor& elements,
                const at::Scalar& test_elements,
                bool assume_unique,
                bool invert) {
  DO_COMPATIBILITY(aclnnIsInTensorScalar, acl_op::isin(elements, test_elements, assume_unique, invert));

  at::Tensor result =
      at_npu::native::OpPreparation::apply_tensor_without_format(elements.sizes(), elements.options().dtype(at::kBool));

  EXEC_NPU_CMD(aclnnIsInTensorScalar, elements, test_elements, assume_unique, invert, result);
  return result;
}

}

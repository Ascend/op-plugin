// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCat, acl_op::cat_out(tensors, dim, result));
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

at::Tensor cat(at::TensorList tensors, at::Dimname dim) {
  DO_COMPATIBILITY(aclnnCat, acl_op::cat(tensors, dim));
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

}  // namespace op_api

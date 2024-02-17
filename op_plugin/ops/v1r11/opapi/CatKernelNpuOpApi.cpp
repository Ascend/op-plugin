// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

static c10::SmallVector<at::Tensor, op_infer::N> cat_dest_tensor_list_opapi(at::TensorList tensors) {
  c10::SmallVector<at::Tensor, op_infer::N> dst_tensor_list;
  // pytorch supports empty tensors, which needs to be removed from the NPU.
  for (at::Tensor tensor : tensors) {
    if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
      continue;
    }
    dst_tensor_list.emplace_back(tensor);
  }
  return dst_tensor_list;
}

static c10::SmallVector<int64_t, op_infer::SIZE> cat_npu_output_size_opapi(
    c10::SmallVector<at::Tensor, op_infer::N>& tensors, int64_t dimension) {
  bool allSkipped = true;
  int64_t nDims = 0;
  at::Tensor* notSkippedTensor;
  int numInputs = tensors.size();
  auto should_skip = [](const at::Tensor* t) { return t->nbytes() == 0 && t->dim() == 1; };
  for (int i = 0; i < numInputs; i++) {
    if (should_skip((at::Tensor*)&tensors[i])) {
      continue;
    }
    // found a non-empty tensor
    allSkipped = false;
    notSkippedTensor = (at::Tensor*)&tensors[i];
    nDims = notSkippedTensor->dim();
    break;
  }

  if (allSkipped) {
    c10::SmallVector<int64_t, op_infer::SIZE> size = {0};
    return size;
  }

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    at::Tensor* tensor = (at::Tensor*)&tensors[i];
    if (should_skip(tensor)) {
      continue;
    }
    cat_dim_size += tensor->size(dimension);
  }

  // Compute the size of the result
  c10::SmallVector<int64_t, op_infer::SIZE> size;
  size.resize(nDims);
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor->size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }

  return size;
}

at::Tensor& _cat_out(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCat, acl_op::_cat_out(tensors, dim, result));
  EXEC_NPU_CMD(aclnnCat, tensors, dim, result);
  return result;
}

at::Tensor& cat_out(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCat, acl_op::cat_out(tensors, dim, result));
  c10::SmallVector<at::Tensor, op_infer::N> inputTensors = cat_dest_tensor_list_opapi(tensors);

  int64_t dim_post_expr = 0;
  if (inputTensors.size() > 0) {
    dim_post_expr = inputTensors[0].dim();
  } else {
    return result;
  }
  dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
  auto outputSize = cat_npu_output_size_opapi(inputTensors, dim);
  npu_preparation::check_tensor({tensors[0]}, result, tensors[0].scalar_type(), outputSize);
  return at::_cat_out(result, tensors, dim);
}

at::Tensor _cat(at::TensorList tensors, int64_t dim) {
  DO_COMPATIBILITY(aclnnCat, acl_op::_cat(tensors, dim));
  c10::SmallVector<at::Tensor, op_infer::N> inputTensors = cat_dest_tensor_list_opapi(tensors);
  at::ScalarType high_type = at::native::result_type(tensors);

  int64_t dim_post_expr = 0;
  if (inputTensors.size() > 0) {
    dim_post_expr = inputTensors[0].dim();
  } else {
    at::Tensor result = npu_preparation::apply_tensor_without_format(tensors[0]);
    return result;
  }
  dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);

  // calculate the output size
  auto outputSize = cat_npu_output_size_opapi(inputTensors, dim);
  at::Tensor result =
      npu_preparation::apply_tensor_without_format(outputSize, inputTensors[0].options().dtype(high_type));
  op_api::_cat_out(tensors, dim, result);
  return result;
}

at::Tensor cat(at::TensorList tensors, int64_t dim) {
  DO_COMPATIBILITY(aclnnCat, acl_op::cat(tensors, dim));
  return at::_cat(tensors, dim);
}
}  // namespace op_api

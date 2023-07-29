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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

at::Tensor& npu_reshape_out(
    const at::Tensor& src,
    at::IntArrayRef shape,
    bool can_refresh,
    at::Tensor& result) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    std::vector<int64_t> out_strides = at::detail::defaultStrides(shape);
    if (result.sizes() != shape || result.strides() != out_strides) {
      auto allow_flag =
          result.unsafeGetTensorImpl()->allow_tensor_metadata_change();
      result.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
      at_npu::native::StorageDescHelper::SetDesc(result, shape, out_strides);
      result.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(allow_flag);
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("Reshape")
        .InputWithoutContiguous(src)
        .Input(shape, at::kLong)
        .Output(result)
        .Run();
  } else if (can_refresh) {
    at_npu::native::StorageDescHelper::SetDesc(
        result,
        op_infer::array_to_small_vector(result.sizes()),
        op_infer::array_to_small_vector(result.strides()));
  } else {
    at_npu::native::copy_d2d_by_memcpy(
        result,
        src,
        at_npu::native::NPUNativeFunctions::get_storage_size(result));
  }
  return result;
}

at::Tensor npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh) {
  at::Tensor result = npu_preparation::apply_tensor(self, shape);
  op_plugin::npu_reshape_out(self, shape, can_refresh, result);

  return result;
}
} // namespace op_plugin

// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_attention_update(
    at::TensorList lse,
    at::TensorList local_out,
    int64_t update_type) {
  // Packed inputs contain one local_out [P, T, D] and one lse [P, T].
  // Their outputs are [T, D] and [T], just like the legacy shard-list inputs.
  const bool is_packed_input = local_out.size() == 1 && lse.size() == 1 && local_out[0].dim() == 3 && lse[0].dim() == 2;
  if (is_packed_input) {
    static const bool supports_packed_input = IsGteCANNVersion("9.3.0", "CANN");
    TORCH_CHECK(
        supports_packed_input,
        "Packed input for npu_attention_update requires CANN 9.3.0 or later.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(
        local_out[0].size(0) == lse[0].size(0) && local_out[0].size(1) == lse[0].size(1),
        "Packed local_out and lse must have matching P and T dimensions.",
        OPS_ERROR(ErrCode::PARAM));
  }
  auto output_size_0 = local_out[0].sizes();
  if (is_packed_input) {
    output_size_0 = output_size_0.slice(1);
  }
  auto output_dtype_0 = local_out[0].scalar_type();

  at::Tensor out =
      npu_preparation::apply_tensor_without_format(output_size_0, local_out[0].options().dtype(output_dtype_0));
  at::Tensor lse_out;
  if (update_type == 1) {
    auto output_size_1 = lse[0].sizes();
    if (is_packed_input) {
      output_size_1 = output_size_1.slice(1);
    }
    lse_out = npu_preparation::apply_tensor_without_format(output_size_1, lse[0].options().dtype(lse[0].scalar_type()));
  }

  EXEC_NPU_CMD(aclnnAttentionUpdate, lse, local_out, update_type, out, lse_out);

  return std::make_tuple(out, lse_out);
}
} // namespace op_api

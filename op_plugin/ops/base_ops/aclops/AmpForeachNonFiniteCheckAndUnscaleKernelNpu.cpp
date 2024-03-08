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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
namespace {
const int FLOAT_STATUS_OP_DIMS_SIZE = 8;
} // namespace

bool _amp_foreach_non_finite_check(at::TensorList scaled_grads) {
  TORCH_NPU_WARN_ONCE("Non finite check on NPU device!");

  auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kFloat);
  at::Tensor float_status = at::zeros({FLOAT_STATUS_OP_DIMS_SIZE}, options);
  auto ans = acl_op::npu_get_float_status(float_status);

  auto result = ans[0].item().to<bool>();
  if (result) {
    auto ans_clear = acl_op::npu_clear_float_status(float_status);
  }

  return result;
}

void _amp_foreach_non_finite_check_and_unscale_(
    at::TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
    TORCH_NPU_WARN_ONCE("Non finite check and unscale on NPU device!");
    TORCH_CHECK(torch_npu::utils::is_npu(inv_scale), "inv_scale must be NPU-Tensor" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor" + OPS_ERROR(ErrCode::TYPE));

    if (scaled_grads.empty()) {
        return;
    }

    bool is_finite = true;
    if (c10_npu::IsSupportInfNan()) {
        for (const auto& scaled_grad : scaled_grads) {
            auto res = acl_op::sum(scaled_grad, at::ScalarType::Float);
            float cpu_sum = res.item().toFloat();
            if (!std::isfinite(cpu_sum)) {
                is_finite = false;
                break;
            }
        }
    } else {
        is_finite = !acl_op::_amp_foreach_non_finite_check(scaled_grads);
    }

    if (is_finite) {
        auto expected_device = scaled_grads[0].device();
        auto expected_dtype = scaled_grads[0].dtype();
        for (const auto& t : scaled_grads) {
            TORCH_CHECK(torch_npu::utils::is_npu(t), "one of scaled_grads was not a NPU tensor." + OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(t.device() == expected_device, "scaled_grads must be on the same device." + OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(t.dtype() == expected_dtype, "scaled_grads must have the same dtype." + OPS_ERROR(ErrCode::TYPE));
            TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor." + OPS_ERROR(ErrCode::PARAM));

            t.mul_(inv_scale);
        }
    } else {
        found_inf.add_(1);
    }
}
} // namespace acl_op

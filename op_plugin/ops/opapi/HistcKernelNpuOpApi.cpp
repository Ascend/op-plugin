// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <ATen/ops/aminmax.h>
#include <cmath>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
std::pair<at::Scalar, at::Scalar> histc_legacy_min_max(const at::Tensor &self, const at::Scalar &min,
                                                       const at::Scalar &max)
{
    double leftmost_edge = min.to<double>();
    double rightmost_edge = max.to<double>();
    if (leftmost_edge == rightmost_edge && self.numel() > 0) {
        const auto minmax = at::aminmax(self);
        leftmost_edge = std::get<0>(minmax).item<double>();
        rightmost_edge = std::get<1>(minmax).item<double>();
    }
    if (leftmost_edge == rightmost_edge) {
        leftmost_edge -= 1.0;
        rightmost_edge += 1.0;
    }
    TORCH_CHECK(
        !(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) || std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
        "range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(leftmost_edge < rightmost_edge, "max must be larger than min", OPS_ERROR(ErrCode::VALUE));
    return std::make_pair(at::Scalar(leftmost_edge), at::Scalar(rightmost_edge));
}

std::pair<at::Scalar, at::Scalar> histc_check_and_get_min_max(const at::Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max)
{
    TORCH_CHECK(bins > 0, "bins must be > 0", OPS_ERROR(ErrCode::VALUE));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return histc_legacy_min_max(self, min, max);
    }
    double leftmost_edge = min.to<double>();
    double rightmost_edge = max.to<double>();
    if (leftmost_edge == rightmost_edge && self.numel() > 0) {
        if (min.isIntegral(false) && max.isIntegral(false)) {
            TORCH_CHECK(min.to<int64_t>() <= max.to<int64_t>(), "max must be larger than min",
                OPS_ERROR(ErrCode::VALUE));
        }
        const auto data_minmax = at::aminmax(self);
        double data_min = std::get<0>(data_minmax).item<double>();
        double data_max = std::get<1>(data_minmax).item<double>();
        TORCH_CHECK(
            !(std::isinf(data_min) || std::isinf(data_max) || std::isnan(data_min) || std::isnan(data_max)),
            "range of [", data_min, ", ", data_max, "] is not finite",
            OPS_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK(
            !(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) || std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
            "range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite",
            OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(leftmost_edge <= rightmost_edge, "max must be larger than min", OPS_ERROR(ErrCode::VALUE));
    }
    return std::make_pair(min, max);
}
} // namespace

at::Tensor histc(const at::Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max)
{
    auto min_max = histc_check_and_get_min_max(self, bins, min, max);
    if (self.numel() == 0) {
        return at::zeros({bins}, self.options());
    }
    auto min_arg = min_max.first;
    auto max_arg = min_max.second;
    DO_COMPATIBILITY(aclnnHistc, acl_op::histc(self, bins, min_arg, max_arg));
    auto output_size_0 = {bins};
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0, self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHistc, self, bins, min_arg, max_arg, out);
    return out;
}

at::Tensor &histc_out(const at::Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max, at::Tensor &out)
{
    auto min_max = histc_check_and_get_min_max(self, bins, min, max);
    if (self.numel() == 0) {
        return out.zero_();
    }
    auto min_arg = min_max.first;
    auto max_arg = min_max.second;
    DO_COMPATIBILITY(aclnnHistc, acl_op::histc_out(self, bins, min_arg, max_arg, out));
    auto output_size_0 = {bins};
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHistc, self, bins, min_arg, max_arg, out);
    return out;
}
} // namespace op_api

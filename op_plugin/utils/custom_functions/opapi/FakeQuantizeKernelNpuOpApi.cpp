// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include <cmath>
#include <c10/core/MemoryFormat.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

using namespace std;

namespace {
    constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;
    struct TensorQuantizationParams {
        double scale;
        std::int32_t zero_point;
        int precision;
    };
}

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

void calculate_moving_average(const at::Tensor& x, at::Tensor & running_min, at::Tensor & running_max,
                              float averaging_const, bool per_row_fake_quant, int ch_axis)
{
    at::Tensor x_min;
    at::Tensor x_max;
    if (per_row_fake_quant) {
        TORCH_CHECK(ch_axis == 0,
            "Per-channel FakeQuant in fused_moving_avg_obs_fake_quant is only supported on axis == 0",
            OPS_ERROR(ErrCode::PARAM));
        std::tie(x_min, x_max) = at::aminmax(x, 1);
        for (const auto i : c10::irange(std::min(x_min.numel(), running_min.numel()))) {
            float min_value = running_min[i].item().toFloat();
            float max_value = running_max[i].item().toFloat();
            if (std::isinf(min_value)) {
                running_min[i] = x_min[i].item().toFloat();
            } else {
                running_min[i] = running_min[i].item().toFloat() + averaging_const *
                    (x_min[i].item().toFloat() - running_min[i].item().toFloat());
            }
            if (std::isinf(max_value)) {
                running_max[i] = x_max[i].item().toFloat();
            } else {
                running_max[i] = running_max[i].item().toFloat() + averaging_const *
                    (x_max[i].item().toFloat() - running_max[i].item().toFloat());
            }
        }
    } else {
        std::tie(x_min, x_max) = at::aminmax(x);
        float min_value = running_min[0].item().toFloat();
        float max_value = running_max[0].item().toFloat();
        if (std::isinf(min_value)) {
            running_min[0] = x_min.item().toFloat();
        } else {
            running_min[0] = running_min[0].item().toFloat() + averaging_const *
                (x_min.item().toFloat() - running_min[0].item().toFloat());
        }
        
        if (std::isinf(max_value)) {
            running_max[0] = x_max.item().toFloat();
        } else {
            running_max[0] = running_max[0].item().toFloat() + averaging_const *
                (x_max.item().toFloat() - running_max[0].item().toFloat());
        }
    }
    return;
}

TensorQuantizationParams choose_quantization_params(float min, float max, int32_t qmin, int32_t qmax,
    bool preserve_sparsity = false, bool force_scale_power_of_two = false, bool reduce_range = false)
{
    TORCH_CHECK(min <= max, "In choose_quantization_params, min should be less than or equal to mask",
                OPS_ERROR(ErrCode::PARAM));
    if (reduce_range) {
        qmin = qmin / 2;
        qmax = qmax / 2;
    }
    if (min < 0 && max > 0 && preserve_sparsity) {
        int symmetric_qmin = -((qmax - qmin) / 2 + 1);
        int symmetric_qmax = (qmax - qmin) / 2;
        double max_scale = std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
        min = max_scale * symmetric_qmin;
        max = max_scale * symmetric_qmax;
    }
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);
    TORCH_CHECK(qmin < qmax, "In choose_quantization_params, qmin should be less than qmax",
                OPS_ERROR(ErrCode::PARAM));
        double scale = (static_cast<double>(max) - min) / (qmax - qmin);
    if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
        scale = 0.1;
    }
    TORCH_CHECK(scale > 0, "quantization scale should be > 0", OPS_ERROR(ErrCode::VALUE));

    if (force_scale_power_of_two) {
        if (scale < 1) {
            scale = 1.0 / (1 << static_cast<int>(std::floor(std::log(1.0 / scale) / std::log(2))));
        } else {
            scale = 1 << static_cast<int>(std::ceil(std::log(scale) / std::log(2)));
        }
    }
    if (scale < SMALL_SCALE_THRESHOLD) {
        float org_scale = scale;
        scale = SMALL_SCALE_THRESHOLD;
        if (min == 0.0f) {
            max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
        } else if (max == 0.0f) {
            min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
        } else {
            float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
            min *= amplifier;
            max *= amplifier;
        }
    }
    
    double zero_point_from_min = qmin - min / static_cast<double>(scale);
    double zero_point_from_max = qmax - max / static_cast<double>(scale);
    double zero_point_form_min_error = std::abs(qmin) - std::abs(min / static_cast<double>(scale));
    double zero_point_form_max_error = std::abs(qmax) - std::abs(max / static_cast<double>(scale));
    double initial_zero_point = zero_point_form_min_error < zero_point_form_max_error ?
        zero_point_from_min : zero_point_from_max;
    
    if (min < 0 && max > 0 && preserve_sparsity) {
        initial_zero_point = static_cast<double>(qmin + qmax) / 2;
    }
    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point = nearbyint(initial_zero_point);
    }
    TensorQuantizationParams result;
    result.scale = scale;
    result.zero_point = nudged_zero_point;
    return result;
}

std::tuple<at::Tensor, at::Tensor> choose_qparams_fake_quant(const at::Tensor& x,
    const at::Tensor& inp_running_min, const at::Tensor& inp_running_max, at::Tensor& scale, at::Tensor& zero_point,
    bool per_row_fake_quant, bool symmetric_quant, int qmin, int qmax, int ch_axis)
{
    std::tuple<at::Tensor, at::Tensor> fake_quant_out;
    at::Tensor x_min;
    at::Tensor x_max;
    if (per_row_fake_quant) {
        float* x_min_data = inp_running_min.data_ptr<float>();
        float* x_max_data = inp_running_max.data_ptr<float>();
        for (const auto i : c10::irange(inp_running_min.numel())) {
            TensorQuantizationParams result = choose_quantization_params(inp_running_min[i].item().toFloat(),
                inp_running_max[i].item().toFloat(), qmin, qmax, symmetric_quant, false);
            scale[i] = result.scale;
            zero_point[i] = result.zero_point;
        }
        fake_quant_out = at::fake_quantize_per_channel_affine_cachemask(x, scale, zero_point, ch_axis, qmin, qmax);
    } else {
        TensorQuantizationParams result = choose_quantization_params(inp_running_min.item().toFloat(),
            inp_running_max.item().toFloat(), qmin, qmax, symmetric_quant, false);
        scale[0] = result.scale;
        zero_point[0] = result.zero_point;
        auto fake_quant_enabled = at::ones(1, x.options().dtype(at::kLong));
        fake_quant_out = at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(x, scale, zero_point,
            fake_quant_enabled, qmin, qmax);
    }
    return fake_quant_out;
}

std::tuple<at::Tensor, at::Tensor> _fused_moving_avg_obs_fq_helper_common(
    const at::Tensor& self, const at::Tensor& observer_on, const at::Tensor& fake_quant_on,
    at::Tensor& running_min, at::Tensor& running_max, at::Tensor& scale, at::Tensor& zero_point,
    const double averaging_const, const int64_t quant_min, const int64_t quant_max, const int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant)
{
    TORCH_CHECK(ch_axis < self.dim(), "Error in fused_moving_avg_obs_fake_quant: ch_axis must be < self.dim()",
                OPS_ERROR(ErrCode::PARAM));
    auto observe = observer_on.item().toInt();
    if (running_min.numel() == 1) {
        running_min = running_min.reshape(1);
        running_max = running_max.reshape(1);
    }
    if (per_row_fake_quant) {
        at::Tensor y = self;
        if (self.dim() != 2) {
            c10::SmallVector<int64_t, SIZE> res = op_infer::array_to_small_vector(self.sizes());
            std::iota(res.begin(), res.end(), 0);
            res[ch_axis] = 0;
            res[0] = ch_axis;

            y = self.permute(res);
            y = y.flatten(1);
        }
        int64_t size = self.size(ch_axis);
        if (running_min.numel() == 0) {
            float inf = std::numeric_limits<float>::infinity();
            running_min.resize_(size).fill_(inf);
            running_max.resize_(size).fill_(-inf);
            scale.resize_(size);
            zero_point.resize_(size);
        }
        if (observe) {
            calculate_moving_average(y, running_min, running_max, averaging_const, per_row_fake_quant, ch_axis);
        }
    } else {
        if (observe) {
            calculate_moving_average(self, running_min, running_max, averaging_const, per_row_fake_quant, ch_axis);
        }
    }

    auto fake_quant = fake_quant_on.item().toInt();
    if (fake_quant) {
        return choose_qparams_fake_quant(self, running_min, running_max, scale, zero_point, per_row_fake_quant,
                                         symmetric_quant, quant_min, quant_max, ch_axis);
    }
    auto mask = at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
    return std::make_tuple(self.clone(), mask);
}

} // namespace acl_op


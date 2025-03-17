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

#include <c10/util/MathConstants.h>
#include "op_plugin/utils/custom_functions/opapi/fft_plan_op_api.h"


namespace op_api {

    using npu_preparation = at_npu::native::OpPreparation;

    // utils functions

    void copy_quarter(at::Tensor& dst, at::Tensor& src, int64_t prev_n, int64_t factor, bool full_in, bool full_out, int64_t x, int64_t y)
    {
        if (full_in && full_out) {
            auto view = at::slice(dst, 0, 0, prev_n, 1);
            view = at::slice(view, 1, 0, factor, 1);
            view = at::select(view, 2, x);
            view = at::select(view, 2, y);
            view = at::slice(view, 2, 0, factor, 1);

            view.copy_(src);
        } else if (!full_in) {
            auto view = at::slice(dst, 0, 0, prev_n, 1);
            view = at::slice(view, 1, 0, factor, 1);
            view = at::select(view, 2, x);
            view = at::select(view, 2, y);
            view = at::slice(view, 2, 0, (factor / 2) + 1, 1);

            src = at::slice(src, 2, 0, (factor / 2) + 1, 1);
            view.copy_(src);
        } else if (!full_out) {
            auto view = at::slice(dst, 0, 0, prev_n, 1);
            view = at::slice(view, 1, 0, (factor / 2) + 1, 1);
            view = at::select(view, 2, x);
            view = at::select(view, 2, y);
            view = at::slice(view, 2, 0, factor, 1);

            src = at::slice(src, 1, 0, (factor / 2) + 1, 1);
            view.copy_(src);
        }
        // 不能同时!full_in和!full_out
    }

    at::Tensor one_rotate_matrix(int64_t prev_n, PlanKey plan_key, std::vector<int64_t> factors, int index)
    {
        auto options = at::TensorOptions()
            .dtype(at::ScalarType::Double)
            .layout(at::Layout::Strided)
            .device(at::DeviceType::CPU);

        int64_t factor = factors[index];
        // compute theta
        std::array<int64_t, 3> dim_shape{prev_n, 1, 1};
        auto first_dim = at::reshape(at::arange(0, prev_n, 1, options), dim_shape);

        dim_shape = {1, factor, 1};
        auto second_dim = at::reshape(at::arange(0, prev_n * factor, prev_n, options), dim_shape);

        dim_shape = {1, 1, factor};

        auto third_dim = at::reshape(at::arange(0, -factor, -1, options), dim_shape);
        third_dim = at::mul(third_dim, c10::pi<double_t> * 2 / (prev_n * factor));

        auto theta = at::add(first_dim, second_dim);
        theta = at::mul(theta, third_dim);

        // compute rotate
        auto triangle = at::empty_like(theta);

        int64_t out_n = ((plan_key.plan_mode == PlanMode::r2c) && (index == static_cast<int64_t>(factors.size() - 1))) ? (factor / 2) + 1 : factor;
        int64_t out_complex = ((plan_key.plan_mode == PlanMode::c2r) && (index == static_cast<int64_t>(factors.size() - 1))) ? static_cast<int64_t>(1) : static_cast<int64_t>(2);
        int64_t in_n = factor;
        int64_t in_complex = ((plan_key.plan_mode == PlanMode::r2c || plan_key.plan_mode == PlanMode::r2c_bothside) && (index == 0)) ? static_cast<int64_t>(1) : static_cast<int64_t>(2);

        std::array<int64_t, 5> rotate_shape{prev_n, out_n, out_complex, in_complex, in_n};
        auto rotate_matrix = at::empty(rotate_shape, options);
        at::cos_outf(theta, triangle);
        copy_quarter(rotate_matrix, triangle, prev_n, factor, in_n == factor, out_n == factor, 0, 0);
        if (in_complex == 2 && out_complex == 2) {
            copy_quarter(rotate_matrix, triangle, prev_n, factor, in_n == factor, out_n == factor, 1, 1);
        }
        at::sin_outf(theta, triangle);
        if (plan_key.is_forward) {
            if (out_complex == 2) {
                copy_quarter(rotate_matrix, triangle, prev_n, factor, in_n == factor, out_n == factor, 1, 0);
            }
            if (in_complex == 2) {
                at::neg_(triangle);
                copy_quarter(rotate_matrix, triangle, prev_n, factor, in_n == factor, out_n == factor, 0, 1);
            }
        } else {
            if (in_complex == 2) {
                copy_quarter(rotate_matrix, triangle, prev_n, factor, in_n == factor, out_n == factor, 0, 1);
            }
            if (out_complex == 2) {
                at::neg_(triangle);
                copy_quarter(rotate_matrix, triangle, prev_n, factor, in_n == factor, out_n == factor, 1, 0);
            }
        }
        rotate_matrix = rotate_matrix.to(plan_key.scalar_dtype);

        // transpose dims
        std::vector<int64_t> split_shape(index + 4);
        std::copy(factors.begin(), factors.begin() + index, split_shape.rbegin() + 4);
        std::copy(rotate_shape.begin() + 1, rotate_shape.end(), split_shape.begin() + index);

        std::vector<int64_t> permute_shape(index + 4);
        std::iota(permute_shape.rbegin() + 4, permute_shape.rend(), int64_t{0});
        std::iota(permute_shape.begin() + index, permute_shape.end(), int64_t{index});

        rotate_matrix = rotate_matrix.reshape(split_shape).permute(permute_shape).contiguous();

        std::array<int64_t, 3> reshape_shape{prev_n, out_n * out_complex, in_complex * in_n};
        return at::reshape(rotate_matrix, reshape_shape);
    }

    std::vector<int64_t> factorize(int64_t size)
    {
        std::vector<int64_t> factors{};

        int64_t bound = std::sqrt(size);
        for (int64_t factor = 2; factor <= bound;) {
            if (size % factor == 0) {
                factors.push_back(factor);
                size /= factor;
                bound = std::sqrt(size);
            } else {
                factor++;
            }
        }

        if (size != 1) {
            factors.push_back(size);
        }

        return factors;
    }

    std::vector<int64_t> make_sure_first_alpha(std::vector<int64_t> &factors)
    {
        if ((factors.size() == 1) || (factors[0] >= 16)) {
            return factors;
        }
        for (size_t i = 1; i < factors.size(); i++) {
            if (factors[i] >= 16) {
                int64_t tmp = factors[0];
                factors[0] = factors[i];
                factors[i] = tmp;
                break;
            }
        }
        return factors;
    }

    std::vector<int64_t> merge(const std::vector<int64_t> &factors_)
    {
        TORCH_CHECK(factors_.size() > 0, "size must be greater than 0" + OPS_ERROR(ErrCode::PARAM));
        std::vector<int64_t> factors(factors_.size());
        std::copy(factors_.rbegin(), factors_.rend(), factors.begin());
        std::vector<int64_t> merged_factors{};
        std::vector<bool> is_merged(factors.size());
        for (size_t i = 0; i < is_merged.size(); i++) {
            is_merged[i] = false;
        }
        for (size_t i = 0; i < factors.size(); i++) {
            int64_t factor = 1;
            for (size_t j = i; j < factors.size(); j++) {
                if (is_merged[j]) {
                    continue;
                }
                if ((factor == 1) || ((factors[j] * factor) <= FACTOR_BOUND)) {
                    factor *= factors[j];
                    is_merged[j] = true;
                }
            }
            if (factor == 1) {
                break;
            }
            merged_factors.push_back(factor);
        }
        std::sort(merged_factors.begin(), merged_factors.end());
        if (merged_factors.size() > NDIM_BOUND) {
            std::vector<int64_t> merged_factors_(NDIM_BOUND);
            std::copy(merged_factors.begin() + merged_factors.size() - NDIM_BOUND, merged_factors.end(), merged_factors_.begin());
            for (size_t i = 0; i < (merged_factors.size() - NDIM_BOUND); i++) {
                auto min_ = std::min_element(merged_factors_.begin(), merged_factors_.end());
                *min_ *= merged_factors[i];
            }
            std::sort(merged_factors_.begin(), merged_factors_.end());
            return make_sure_first_alpha(merged_factors_);
        }
        return make_sure_first_alpha(merged_factors);
    }

    FFTPlanItem make_plan(PlanKey &plan_key)
    {
        TORCH_CHECK(plan_key.prb_size > 1, "prb_size must be greater than 1" + OPS_ERROR(ErrCode::PARAM));

        std::vector<int64_t> factors = factorize(plan_key.prb_size);
        if (plan_key.prb_size <= 128) {
            factors = {plan_key.prb_size};
        } else {
            factors = merge(factors);
        }

        FFTPlanItem fftPlanItem{factors};

        int64_t factor;
        int64_t prev_n = 1;
        for (size_t i = 0; i < factors.size(); i++) {
            at::Tensor device_tensor = npu_preparation::copy_tensor_host_to_device(one_rotate_matrix(prev_n, plan_key, factors, i));
            fftPlanItem.insert_rotate_matrix(i, device_tensor);
            prev_n *= factors[i];
        }
        return fftPlanItem;
    }

} // namespace op_api

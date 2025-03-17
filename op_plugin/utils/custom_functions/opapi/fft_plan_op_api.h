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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_FFT_PLAN_OP_API__
#define __TORCH_NPU_OP_PLUGIN_UTILS_FFT_PLAN_OP_API__

#include <array>
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

#define FACTOR_BOUND 32
#define NDIM_BOUND 5

namespace op_api {

    enum PlanMode {
        c2c,
        r2c,
        c2r,
        r2c_bothside,
    };


    // PlanKey

    class PlanKey {
    public:
        PlanKey();
        int64_t prb_size;
        bool is_forward;
        PlanMode plan_mode;
        at::ScalarType scalar_dtype;
        PlanKey(int64_t size, bool inv, PlanMode mode, at::ScalarType dtype_);
    };

    inline PlanKey::PlanKey()
        : prb_size(1),
          is_forward(true),
          plan_mode(PlanMode::c2c),
          scalar_dtype(at::ScalarType::ComplexHalf) {}

    inline PlanKey::PlanKey(int64_t size, bool inv, PlanMode mode, at::ScalarType dtype_)
        : prb_size(size),
          is_forward(inv),
          plan_mode(mode),
          scalar_dtype(dtype_) {}

    inline bool operator==(const PlanKey &one, const PlanKey &other)
    {
        return one.prb_size == other.prb_size
            && one.is_forward == other.is_forward
            && one.plan_mode == other.plan_mode;
    }


    // FFTPlanItem

    class FFTPlanItem {
    public:
        FFTPlanItem() {}
        FFTPlanItem(std::vector<int64_t> factors_);
        void insert_rotate_matrix(int i, at::Tensor matrix);
        int get_size();
        int64_t get_prev_n(int i);
        int64_t get_factor(int i);
        std::vector<int64_t>& get_factors();
        at::Tensor& get_rotate_matrix(int i);
        std::vector<at::Tensor>& get_rotate_matrices();
    private:
        std::vector<at::Tensor> matrices;
        std::vector<int64_t> factors;
    };

    inline FFTPlanItem::FFTPlanItem(std::vector<int64_t> factors_) : matrices(factors_.size()), factors(factors_) {
    }

    inline int FFTPlanItem::get_size()
    {
        return matrices.size();
    }

    inline void FFTPlanItem::insert_rotate_matrix(int i, at::Tensor matrix)
    {
        TORCH_CHECK(i < get_size(), "i must less than size" + OPS_ERROR(ErrCode::PARAM));
        matrices[i] = matrix;
    }

    inline int64_t FFTPlanItem::get_prev_n(int i)
    {
        return get_rotate_matrix(i).sizes()[0];
    }

    inline int64_t FFTPlanItem::get_factor(int i)
    {
        TORCH_CHECK(i < get_size(), "i must less than size" + OPS_ERROR(ErrCode::PARAM));
        return factors[i];
    }

    inline std::vector<int64_t>& FFTPlanItem::get_factors()
    {
        return factors;
    }

    inline std::vector<at::Tensor>& FFTPlanItem::get_rotate_matrices()
    {
        return matrices;
    }

    inline at::Tensor& FFTPlanItem::get_rotate_matrix(int i)
    {
        TORCH_CHECK(i < get_size(), "i must less than size" + OPS_ERROR(ErrCode::PARAM));
        return matrices[i];
    }

    // utils interfaces
    FFTPlanItem make_plan(PlanKey &plan_key);

} // namespace op_api

#endif

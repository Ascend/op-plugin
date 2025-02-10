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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

static const int POSITIVE = 1;
static const int NEGETIVE = 2;
static const int SIZE_T_TWICE = 2;
static const int DIM_THRESHOLD = 8;

namespace {

void check_negetive(const at::Tensor &self, at::IntArrayRef pad, std::vector<int64_t> &out_shape, int &sign_symbol)
{
    auto self_dim = self.dim();
    auto pad_cover = static_cast<int64_t>(pad.size()) / 2;
    bool hasZero = false;
    // pad中每个值都不能让out的shape小于0, 如果pad中存在正数, 则out的shape中不能有0
    for (auto i = 0; i < pad_cover; ++i) {
        auto cur_shape = self.sizes()[self_dim - i -1];
        auto begin = pad[SIZE_T_TWICE * i];
        auto end = pad[SIZE_T_TWICE * i + 1];
        auto newShape = cur_shape + begin + end;
        auto min = std::min(begin, end);
        min = std::min(min, begin + end);
        TORCH_CHECK(cur_shape + min >= 0, "The input size ", cur_shape, "plus padding ", begin, " and ", end,
                    " resluted in a negative output size, which is invalid. Check dimension ",
                    self_dim - i - 1, " of yout input." + OPS_ERROR(ErrCode::PARAM));
        if (begin > 0 || end > 0) {
            sign_symbol |= POSITIVE;
        }
        if (begin < 0 || end < 0) {
            sign_symbol |= NEGETIVE;
        }
        if (newShape == 0) {
            hasZero = true;
        }
    }
    if (hasZero && ((sign_symbol & POSITIVE) == POSITIVE)) {
        TORCH_CHECK(false, "The output size with zero element is invalid, please check your input." + OPS_ERROR(ErrCode::PARAM));
    }
    return ;
}

void check_params(int l_pad, int l_diff,  at::IntArrayRef input_sizes, at::IntArrayRef pad)
{
    for (int64_t i = 0; i < l_pad; i++) {
        auto pad_idx = static_cast<int64_t>(pad.size()) - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        TORCH_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ", pad[pad_idx],
            " and ", pad[pad_idx + 1],
            "resulted in a negative output size, "
            "which is invalid. Check dimension ",
            l_diff + i, "of your input." + OPS_ERROR(ErrCode::PARAM));
    }
}

at::Tensor do_indexing(const at::Tensor &self, at::IntArrayRef pad)
{
    TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size(),
        OPS_ERROR(ErrCode::PARAM));

    int64_t max_pad_size = 2 * self.dim();
    auto pad_vec = op_infer::array_to_small_vector(pad);
    for (auto i = 0 ; i < pad_vec.size(); i++) {
        pad_vec[i] = pad_vec[i] >= 0 ? 0 : pad_vec[i];
    }
    if (static_cast<int64_t>(pad.size()) < max_pad_size) {
        for (int64_t i = 0; i < max_pad_size - static_cast<int64_t>(pad.size()); i++) {
            pad_vec.emplace_back(0);
        }
    }

    c10::SmallVector<int64_t, SIZE> begin_list(self.dim(), 0);
    c10::SmallVector<int64_t, SIZE> end_list;
    for (auto i : self.sizes()) {
        end_list.push_back(i);
    }
    c10::SmallVector<int64_t, SIZE> strides(self.dim(), 1);

    at::Tensor result = self;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (pad_vec[max_pad_size - 2 * (i + 1)] == 0 && pad_vec[max_pad_size - 1 - 2 * i] == 0) {
            continue;
        }
        begin_list[i] = begin_list[i] + (-pad_vec[max_pad_size - 2 * (i + 1)]);
        end_list[i] = end_list[i] + pad_vec[max_pad_size - 1 - 2 * i];
        result = acl_op::npu_indexing(result, begin_list, end_list, strides, 0, 0, 0, 0, 0);
        begin_list[i] = 0;
        end_list[i] = result.size(i);
    }

    return result;
}

} // namespace

at::Tensor constant_pad_nd(const at::Tensor &self, at::IntArrayRef pad, const at::Scalar &value)
{
    TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size(),
        OPS_ERROR(ErrCode::PARAM));

    int sign_symbol = 0; // 0代表pad全0或没有元素, 1代表有正数, 2代表有负数, 3代表同时有正数和负数
    auto input_sizes = self.sizes();
    auto l_inp = self.dim();
    auto l_pad = static_cast<int64_t>(pad.size()) / 2;
    auto l_diff = l_inp - l_pad;
    TORCH_CHECK(l_inp >= l_pad,
        "Length of pad should be no more than twice the number of "
        "dimensions of the input. Pad length is ",
        pad.size(), "while the input has ", l_inp, "dimensions."
        + OPS_ERROR(ErrCode::PARAM));

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < (size_t)l_diff; i++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    check_params(l_pad, l_diff, input_sizes, pad);

    c10::SmallVector<int64_t, N> vector_int;
    c10::SmallVector<int64_t, N> paddings_vector = op_infer::array_to_small_vector(pad);
    paddings_vector.resize(2 * self.dim(), 0);
    for (int64_t i = static_cast<int>(paddings_vector.size()); i > 0; i -= 2) {
        paddings_vector[i - 2] = paddings_vector[i - 2] > 0 ? paddings_vector[i - 2] : 0;
        paddings_vector[i - 1] = paddings_vector[i - 1] > 0 ? paddings_vector[i - 1] : 0;

        vector_int.emplace_back(paddings_vector[i - 2]);
        vector_int.emplace_back(paddings_vector[i - 1]);
    }
    for (int64_t i = 0; i < l_pad; i++) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        auto positive_pad1 = pad[pad_idx] > 0 ? pad[pad_idx] : 0;
        auto positive_pad2 = pad[pad_idx + 1] > 0 ? pad[pad_idx + 1] : 0;
        auto new_dim = input_sizes[l_diff + i] + positive_pad1 + positive_pad2;
        new_shape.emplace_back(new_dim);
    }
    check_negetive(self, pad, new_shape, sign_symbol);
    at::Tensor result = npu_preparation::apply_tensor(self, new_shape);
    if (sign_symbol != NEGETIVE) {   // pad参数中包含正数.
        if (self.numel() == 0) {
            acl_op::fill_(result, value);
            return result;
        }
        float val = op_plugin::utils::get_scalar_float_value(value);
        at::Tensor value_tensor = at::empty({1}, self.options());
        acl_op::fill_(value_tensor, val);

        at_npu::native::OpCommand cmd;
        if (l_inp <= DIM_THRESHOLD) {
            cmd.Name("PadV3")
                .Input(self)
                .Input(vector_int, at::kInt)
                .Input(value_tensor)
                .Output(result)
                .Attr("mode", (string) "constant")
                .Attr("paddings_contiguous", true)
                .Run();
        } else {
            cmd.Name("PadV3")
                .Input(self)
                .Input(vector_int, at::kInt)
                .Input(value_tensor)
                .Output(result)
                .Attr("_exclude_engines", (string) "AiCore")
                .Attr("mode", (string) "constant")
                .Attr("paddings_contiguous", true)
                .Run();
        }
    }
    if ((sign_symbol & NEGETIVE) != 0) {     // pad参数中存在负数.
        if (sign_symbol == NEGETIVE) {
            return do_indexing(self, pad);
        } else {
            return do_indexing(result, pad);
        }
    }
    return result;
}
} // namespace acl_op

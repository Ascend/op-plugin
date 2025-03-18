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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor repeat_interleave_backward_int(const at::Tensor& input_grad, const at::Tensor& self, int64_t repeats,
    c10::optional<int64_t> dim)
{
    int64_t dim_pos = dim.value_or(-1);
    int64_t grad_dim = input_grad.dim();
    if (dim_pos < 0) {
        dim_pos = dim_pos + grad_dim;
    }
    at::SmallVector<int64_t, SIZE> input_grad_new_shape;
    for (int64_t dim_index = 0; dim_index < grad_dim; dim_index++) {
        if (dim_index != dim_pos) {
            input_grad_new_shape.emplace_back(input_grad.size(dim_index));
        } else {
            TORCH_CHECK(repeats != 0, "repeats must be not zero", OPS_ERROR(ErrCode::VALUE));
            input_grad_new_shape.emplace_back(input_grad.size(dim_index) / repeats);
            input_grad_new_shape.emplace_back(repeats);
        }
    }
    auto input_grad_reshape = input_grad.view(input_grad_new_shape);
    auto result = input_grad_reshape.sum(dim_pos + 1).view(self.sizes());
    return result;
}

at::Tensor repeat_interleave_backward_tensor(const at::Tensor& input_grad, const at::Tensor& self, const at::Tensor& repeats,
    c10::optional<int64_t> dim)
{
    if (repeats.numel() == 1) {
        int64_t repeats_int = repeats.item().toLong();
        return repeat_interleave_backward_int(input_grad, self, repeats_int, dim);
    }
    int64_t dim_pos = dim.value_or(-1);
    int64_t grad_dim = input_grad.dim();
    if (dim_pos < 0) {
        dim_pos = dim_pos + grad_dim;
    }
    at::SmallVector<int64_t, SIZE> result_shape;
    for (int64_t dim_index = 0; dim_index < grad_dim; dim_index++) {
        if (dim_index != dim_pos) {
            result_shape.emplace_back(input_grad.size(dim_index));
        } else {
            result_shape.emplace_back(repeats.size(0));
        }
    }
    at::Tensor result = npu_preparation::apply_tensor_with_format(result_shape, input_grad.options(), ACL_FORMAT_ND);
    EXEC_NPU_CMD(aclnnRepeatInterleaveGrad, input_grad, repeats, dim_pos, result);
    result = result.view(self.sizes());
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor repeat_interleave_backward_one_repeat(const at::Tensor& input_grad, const at::Tensor& self, int64_t repeats,
    c10::optional<int64_t> dim)
{
    int64_t dim_pos = dim.value_or(-1);
    int64_t grad_dim = input_grad.dim();
    if (dim_pos < 0) {
        dim_pos = dim_pos + grad_dim;
    }
    at::SmallVector<int64_t, SIZE> input_grad_new_shape;
    for (int64_t dim_index = 0; dim_index < grad_dim; dim_index++) {
        if (dim_index != dim_pos) {
            input_grad_new_shape.emplace_back(input_grad.size(dim_index));
        } else {
            TORCH_CHECK(repeats != 0, "repeats must be not zero", OPS_ERROR(ErrCode::VALUE));
            input_grad_new_shape.emplace_back(input_grad.size(dim_index) / repeats);
            input_grad_new_shape.emplace_back(repeats);
        }
    }
    auto input_grad_reshape = input_grad.view(input_grad_new_shape);
    auto result = input_grad_reshape.sum(dim_pos + 1).view(self.sizes());
    return result;
}

at::Tensor repeat_interleave_backward_int_symint(const at::Tensor& input_grad, const at::Tensor& self, c10::SymInt repeats,
    c10::optional<int64_t> dim)
{
    int64_t repeats_int = repeats.expect_int();
    return repeat_interleave_backward_one_repeat(input_grad, self, repeats_int, dim);
}

at::Tensor repeat_interleave_backward_tensor(const at::Tensor& grad, const at::Tensor& self, const at::Tensor& repeats,
    c10::optional<int64_t> dim)
{
    if (repeats.numel() == 1) {
        int64_t repeats_int = repeats.item().toLong();
        return repeat_interleave_backward_one_repeat(grad, self, repeats_int, dim);
    }
    int64_t dim_pos = dim.value_or(-1);
    int64_t grad_dim = grad.dim();
    if (dim_pos < 0) {
        dim_pos = dim_pos + grad_dim;
    }
    at::SmallVector<int64_t, SIZE> result_shape;
    for (int64_t dim_index = 0; dim_index < grad_dim; dim_index++) {
        if (dim_index != dim_pos) {
            result_shape.emplace_back(grad.size(dim_index));
        } else {
            result_shape.emplace_back(repeats.size(0));
        }
    }
    at::Tensor result = npu_preparation::apply_tensor_with_format(result_shape, grad.options(), ACL_FORMAT_ND);
    EXEC_NPU_CMD(aclnnRepeatInterleaveGrad, grad, repeats, dim_pos, result);
    result = result.view(self.sizes());
    return result;
}
#endif

}

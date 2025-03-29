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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/OpUtils.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>;

namespace {
at::Tensor &rotary_mul_nocheck(at::Tensor &y, const at::Tensor &x, const at::Tensor &r1, const at::Tensor &r2)
{
    if (x.sizes()[3] % 64 != 0) {
        std::vector<at::Tensor> chunkResult = x.chunk(2, -1);
        at::Tensor x_new = at::cat({chunkResult[1] * (-1), chunkResult[0]}, 3);
        y = at::mul(r1, x) + at::mul(r2, x_new);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("RotaryMul").Input(x).Input(r1).Input(r2).Output(y).Run();
    }
    return y;
}

tensor_list rotary_mul_backward_nocheck(at::Tensor &dx, at::Tensor &dr1, at::Tensor &dr2, const at::Tensor &x,
                                        const at::Tensor &r1, const at::Tensor &r2, const at::Tensor &dy)
{
    TORCH_CHECK(x.dim() == 4, "The dim of input tensor [x] shoule equal to four." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(r1.dim() == 4, "The dim of input tensor [r1] shoule equal to four." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(r2.dim() == 4, "The dim of input tensor [r2] shoule equal to four." + OPS_ERROR(ErrCode::PARAM));
    bool check_support = true;
    int64_t broadcast_dim_num = 1;
    for (int64_t i = 0; i < x.dim(); i++) {
        if (x.sizes()[i] != r1.sizes()[i]) {
            broadcast_dim_num = broadcast_dim_num * x.sizes()[i];
        }
        if (broadcast_dim_num > 1024) {
            check_support = false;
            break;
        }
    }
    if (x.sizes()[3] % 64 != 0 || check_support == false) {
        at::Tensor x_grad_mul = at::mul(x, dy);
        at::Tensor x1_grad_mul = at::mul(r1, dy);
        at::Tensor x2_grad_mul = at::mul(r2, dy);
        std::vector<at::Tensor> x2_chunk = x2_grad_mul.chunk(2, -1);
        at::Tensor x2_chunk_cat = at::cat({x2_chunk[1], x2_chunk[0] * (-1)}, 3);
        dx = at::add(x2_chunk_cat, x1_grad_mul);
        c10::SmallVector<int64_t, SIZE> dims;
        for (int i = 0; i < 4; i++) {
            if (x.sizes()[i] != r1.sizes()[i]) {
                dims.emplace_back(i);
            }
        }
        std::vector<at::Tensor> x_chunk = x.chunk(2, -1);
        at::Tensor xq_chunk_cat = at::cat({x_chunk[1] * (-1), x_chunk[0]}, 3);
        at::Tensor dr2_result = at::mul(xq_chunk_cat, dy);
        dr2 = at::sum(dr2_result, dims, true);
        dr1 = at::sum(x_grad_mul, dims, true);
    } else {
        if (r1.requires_grad() && r2.requires_grad()) {
            bool need_backward = true;
            at_npu::native::OpCommand cmd;
            cmd.Name("RotaryMulGrad")
                .Input(x)
                .Input(r1)
                .Input(r2)
                .Input(dy)
                .Output(dx)
                .Output(dr1)
                .Output(dr2)
                .Attr("need_backward", need_backward)
                .Run();
        } else {
            bool need_backward = false;
            at_npu::native::OpCommand cmd;
            cmd.Name("RotaryMulGrad")
                .Input(x)
                .Input(r1)
                .Input(r2)
                .Input(dy)
                .Output(dx)
                .Output(dr1)
                .Output(dr2)
                .Attr("need_backward", need_backward)
                .Run();
        }
    }
    return std::tie(dx, dr1, dr2);
}
} // namespace

at::Tensor npu_rotary_mul(const at::Tensor &self, const at::Tensor &r1, const at::Tensor &r2, c10::string_view rotary_mode)
{
    TORCH_CHECK(rotary_mode == "half",
        "npu_rotary_mul in aclop only support rotary_mode with half, but got ", rotary_mode,
        OPS_ERROR(ErrCode::PARAM));
    int64_t mode = op_plugin::utils::get_rotary_mode(rotary_mode);
    at::Tensor result = npu_preparation::apply_tensor(self);
    rotary_mul_nocheck(result, self, r1, r2);
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(const at::Tensor &grad, const at::Tensor &self,
                                                                       const at::Tensor &r1, const at::Tensor &r2, c10::string_view rotary_mode)
{
    TORCH_CHECK(rotary_mode == "half",
        "npu_rotary_mul_backward in aclop only support rotary_mode with half, but got ", rotary_mode,
        OPS_ERROR(ErrCode::PARAM));
    int64_t mode = op_plugin::utils::get_rotary_mode(rotary_mode);
    at::Tensor dx = npu_preparation::apply_tensor(self);
    at::Tensor dr1 = npu_preparation::apply_tensor(r1);
    at::Tensor dr2 = npu_preparation::apply_tensor(r2);
    rotary_mul_backward_nocheck(dx, dr1, dr2, self, r1, r2, grad);
    return std::tie(dx, dr1, dr2);
}
} // namespace acl_op

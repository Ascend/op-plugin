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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {

at::Tensor& roll_out_npu_no_transpose(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef shifts,
    at::IntArrayRef dims)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Roll")
        .Input(self)
        .Output(result)
        .Attr("shifts", shifts)
        .Attr("dims", dims)
        .Run();

    return result;
}

at::Tensor& roll_transpose(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t axis,
    int64_t first_dim,
    at::IntArrayRef shifts,
    int64_t id)
{
    c10::SmallVector<int64_t, SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
        perm.emplace_back(i);
    }
    std::swap(perm[axis], perm[first_dim]);
    at::Tensor transpose_self = acl_op::npu_transpose(self, perm, true);
    auto output_size = op_infer::transpose_npu_output_size(result, perm);
    at::Tensor transpose_result = npu_preparation::apply_tensor(self, output_size);
    c10::SmallVector<int64_t, SIZE> dim = {first_dim};
    c10::SmallVector<int64_t, SIZE> shift_bak = {shifts[id]};
    at::IntArrayRef dim_now = at::IntArrayRef(dim);
    at::IntArrayRef shift_now = at::IntArrayRef(shift_bak);
    roll_out_npu_no_transpose(transpose_result, transpose_self, shift_now, dim_now);
    acl_op::npu_transpose_out(transpose_result, perm, true, result);
    return result;
}

at::Tensor& roll_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef shifts,
    at::IntArrayRef dims)
{
    if (dims.size() == 0) {
        roll_out_npu_no_transpose(result, self, shifts, dims);
    } else {
        TORCH_CHECK(dims.size() == shifts.size(),
                    "The size of shifts and dims should be the same when the size of dims is not 0."
                    + OPS_ERROR(ErrCode::PARAM));
        int64_t first_dim = op_plugin::utils::make_warp_dim(0, self.dim());
        for (uint i = 0; i < dims.size(); i++) {
            int64_t axis = op_plugin::utils::make_warp_dim(dims[i], self.dim());
            if (i == 0) {
                if (axis == first_dim) {
                    c10::SmallVector<int64_t, SIZE> dim = {first_dim};
                    c10::SmallVector<int64_t, SIZE> shift_bak = {shifts[i]};
                    at::IntArrayRef dim_now = at::IntArrayRef(dim);
                    at::IntArrayRef shift_now = at::IntArrayRef(shift_bak);
                    roll_out_npu_no_transpose(result, self, shift_now, dim_now);
                } else {
                    roll_transpose(result, self, axis, first_dim, shifts, i);
                }
            } else {
                roll_transpose(result, result, axis, first_dim, shifts, i);
            }
        }
    }
    return result;
}
} // namespace

at::Tensor roll(
    const at::Tensor& self,
    at::IntArrayRef shifts,
    at::IntArrayRef dims)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    roll_out_npu(result, self, shifts, dims);
    return result;
}

} // namespace acl_op

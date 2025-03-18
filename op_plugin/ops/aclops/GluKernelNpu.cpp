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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& glu_npu_out_nocheck(at::Tensor& result, const at::Tensor& self, int64_t dim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("GLU")
        .Input(self)
        .Output(result)
        .Attr("dim", dim)
        .Run();
    return result;
}
}  // namespace

at::Tensor& glu_out(const at::Tensor& self, int64_t dim, at::Tensor& out)
{
    auto output_size = op_infer::glu_npu_output_size(self, dim);
    npu_preparation::CheckOut(
        {self},
        out,
        self,
        output_size);

    TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional at::Tensors" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    const int64_t n_in = self.size(wrap_dim);
    TORCH_CHECK(n_in % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", n_in,
                OPS_ERROR(ErrCode::PARAM));

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        glu_npu_out_nocheck(contiguous_result, self, dim);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        glu_npu_out_nocheck(out, self, dim);
    }

    return out;
}

at::Tensor glu(const at::Tensor& self, int64_t dim)
{
    TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional at::Tensors" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    const int64_t n_in = self.size(wrap_dim);
    TORCH_CHECK(n_in % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", n_in,
                OPS_ERROR(ErrCode::PARAM));

    auto output_size = op_infer::glu_npu_output_size(self, dim);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    glu_npu_out_nocheck(result, self, dim);
    return result;
}
}  // namespace acl_op

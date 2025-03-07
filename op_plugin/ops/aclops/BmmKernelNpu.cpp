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

#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
    using npu_format_helper = at_npu::native::FormatHelper;
    using npu_preparation = at_npu::native::OpPreparation;
    using npu_utils = at_npu::native::NpuUtils;

namespace {
    at::Tensor &bmm_out_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &mat2)
    {
        bool is_self_t = op_plugin::utils::is_transpose_last_two_dims(self);
        bool is_mat2_t = op_plugin::utils::is_transpose_last_two_dims(mat2);
        at::Tensor contiguous_self = is_self_t ? self : npu_utils::format_contiguous_add_copy_optimize(self);
        at::Tensor contiguous_mat2 = is_mat2_t ? mat2 : npu_utils::format_contiguous_add_copy_optimize(mat2);

        at_npu::native::OpCommand cmd;
        cmd.Name("BatchMatMul")
            .InputWithoutContiguous(contiguous_self)
            .InputWithoutContiguous(contiguous_mat2)
            .Output(result)
            .Attr("adj_x1", is_self_t)
            .Attr("adj_x2", is_mat2_t)
            .Run();
        return result;
    }
} // namespace

at::Tensor &bmm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &result)
{
    TORCH_CHECK(self.device() == mat2.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        (torch_npu::utils::is_npu(self) ? "npu" : "cpu"),
        " and ",
        (torch_npu::utils::is_npu(mat2) ? "npu! " : "cpu! "),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.scalar_type() != at::ScalarType::Char && mat2.scalar_type() != at::ScalarType::Char,
        "bmm_out is not support int8 dtype" + OPS_ERROR(ErrCode::TYPE))
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};
    npu_preparation::CheckOut(
        {self, mat2},
        result,
        self,
        output_size);

    if (!result.is_contiguous()) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        bmm_out_nocheck(contiguous_result, self, mat2);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        bmm_out_nocheck(result, self, mat2);
    }
    return result;
}

at::Tensor bmm(const at::Tensor &self, const at::Tensor &mat2)
{
    TORCH_CHECK(self.device() == mat2.device(),
                "Expected all tensors to be on the same device, but found at least two devices, ",
                (torch_npu::utils::is_npu(self) ? "npu" : "cpu"),
                " and ",
                (torch_npu::utils::is_npu(mat2) ? "npu! " : "cpu! "),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "bmm expect self 3D tensors, but got: ",
        self.dim(), "D and ", mat2.dim(), "D" + OPS_ERROR(ErrCode::PARAM));
    // 1.cann bmm support int8(input)->int32(out)
    // 2.onnx can support because of change y dtype to be int32.
    // 3.torch need int8(input)->int8(out), cann can not support.
    TORCH_CHECK(self.scalar_type() != at::ScalarType::Char && mat2.scalar_type() != at::ScalarType::Char,
                "bmm is not support int8 dtype" + OPS_ERROR(ErrCode::TYPE))
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};

    at::Tensor result;
    bool need_nd_out = false;
    // Check if the mm output is specified as NCHW.
    // It will be deleted after the overall strategy of the NLP model is formulated.
    if ((self.scalar_type() == at::kHalf)) {
        // check is 16-algined with high-performance
        auto is_aligin = [&]() {
            return (!(static_cast<uint64_t>(self.size(1)) & 0xF)) && (!(static_cast<uint64_t>(self.size(2)) & 0xF)) &&
                    (!(static_cast<uint64_t>(mat2.size(1)) & 0xF)) && (!(static_cast<uint64_t>(mat2.size(2)) & 0xF));
        };
        static auto mm_bmm_nd = !at_npu::native::env::CheckMmBmmNDDisable();
        static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
        // There is a data trampling problem in non-aligned scenes.
        // For the time being, only aligned scenes are supported.
        if (npu_format_helper::IsBaseFormatType(self) && npu_format_helper::IsBaseFormatType(mat2) &&
            mm_bmm_nd && ((is_support_nd_out && op_plugin::utils::is_nd_to_nz_on_fly(self, mat2)) ||
            (!is_support_nd_out && is_aligin()))) {
            result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_ND);
        } else {
            result = npu_preparation::apply_tensor_with_format(output_size, self.options(),
                                                               ACL_FORMAT_FRACTAL_NZ, true);
            need_nd_out = mm_bmm_nd;
        }
    } else {
        result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_ND);
    }

    bmm_out_nocheck(result, self, mat2);
    if (need_nd_out) {
        result = at_npu::native::custom_ops::npu_format_cast(result, ACL_FORMAT_ND);
    }
    return result;
}
} // namespace acl_op

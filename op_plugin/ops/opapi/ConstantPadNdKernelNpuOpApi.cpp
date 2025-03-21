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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
at::Tensor constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value)
{
    DO_COMPATIBILITY(aclnnConstantPadNd, acl_op::constant_pad_nd(self, pad, value));
    TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size(), OPS_ERROR(ErrCode::PARAM));

    auto input_sizes = self.sizes();
    auto l_inp = static_cast<uint64_t>(self.dim());
    auto l_pad = pad.size() / 2;
    auto l_diff = l_inp - l_pad;
    TORCH_CHECK(l_inp >= l_pad, "Length of pad should be no more than twice the number of "
                                "dimensions of the input. Pad length is ",
                pad.size(), "while the input has ",
                l_inp, "dimensions.", OPS_ERROR(ErrCode::PARAM));

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < static_cast<size_t>(l_diff); i++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    for (size_t i = 0; i < static_cast<size_t>(l_pad); i++) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        TORCH_CHECK(new_dim >= 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                    pad[pad_idx], " and ", pad[pad_idx + 1], "resulted in a negative output size, "
                                                             "which is invalid. Check dimension ",
                    l_diff + i, "of your input.", OPS_ERROR(ErrCode::PARAM));
        new_shape.emplace_back(new_dim);
    }

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, new_shape);

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnConstantPadNd, self, pad, value, result);

    return result;
}
}

// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
const int64_t INT4_NUMS_IN_INT32_SPACE = 8;
const int64_t FP4_IN_UINT8_NUM = 2;
using npu_preparation = at_npu::native::OpPreparation;

c10::SmallVector<int64_t, SIZE> cal_out_size(const at::Tensor& self, aclDataType dst_type)
{
    auto outputSize = op_infer::array_to_small_vector(self.sizes());
    auto dimNum = self.dim();
    if (dst_type == aclDataType::ACL_FLOAT4_E2M1) {
        TORCH_CHECK(outputSize[dimNum - 1] % FP4_IN_UINT8_NUM == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "output dtype is torch_npu.float4_e2m1" + OPS_ERROR(ErrCode::PARAM));
        c10::SmallVector<int64_t, SIZE> output;
        output = {outputSize[0], outputSize[dimNum - 1] * outputSize[dimNum - 2] / FP4_IN_UINT8_NUM};
        return output;
    }
    TORCH_CHECK(outputSize[dimNum - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
        "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
    outputSize[dimNum - 1] /= INT4_NUMS_IN_INT32_SPACE;
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> cal_quant_scale_size(const at::Tensor& self, aclDataType dst_type)
{
    int64_t resultSize = self.size(0);
    auto outputSize = op_infer::array_to_small_vector(self.sizes());
    auto dimNum = self.dim();
    if (dst_type == aclDataType::ACL_FLOAT4_E2M1) {
        int64_t alignBase = 64UL;
        auto alignSize = (outputSize[dimNum - 1] * outputSize[dimNum - 2] + alignBase - 1) / alignBase;
        outputSize = {resultSize, alignSize, 2};
    } else {
        outputSize = {resultSize};
    }
    return outputSize;
}

::std::tuple<at::Tensor, at::Tensor> npu_kronecker_quant(const at::Tensor& x, const at::Tensor& kronecker_p1,
                                                         const at::Tensor& kronecker_p2,
                                                         c10::optional<double> clip_ratio,
                                                         c10::optional<int64_t> dst_dtype)
{
    aclDataType dst_acltype = aclDataType::ACL_INT32;
    if (dst_dtype.has_value()) {
        dst_acltype = c10_npu::GetAclDataType(dst_dtype.value());
    }
    auto out_dtype = dst_acltype == aclDataType::ACL_FLOAT4_E2M1
                         ? npu_preparation::convert_to_scalar_type(aclDataType::ACL_FLOAT4_E2M1)
                         : at::kInt;
    auto scale_dtype = dst_acltype == aclDataType::ACL_FLOAT4_E2M1
                         ? npu_preparation::convert_to_scalar_type(aclDataType::ACL_FLOAT8_E8M0)
                         : at::kFloat;
    auto clip_ratio_attr = clip_ratio.value_or(1.0);
    at::SmallVector<int64_t, SIZE> out_size = cal_out_size(x, dst_acltype);
    at::SmallVector<int64_t, SIZE> quant_scale_size = cal_quant_scale_size(x, dst_acltype);
    at::Tensor out = npu_preparation::apply_tensor_without_format(out_size, x.options().dtype(out_dtype));
    at::Tensor quant_scale =
        npu_preparation::apply_tensor_without_format(quant_scale_size, x.options().dtype(scale_dtype));
    TensorWrapper out_wrapper = {out, dst_acltype};
    TensorWrapper scale_wrapper = {quant_scale,
        dst_acltype == aclDataType::ACL_FLOAT4_E2M1? aclDataType::ACL_FLOAT8_E8M0 : aclDataType::ACL_FLOAT};
    EXEC_NPU_CMD(aclnnFlatQuant, x, kronecker_p1, kronecker_p2, clip_ratio_attr, out_wrapper, scale_wrapper);
    return std::make_tuple(std::move(out), std::move(quant_scale));
}
} // namespace op_api
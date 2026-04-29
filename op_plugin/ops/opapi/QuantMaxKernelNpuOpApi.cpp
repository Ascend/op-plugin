// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

namespace {
constexpr int64_t AMAX_SHAPE_SIZE = 1;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_quant_max(
    const at::Tensor &x, const at::Tensor &scale, c10::string_view round_mode, int64_t dst_dtype) {

    static const bool is_available = check_aclnn_kernel_available("aclnnQuantMax");
    TORCH_CHECK(is_available,
        "Current CANN version do not support this api: npu_quant_max. Please try to update the "
        "version of CANN." +
            OPS_ERROR(ErrCode::PARAM));
    // 1. 推导 aclDataType（将 torch_npu DType 枚举值 -256 得到 ACL 枚举值）
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_dtype);
    int64_t acl_dst_type = static_cast<int64_t>(y_acltype);

    // 2. 分配输出 y（shape 与 x 一致，dtype 由 dst_dtype 指定）
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    TensorWrapper y_wrapper = {y, y_acltype};

    // 3. 分配输出 amax（shape=[1]，dtype 与 x 一致）
    at::Tensor amax = npu_preparation::apply_tensor_without_format({AMAX_SHAPE_SIZE}, x.options());

    // 4. 字符串参数转 char*
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // 5. 调用 aclnn
    EXEC_NPU_CMD(aclnnQuantMax, x, scale, round_mode_ptr, acl_dst_type, y_wrapper, amax);

    return std::make_tuple(y, amax);
}

} // namespace op_api

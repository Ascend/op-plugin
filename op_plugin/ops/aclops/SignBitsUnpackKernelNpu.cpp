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

at::Tensor npu_sign_bits_unpack(
    const at::Tensor& input,
    int64_t size,
    c10::ScalarType dtype) {
    int64_t dim = input.dim();
    TORCH_CHECK(dim == 1, "input value should be a 1-dimensional tensor" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Byte, "sign_bits_unpack input only supports torch.uint8 "
        + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(size > 0, "The argument 'size' is not valid because it is less than or equal to zero"
        + OPS_ERROR(ErrCode::PARAM));

    int64_t input_size = input.numel();
    TORCH_CHECK((input_size * 8) % size == 0, "input value length*8 must be multiple of size" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dtype == at::ScalarType::Float || dtype == at::ScalarType::Half, "The argument 'dtype'  must be torch.float32 or torch.float16"
        + OPS_ERROR(ErrCode::TYPE));
    int64_t m = input_size * 8 / size;
    at::Tensor result = npu_preparation::apply_tensor({size, m}, input.options().dtype(dtype), input);

    int64_t type_enum = dtype == at::ScalarType::Half ? 1 : 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("SignBitsUnpack")
        .Input(input)
        .Output(result)
        .Attr("dtype", type_enum)
        .Attr("size", size)
        .Run();
    return result;
}

} // namespace acl_op

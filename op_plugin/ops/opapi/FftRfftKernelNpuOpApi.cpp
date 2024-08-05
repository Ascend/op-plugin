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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <iostream>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
int64_t get_norm_RFFT(c10::string_view norm)
{
    if (norm == "backward") {
        return 1;
    }
    if (norm == "forward") {
        return 2;
    }
    if (norm == "ortho") {
        return 3;
    }
    return 4; // incorrect value
}
}

at::Tensor fft_rfft(const at::Tensor &self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm)
{
    c10::string_view norm1 = norm.value_or("backward");

    if (dim < 0) {
        dim += self.dim();
    }
    
    TORCH_CHECK((dim < self.dim() || dim >= 0), "Dimension out of range (expected to be in range of [-dims, dims - 1])", OPS_ERROR(ErrCode::PARAM));
    int64_t N = self.size(dim);
    int64_t length = n.value_or(N);
    int64_t normalize = get_norm_RFFT(norm1);
    
    TORCH_CHECK(length > 0, "Invalid n value (n should be > 0)", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(normalize != 4, "Invalid normalization mode", OPS_ERROR(ErrCode::PARAM));

    at::Tensor kernel_result;
    at::Tensor resFloat;
    at::Tensor result;
    
    at::SmallVector<int64_t, op_infer::SIZE> output_shape;
    
    for (int64_t i = 0; i < dim; i++) {
        output_shape.emplace_back(self.size(i));
    }
    output_shape.emplace_back(length / 2 + 1);
    for (int64_t i = dim + 1; i < self.dim(); i++) {
        output_shape.emplace_back(self.size(i));
    }
    output_shape.emplace_back(2);
    kernel_result = npu_preparation::apply_tensor(self, output_shape);
    result = npu_preparation::apply_tensor(self, output_shape);
        
    EXEC_NPU_CMD(aclRfft1D, self, length, dim, normalize, kernel_result);
    
    resFloat = kernel_result.to(at::ScalarType::Float);
    result = at::native::view_as_complex(resFloat);
    return result;
}
}
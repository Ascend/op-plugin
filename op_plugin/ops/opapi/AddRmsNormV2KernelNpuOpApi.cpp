// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using namespace op_infer;

    at::Tensor npu_add_rms_norm_v2(at::Tensor &x1, at::Tensor &x2, const at::Tensor &gamma, double epsilon)
    {
        auto output_size_0 = rms_norm_npu_output_size(x1, gamma)[1];
        auto output_dtype_0 = at::kFloat;
        at::Tensor y = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                    x1.options().dtype(output_dtype_0));
        EXEC_NPU_CMD(aclnnInplaceAddRmsNorm, x1, x2, gamma, epsilon, y);
        return y;
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_v2_functional(const at::Tensor &x1,
                                                                                  const at::Tensor &x2,
                                                                                  const at::Tensor &gamma,
                                                                                  double epsilon)
    {
        auto output_size_0 = rms_norm_npu_output_size(x1, gamma)[1];
        auto output_dtype_0 = at::kFloat;
        at::Tensor y = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                    x1.options().dtype(output_dtype_0));
        at::Tensor x1_inplace = x1.clone();
        at::Tensor x2_inplace = x2.clone();
        EXEC_NPU_CMD(aclnnInplaceAddRmsNorm, x1_inplace, x2_inplace, gamma, epsilon, y);
        return std::tie(y, x1_inplace, x2_inplace);
    }
}
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

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_attention_update(at::TensorList lse, at::TensorList local_out, int64_t update_type)
{
    auto output_size_0 = local_out[0].sizes();
    auto output_dtype_0 = local_out[0].scalar_type();

    at::Tensor out = npu_preparation::apply_tensor_without_format(
        output_size_0,
        lse[0].options().dtype(output_dtype_0)
    );
    at::Tensor lse_out;
    EXEC_NPU_CMD(aclnnAttentionUpdate, lse, local_out, update_type, out, lse_out);

    return std::make_tuple(out, lse_out);
}
}
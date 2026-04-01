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

#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using npu_utils = at_npu::native::NpuUtils;

    bool all_contiguous(at::TensorList tensors)
    {
        for (const auto& t : tensors) {
            if (!t.is_contiguous()) {
                return false;
        }
    }
        return true;
    }

    bool isSupportedDtype(at::ScalarType dtype)
    {
        return dtype == at::ScalarType::Float ||
            dtype == at::ScalarType::Half ||  
            dtype == at::ScalarType::BFloat16;
    }
    
    bool isSupportedDtypeCombination(at::ScalarType inputDtype, at::ScalarType outputDtype)
    {
        if (!isSupportedDtype(inputDtype) || !isSupportedDtype(outputDtype)) {
            return false;
        }
        if (inputDtype == at::ScalarType::Float) {
            return outputDtype == at::ScalarType::Float;
        }
        return true;
    }

    static c10::SmallVector<int64_t, op_infer::SIZE> get_chunk_cat_out_sizes(
        at::TensorList tensors,
        int64_t dim,
        int64_t num_chunks)
    {
        auto first_tensor = tensors[0];
        auto first_sizes = first_tensor.sizes();
        int64_t input_dim_num = first_sizes.size();
        c10::SmallVector<int64_t, op_infer::SIZE> out_shape;
        out_shape.resize((dim + 1) + 1);
        
        for (int64_t j = 0; j < dim; j++) {
            out_shape[j] = first_sizes[j];
        }
        out_shape[dim] = num_chunks;
        int64_t outputCol = 0;
        for (const auto& tensor : tensors)
        {
            auto sizes = tensor.sizes();
            int64_t tensor_num_dim = sizes.size();
            int64_t chunkDimSize = sizes[dim];
            if (num_chunks>0) {
                int64_t chunkCol = (chunkDimSize + num_chunks - 1) / num_chunks;
                int64_t dim1Size = chunkCol;
                for (int64_t j = dim + 1; j < tensor_num_dim; j++) {
                    dim1Size *= sizes[j];
                }
                outputCol += dim1Size;
            }     
        }
        out_shape[dim + 1] = outputCol;
        return out_shape;
    }

    at::Tensor _chunk_cat( at::TensorList tensors, int64_t dim, int64_t num_chunks)
    {
        static bool npu_support_aclnn = check_aclnn_kernel_available("aclnnChunkCat")
                                          && c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950;
        if (!npu_support_aclnn) {
            return at::native::_chunk_cat(tensors, dim, num_chunks);
        }
        auto view_sizes = get_chunk_cat_out_sizes(tensors, dim, num_chunks);
        at::Tensor result = npu_preparation::apply_tensor_without_format(view_sizes, tensors[0].scalar_type());
        if (all_contiguous(tensors) &&
            isSupportedDtype(tensors[0].scalar_type()) &&
            dim == 0) {
            EXEC_NPU_CMD(aclnnChunkCat, tensors, dim, num_chunks, result);
        } else {
            at::native::_chunk_cat_out(tensors, dim, num_chunks, result);
        }
        return result;
    }

    at::Tensor& _chunk_cat_out( at::TensorList tensors, int64_t dim, int64_t num_chunks, at::Tensor& out)
    {
        static bool npu_support_aclnn = check_aclnn_kernel_available("aclnnChunkCat")
                                          && c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950;
        if (!npu_support_aclnn) {
            return at::native::_chunk_cat_out(tensors, dim, num_chunks, out);
        }
        TORCH_CHECK(
            tensors[0].device() == out.device(),
            "_chunk_cat_out: mismatch between input and out tensor devices");
        bool both_input_output_contiguous = all_contiguous(tensors) && out.is_contiguous();
        auto view_sizes = get_chunk_cat_out_sizes(tensors, dim, num_chunks);
        npu_preparation::check_tensor({tensors[0]}, out, at::IntArrayRef(view_sizes));
        if (both_input_output_contiguous &&
            isSupportedDtypeCombination(tensors[0].scalar_type(), out.scalar_type()) &&
            dim == 0) {
            EXEC_NPU_CMD(aclnnChunkCat, tensors, dim, num_chunks, out);
        } else {
            at::native::_chunk_cat_out(tensors, dim, num_chunks, out);
        }
        return out;
    }
}

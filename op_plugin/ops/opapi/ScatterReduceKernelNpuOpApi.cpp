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

namespace {
int64_t get_reduce(c10::string_view reduce, const char* op_name)
{
    if (reduce == "none") {
        return 0;
    } else if (reduce == "sum" || reduce == "add") {
        return 1;
    } else if (reduce == "amin" || reduce == "min") {
        return 4;
    } else if (reduce == "amax" || reduce == "max") {
        return 3;
    } else if (reduce == "prod" || reduce == "mul") {
        return 2;
    } else if (reduce == "mean") {
        return 5;
    }
    TORCH_CHECK(
        false, op_name,
        ": expected reduce to be one of none, sum, add, amin, min, amax, max, mul, prod or mean, but got ", reduce,
        OPS_ERROR(ErrCode::PARAM));
}

void exec_scatter_reduce(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self,
    at::Tensor& out,
    const char* op_name)
{
    int64_t reduction = get_reduce(reduce, op_name);
    EXEC_NPU_CMD(aclnnScatterReduce, self, dim, index, src, reduction, include_self, out);
}
} // namespace

at::Tensor scatter_reduce(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self)
{
    if (include_self && (reduce == "sum" || reduce == "add")) {
        DO_COMPATIBILITY(aclnnScatterReduce, acl_op::scatter_add(self, dim, index, src));
    }
    auto result = self.clone(at::MemoryFormat::Contiguous);
    npu_preparation::CheckMemory({result, index, src}, {result});
    exec_scatter_reduce(result, dim, index, src, reduce, include_self, result, "scatter_reduce()");
    return result;
}

at::Tensor& scatter_reduce_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self,
    at::Tensor& out)
{
    npu_preparation::CheckMemory({self, index, src}, {out});
    exec_scatter_reduce(self, dim, index, src, reduce, include_self, out, "scatter_reduce()");
    return out;
}

at::Tensor& scatter_reduce_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    bool include_self)
{
    npu_preparation::CheckMemory({self, index, src}, {self});
    if (include_self && (reduce == "sum" || reduce == "add")) {
        DO_COMPATIBILITY(aclnnInplaceScatterReduce, acl_op::scatter_add_(self, dim, index, src));
    }
    int64_t reduction = get_reduce(reduce, "scatter_reduce_()");
    EXEC_NPU_CMD(aclnnInplaceScatterReduce, self, dim, index, src, reduction, include_self);
    return self;
}


}

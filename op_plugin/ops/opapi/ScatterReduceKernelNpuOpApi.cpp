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

#include <utility>


namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
enum class ScatterReduceType : int64_t {
    REDUCE_NONE = 0,
    REDUCE_ADD = 1,
    REDUCE_MUL = 2,
    REDUCE_MAX = 3,
    REDUCE_MIN = 4,
    REDUCE_MEAN = 5,
};

const std::pair<c10::string_view, ScatterReduceType> REDUCE_TYPE_MAP[] = {
    {"none", ScatterReduceType::REDUCE_NONE},
    {"sum", ScatterReduceType::REDUCE_ADD},
    {"add", ScatterReduceType::REDUCE_ADD},
    {"prod", ScatterReduceType::REDUCE_MUL},
    {"mul", ScatterReduceType::REDUCE_MUL},
    {"amax", ScatterReduceType::REDUCE_MAX},
    {"max", ScatterReduceType::REDUCE_MAX},
    {"amin", ScatterReduceType::REDUCE_MIN},
    {"min", ScatterReduceType::REDUCE_MIN},
    {"mean", ScatterReduceType::REDUCE_MEAN},
};

int64_t get_reduce(c10::string_view reduce, const char* op_name)
{
    for (const auto& reduce_pair : REDUCE_TYPE_MAP) {
        if (reduce == reduce_pair.first) {
            return static_cast<int64_t>(reduce_pair.second);
        }
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

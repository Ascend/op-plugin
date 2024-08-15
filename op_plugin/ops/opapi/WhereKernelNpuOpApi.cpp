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

vector<at::Tensor> exec_aclnn_where(const at::Tensor &condition, at::Tensor &out)
{
    static auto opApiFuncAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr, OPS_ERROR(ErrCode::PTR));
        return ret;
    }();
    using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnNonzeroV2, condition, out);
    int64_t *view_dims = nullptr;
    uint64_t view_dim_num = 0;
    auto ret = aclGetViewShape(npuAclParams.Get<1>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret == 0, "aclGetViewShape failed.", OPS_ERROR(ErrCode::ACL));
    c10::SmallVector<int64_t, op_infer::SIZE> output_size(view_dims, view_dims + view_dim_num);
    out = out.resize_(output_size);
    delete view_dims;
    view_dims = nullptr;
    auto res = out.unbind(0);
    return res;
}

vector<at::Tensor> where(const at::Tensor &condition)
{
    DO_COMPATIBILITY(aclnnNonzeroV2, acl_op::where(condition));
    int64_t numel = condition.numel();
    int64_t dim = condition.dim();
    at::SmallVector<int64_t, op_infer::SIZE> out_size = {dim, numel};
    at::Tensor out =
        at_npu::native::OpPreparation::apply_tensor_without_format(out_size, condition.options().dtype(at::kLong));
    return exec_aclnn_where(condition, out);
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::where(condition, self, other));
  return at::_s_where(condition, self, other);
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& where_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::where_out(condition, self, other, out));

  auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);

  at::Tensor self_cp;
  at::Tensor other_cp;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_cp = npu_dtype_cast(self, result_type);
    other_cp = npu_dtype_cast(other, result_type);
  } else {
    self_cp = self;
    other_cp = other;
  }

  npu_preparation::check_tensor({condition, self_cp, other_cp}, out, out, output_size);

  EXEC_NPU_CMD(aclnnSWhere, condition, self_cp, other_cp, out);

  return out;
}

at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::where(condition, self, other));
  auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);
  at::Tensor self_cp;
  at::Tensor other_cp;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_cp = npu_dtype_cast(self, result_type);
    other_cp = npu_dtype_cast(other, result_type);
  } else {
    self_cp = self;
    other_cp = other;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(self_cp, output_size);
  EXEC_NPU_CMD(aclnnSWhere, condition, self_cp, other_cp, result);

  return result;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor& where_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::where_out(condition, self, other, out));

  auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);

  at::Tensor self_cp;
  at::Tensor other_cp;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_cp = self.to(result_type);
    other_cp = other.to(result_type);
  } else {
    self_cp = self;
    other_cp = other;
  }

  npu_preparation::check_tensor({condition, self_cp, other_cp}, out, out, output_size);

  EXEC_NPU_CMD(aclnnSWhere, condition, self_cp, other_cp, out);

  return out;
}

at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::where(condition, self, other));
  auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);
  at::Tensor self_cp;
  at::Tensor other_cp;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_cp = self.to(result_type);
    other_cp = other.to(result_type);
  } else {
    self_cp = self;
    other_cp = other;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(self_cp, output_size);
  EXEC_NPU_CMD(aclnnSWhere, condition, self_cp, other_cp, result);

  return result;
}
#endif
}
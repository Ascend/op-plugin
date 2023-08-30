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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
std::tuple<at::Tensor&, at::Tensor&> triangular_solve_out(
    const at::Tensor& self,
    const at::Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    at::Tensor& result,
    at::Tensor& clone_a) {
  at::Tensor result_tmp, clone_a_tmp;
  std::tie(result_tmp, clone_a_tmp) = triangular_solve_out_common_nocheck(self, A, upper, transpose, unitriangular);
  result.resize_as_(result_tmp).copy_(result_tmp);
  clone_a.resize_as_(clone_a_tmp).copy_(clone_a_tmp);
  return std::tie(result, clone_a);
}
} // namespace acl_op

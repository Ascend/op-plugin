// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include <string>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
namespace {
constexpr size_t ASSERT_ASYNC_MAX_MSG_LEN = 255;

void assert_async_cpu(const at::Tensor& self, c10::string_view assert_msg) {
  const auto self_cpu = self.cpu();
  if (assert_msg.empty()) {
    at::_assert_async(self_cpu);
  } else {
    at::_assert_async(self_cpu, assert_msg);
  }
}

bool is_npu_supported() {
  static const bool is_assert_async_available = check_aclnn_kernel_available("aclnnAssertAsync");
  return is_assert_async_available && is_ascend950_path();
}
} // namespace

void _assert_async(const at::Tensor& self) {
  op_api::_assert_async(self, c10::string_view{});
}

void _assert_async(const at::Tensor& self, c10::string_view assert_msg) {
  TORCH_CHECK(
      self.numel() == 1,
      "_assert_async: Expected one element, but got ",
      self.numel(),
      OPS_ERROR(ErrCode::PARAM));

  TORCH_CHECK(
      assert_msg.size() < ASSERT_ASYNC_MAX_MSG_LEN,
      "_assert_async: assert_msg length must be less than ",
      ASSERT_ASYNC_MAX_MSG_LEN,
      " bytes, but got ",
      assert_msg.size(),
      OPS_ERROR(ErrCode::PARAM));

  if (!is_npu_supported()) {
    // Either the platform is not Ascend950 with a recent enough CANN, or the
    // loaded CANN does not provide aclnnAssertAsync, so run on the host instead
    // of failing. The check then becomes synchronous and a failure raises a
    // regular RuntimeError rather than aborting the device.
    TORCH_WARN_ONCE("_assert_async falls back to CPU.");
    assert_async_cpu(self, assert_msg);
    return;
  }

  std::string msg = assert_msg.empty() ? std::string() : std::string(assert_msg.data(), assert_msg.size());
  char* msg_ptr = const_cast<char*>(msg.c_str());
  EXEC_NPU_CMD(aclnnAssertAsync, self, msg_ptr);
}
} // namespace op_api

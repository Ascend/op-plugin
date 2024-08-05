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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#define NON_FINITE_CHECK_RESULT_ESTIMATE(result, host_mem_out, device_mem_out, log_str)                                \
    do {                                                                                                               \
        if ((result) != ACL_ERROR_NONE) {                                                                              \
            TORCH_NPU_WARN_ONCE(log_str);                                                                              \
            if ((host_mem_out) != nullptr) {                                                                           \
                aclrtFreeHost(host_mem_out);                                                                           \
                host_mem_out = nullptr;                                                                                \
            }                                                                                                          \
            if ((device_mem_out) != nullptr) {                                                                         \
                aclrtFree(device_mem_out);                                                                             \
                device_mem_out = nullptr;                                                                              \
            }                                                                                                          \
            return acl_op::_amp_foreach_non_finite_check(scaled_grads);                                                \
        }                                                                                                              \
    } while (0);

const int FLOAT_STATUS_OP_DIMS_SIZE = 8;
constexpr size_t MAX_TENSOR_COUNT = 250;

static bool amp_foreach_non_finite_check(at::TensorList &scaled_grads)
{
    // apply for output host memory
    uint64_t buff_size = 64;
    void *host_mem_out = nullptr;
    void *device_mem_out = nullptr;
    aclrtStream aclStream = c10_npu::getCurrentNPUStream().stream(false);

    aclError err = aclrtMallocHost((void **)&host_mem_out, buff_size);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "malloc host memory failed!");

    err = aclrtMemset(host_mem_out, buff_size, 0, buff_size);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "set host memory failed!");

    // apply for input parameter device memory(size is 64)
    err = aclrtMalloc(&device_mem_out, buff_size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "malloc host memory failed!");

    err = aclrtGetOverflowStatus(device_mem_out, buff_size, aclStream);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "get overflow status failed!");

    err = aclrtSynchronizeStream(aclStream);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "synchronize stream failed!");

    err = aclrtMemcpy(host_mem_out, buff_size, device_mem_out, buff_size, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "memory copy failed!");

    err = aclrtResetOverflowStatus(aclStream);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "reset overflow status failed!");

    err = aclrtSynchronizeStream(aclStream);
    NON_FINITE_CHECK_RESULT_ESTIMATE(err, host_mem_out, device_mem_out, "synchronize stream failed!");

    uint64_t over_flow_flag = *(uint64_t *)host_mem_out;
    aclrtFreeHost(host_mem_out);
    host_mem_out = nullptr;
    aclrtFree(device_mem_out);
    device_mem_out = nullptr;
    return over_flow_flag != 0;
}

void _split_and_exec_npu_cmd_(at::TensorList &scaled_grads, at::Tensor &found_inf, const at::Tensor &inv_scale)
{
    size_t tensor_count = scaled_grads.size();
    size_t loop_time = tensor_count / MAX_TENSOR_COUNT; // Upward division
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_scaled_grads(scaled_grads.data() + i * MAX_TENSOR_COUNT, MAX_TENSOR_COUNT);
        EXEC_NPU_CMD(aclnnForeachNonFiniteCheckAndUnscale, temp_scaled_grads, found_inf, inv_scale);
    }
    size_t remaining_count = tensor_count % MAX_TENSOR_COUNT;
    if (remaining_count) {
        at::TensorList temp_scaled_grads(scaled_grads.data() + loop_time * MAX_TENSOR_COUNT, remaining_count);
        EXEC_NPU_CMD(aclnnForeachNonFiniteCheckAndUnscale, temp_scaled_grads, found_inf, inv_scale);
    }
}

void _amp_foreach_non_finite_check_and_unscale_(at::TensorList scaled_grads, at::Tensor &found_inf,
                                                const at::Tensor &inv_scale)
{
    TORCH_NPU_WARN_ONCE("Non finite check and unscale on NPU device!");
    TORCH_CHECK(torch_npu::utils::is_npu(inv_scale), "inv_scale must be NPU-Tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor"
        + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor"
        + OPS_ERROR(ErrCode::TYPE));

    if (scaled_grads.empty()) {
        return;
    }

    // inf/nan mode
    if (c10_npu::IsSupportInfNan()) {
        _split_and_exec_npu_cmd_(scaled_grads, found_inf, inv_scale);
        return;
    }

    // saturation mode
    // Due to the high false positive rate of rts interface, we fallback to path 3 interface to determinw finite.
    bool is_finite = !acl_op::_amp_foreach_non_finite_check(scaled_grads);
    if (!is_finite) {
        op_api::ones_out(1, found_inf);
        return;
    }

    auto expected_device = scaled_grads[0].device();
    auto expected_dtype = scaled_grads[0].dtype();
    for (auto &t : scaled_grads) {
        TORCH_CHECK(torch_npu::utils::is_npu(t), "one of scaled_grads was not a NPU tensor."
            + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(t.device() == expected_device, "scaled_grads must be on the same device."
            + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(t.dtype() == expected_dtype, "scaled_grads must have the same dtype."
            + OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor."
            + OPS_ERROR(ErrCode::PARAM));

        op_api::mul_out(t, inv_scale, const_cast<at::Tensor &>(t));
    }
}

}

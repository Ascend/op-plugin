// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include <chrono>
#include <filesystem>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"

namespace op_api {

static const int64_t KMILLISECOND_WIDTH = 3;
static const int64_t KMILLISECOND_PER_SECOND = 1000;

namespace {
struct HostFuncArgs {
    at::Tensor tensor;
    std::string path;
};
} // namespace

void write_tensor(const at::Tensor& tensor, const std::string& path)
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        std::ostringstream oss;
        oss << "Failed to open tensor save file: " << path;
        throw std::runtime_error(oss.str());
    }

    auto ivalue = torch::jit::IValue(tensor);
    auto data = torch::pickle_save(ivalue);
    ofs.write(data.data(), data.size());
    if (!ofs.good()) {
        std::ostringstream oss;
        oss << "Failed to save tensor to: " << path;
        throw std::runtime_error(oss.str());
    }

    ofs.close();
    if (!ofs) {
        std::ostringstream oss;
        oss << "Failed to close file after write: " << path;
        throw std::runtime_error(oss.str());
    }
}

void save_tensor_callback(void* args)
{
    TORCH_CHECK(args != nullptr, OPS_ERROR(ErrCode::VALUE));
    HostFuncArgs* host_func_args = static_cast<HostFuncArgs*>(args);
    TORCH_CHECK(host_func_args != nullptr, OPS_ERROR(ErrCode::VALUE));

    at::Tensor tensor = host_func_args->tensor;
    std::string path = host_func_args->path;

    auto source_sizes = tensor.sizes().vec();
    auto source_dtype = tensor.scalar_type();
    at::TensorOptions options = at::TensorOptions()
        .dtype(source_dtype)
        .device(at::kCPU)
        .memory_format(at::MemoryFormat::Contiguous);
    at::Tensor out = at::empty(source_sizes, options);
    at::Tensor tensor_contiguous = tensor.contiguous();
    const size_t nbytes = static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
    if (nbytes != 0) {
        NPU_CHECK_ERROR(aclrtMemcpy(
            out.data_ptr(),
            nbytes,
            tensor_contiguous.data_ptr(),
            nbytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
    }
    write_tensor(out, path);
}

std::string get_final_save_path(c10::optional<c10::string_view> save_path, int dev_index)
{
    std::string final_save_path;
    bool use_default_path = false;
    std::string suffix = ".pt";

    if (!save_path.has_value() || save_path.value().empty()) {
        std::string base_filename_prefix = "tensor";

        auto now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        std::tm* time_info = std::localtime(&current_time);
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() % KMILLISECOND_PER_SECOND; // in microsecond
        std::ostringstream timestamp_oss;
        timestamp_oss << std::put_time(time_info, "%Y%m%d%H%M%S")
                      << std::setw(KMILLISECOND_WIDTH)
                      << std::setfill('0')
                      << millis;

        std::filesystem::path current_path = std::filesystem::current_path();
        std::ostringstream filename_oss;
        filename_oss << base_filename_prefix << "_" << timestamp_oss.str();
        final_save_path = (current_path / filename_oss.str()).string();
    } else {
        std::string path_str(save_path.value().data(), save_path.value().size());
        std::filesystem::path file_path(path_str);
        std::filesystem::path parent_dir = file_path.parent_path();
        TORCH_CHECK(std::filesystem::exists(parent_dir) && (file_path.extension() == ".bin" ||
                    file_path.extension() == ".pt"),
                    "Invalid save path or save format: ", path_str, OPS_ERROR(ErrCode::PARAM));
        suffix = file_path.extension();
        final_save_path = (parent_dir / file_path.stem()).string();
    }

    std::ostringstream device_index_oss;
    device_index_oss << final_save_path;
    if (dev_index != -1) {
        device_index_oss << "_device_" << dev_index;
    }
    device_index_oss << suffix;
    return device_index_oss.str();
}

void save_npugraph_tensor(const at::Tensor& self, c10::optional<c10::string_view> save_path)
{
    const auto dev_type = self.device().type();
    int dev_index = -1;
    TORCH_CHECK(dev_type == at::DeviceType::PrivateUse1,
                "Only support npu tensor dump", OPS_ERROR(ErrCode::TYPE));
    dev_index = self.device().index();
    std::string final_save_path = get_final_save_path(save_path, dev_index);

    auto acl_stream = c10_npu::getCurrentNPUStream();
    HostFuncArgs* args = new HostFuncArgs();
    args->tensor = self;
    args->path = final_save_path;

    OPS_CHECK_ERROR(c10_npu::acl::AclrtLaunchHostFunc(acl_stream, save_tensor_callback, args));
}
} // namespace op_api
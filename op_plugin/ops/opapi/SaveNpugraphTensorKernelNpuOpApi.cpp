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
#include <mutex>
#include <unordered_map>
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
static std::unordered_map<std::string, std::atomic<uint64_t>> g_save_file_counters;
static std::mutex g_save_file_counter_mutex;

namespace {
struct SaveTensorHostFuncArgs {
    at::Tensor tensor;
    std::string path;
};

struct SaveTensorsHostFuncArgs {
    std::vector<at::Tensor> tensors;
    std::string save_dir;
    std::string save_name;
    std::string suffix;
    uint64_t unique_index;
};
} // namespace

uint64_t get_next_file_index(const std::string& file_key, const std::string& suffix)
{
    std::string final_key = file_key + "_" + suffix;
    std::lock_guard<std::mutex> lock(g_save_file_counter_mutex);
    auto it = g_save_file_counters.find(final_key);
    if (it == g_save_file_counters.end()) {
        it = g_save_file_counters.emplace(std::piecewise_construct,
                                          std::forward_as_tuple(final_key),
                                          std::forward_as_tuple(0)).first;
    }
    return it->second.fetch_add(1, std::memory_order_relaxed);
}

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
    SaveTensorHostFuncArgs* host_func_args = static_cast<SaveTensorHostFuncArgs*>(args);
    TORCH_CHECK(host_func_args != nullptr, OPS_ERROR(ErrCode::VALUE));

    at::Tensor tensor = host_func_args->tensor;
    std::string path = host_func_args->path;
    at::Tensor tensor_contiguous = tensor.contiguous();
    auto source_sizes = tensor_contiguous.sizes().vec();
    auto source_dtype = tensor_contiguous.scalar_type();
    at::TensorOptions options = at::TensorOptions()
        .dtype(source_dtype)
        .device(at::kCPU)
        .memory_format(at::MemoryFormat::Contiguous);
    at::Tensor out = at::empty(source_sizes, options);
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

void save_tensors_callback(void* args)
{
    TORCH_CHECK(args != nullptr, OPS_ERROR(ErrCode::VALUE));
    SaveTensorsHostFuncArgs* host_func_args = static_cast<SaveTensorsHostFuncArgs*>(args);
    TORCH_CHECK(host_func_args != nullptr, OPS_ERROR(ErrCode::VALUE));

    std::vector<at::Tensor>& tensors = host_func_args->tensors;
    std::string& save_dir = host_func_args->save_dir;
    std::string& save_name = host_func_args->save_name;
    std::string& suffix = host_func_args->suffix;
    uint64_t unique_index = host_func_args->unique_index;
    for (size_t i = 0; i < tensors.size(); ++i) {
        at::Tensor tensor_contiguous = tensors[i].contiguous();
        auto source_sizes = tensor_contiguous.sizes().vec();
        auto source_dtype = tensor_contiguous.scalar_type();
        at::TensorOptions options = at::TensorOptions()
            .dtype(source_dtype)
            .device(at::kCPU)
            .memory_format(at::MemoryFormat::Contiguous);
        at::Tensor out = at::empty(source_sizes, options);
        const size_t nbytes = static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
        if (nbytes != 0) {
            NPU_CHECK_ERROR(aclrtMemcpy(
                out.data_ptr(),
                nbytes,
                tensor_contiguous.data_ptr(),
                nbytes,
                ACL_MEMCPY_DEVICE_TO_HOST));
        }
        int dev_index = tensors[i].device().index();
        std::filesystem::path dir_path(save_dir);
        std::string final_save_path = (dir_path / save_name).string();
        std::ostringstream filename_oss;
        filename_oss << final_save_path
                     << "_" << i << "_device_" << dev_index;
        filename_oss << "_" << unique_index << suffix;
        write_tensor(out, filename_oss.str());
    }
}

std::string get_default_save_name()
{
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);
    std::tm* time_info = std::localtime(&current_time);
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % KMILLISECOND_PER_SECOND; // in millisecond
    std::ostringstream timestamp_oss;
    timestamp_oss << std::put_time(time_info, "%Y%m%d%H%M%S")
                    << std::setw(KMILLISECOND_WIDTH)
                    << std::setfill('0')
                    << millis;

    std::ostringstream filename_oss;
    filename_oss << "tensor" << "_" << timestamp_oss.str();
    return filename_oss.str();
}

std::string get_final_save_path(c10::optional<c10::string_view> save_path, int dev_index)
{
    if (!save_path.has_value() || save_path.value().empty()) {
        std::filesystem::path current_path = std::filesystem::current_path();
        std::string file_name = get_default_save_name();
        std::ostringstream filepath_oss;
        filepath_oss << current_path.string()
                     << std::filesystem::path::preferred_separator
                     << file_name << "_device_" << dev_index << ".pt";
        return filepath_oss.str();
    }

    std::string final_save_path;
    std::string path_str(save_path.value().data(), save_path.value().size());
    std::filesystem::path file_path(path_str);
    std::string suffix = file_path.extension();
    TORCH_CHECK(suffix == ".bin" || suffix == ".pt",
                "Invalid file extension, must be .pt or .bin: ",
                path_str, OPS_ERROR(ErrCode::PARAM));
    std::filesystem::path parent_dir = file_path.parent_path();
    std::string file_stem = file_path.stem().string();
    if (file_stem.empty()) {
        std::ostringstream oss;
        oss << "Invalid file path: '" << path_str << "', file name cannot be empty";
        throw std::runtime_error(oss.str());
    }
    
    if (parent_dir.empty() || parent_dir == ".") {
        std::filesystem::path current_path = std::filesystem::current_path();
        final_save_path = (current_path / file_stem).string();
    } else {
        if (!std::filesystem::exists(parent_dir)) {
            try {
                std::filesystem::create_directories(parent_dir);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Failed to create directory '" << parent_dir.string() << "': " << e.what();
                throw std::runtime_error(oss.str());
            }
        } else if (!std::filesystem::is_directory(parent_dir)) {
            std::ostringstream oss;
            oss << "Path '" << parent_dir.string() << "' exists but is not a directory";
            throw std::runtime_error(oss.str());
        }
        final_save_path = (parent_dir / file_stem).string();
    }
    std::ostringstream device_oss;
    device_oss << final_save_path << "_device_" << dev_index;
    final_save_path = device_oss.str();
    uint64_t unique_index = get_next_file_index(final_save_path, suffix);
    std::ostringstream unique_index_oss;
    unique_index_oss << final_save_path << "_" << unique_index << suffix;
    return unique_index_oss.str();
}

std::string get_final_save_name(c10::optional<c10::string_view> save_name)
{
    if (!save_name.has_value() || save_name.value().empty()) {
        return get_default_save_name();
    }
    return std::string(save_name.value().data(), save_name.value().size());
}

std::string get_final_save_dir(c10::optional<c10::string_view> save_dir)
{
    std::string final_save_dir;
    if (!save_dir.has_value() || save_dir.value().empty()) {
        final_save_dir = std::filesystem::current_path().string();
    } else {
        std::string dir_str(save_dir.value().data(), save_dir.value().size());
        std::filesystem::path dir_path(dir_str);
        if (!std::filesystem::exists(dir_path)) {
            try {
                std::filesystem::create_directories(dir_path);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Failed to create directory '" << dir_path.string() << "': " << e.what();
                throw std::runtime_error(oss.str());
            }
        } else if (!std::filesystem::is_directory(dir_path)) {
            std::ostringstream oss;
            oss << "Path '" << dir_path.string() << "' exists but is not a directory";
            throw std::runtime_error(oss.str());
        }
        final_save_dir = dir_str;
    }
    return final_save_dir;
}

std::string get_final_suffix(c10::optional<c10::string_view> suffix)
{
    std::string final_suffix;
    if (!suffix.has_value() || suffix.value().empty()) {
        final_suffix = ".pt";
    } else {
        final_suffix = std::string(suffix.value().data(), suffix.value().size());
        TORCH_CHECK(final_suffix == ".pt" || final_suffix == ".bin",
                    "Invalid file extension, must be .pt or .bin: ",
                    final_suffix, OPS_ERROR(ErrCode::PARAM));
    }
    return final_suffix;
}

void save_npugraph_tensor(const at::Tensor& input, c10::optional<c10::string_view> save_path)
{
    const auto dev_type = input.device().type();
    TORCH_CHECK(dev_type == at::DeviceType::PrivateUse1,
                "Only support npu tensor dump", OPS_ERROR(ErrCode::TYPE));
    int dev_index = input.device().index();
    std::string final_save_path = get_final_save_path(save_path, dev_index);

    auto acl_stream = c10_npu::getCurrentNPUStream();
    SaveTensorHostFuncArgs* args = new SaveTensorHostFuncArgs();
    args->tensor = input;
    args->path = final_save_path;

    OPS_CHECK_ERROR(c10_npu::acl::AclrtLaunchHostFunc(acl_stream, save_tensor_callback, args));
}

void save_npugraph_tensor(at::TensorList input,
                          c10::optional<c10::string_view> save_name,
                          c10::optional<c10::string_view> save_dir,
                          c10::optional<c10::string_view> suffix)
{
    TORCH_CHECK(input.size() > 0, "Tensor list cannot be empty", OPS_ERROR(ErrCode::TYPE));

    const auto dev_type = input[0].device().type();
    TORCH_CHECK(dev_type == at::DeviceType::PrivateUse1,
                "Only support npu tensor dump", OPS_ERROR(ErrCode::TYPE));
    for (size_t i = 1; i < input.size(); ++i) {
        TORCH_CHECK(input[i].device().type() == dev_type,
                    "All tensors must be on the same device", OPS_ERROR(ErrCode::TYPE));
    }
    std::string final_save_name = get_final_save_name(save_name);
    std::string final_save_dir = get_final_save_dir(save_dir);
    std::string final_suffix = get_final_suffix(suffix);
    int default_device_index = input[0].device().index();
    std::ostringstream file_key_oss;
    file_key_oss << final_save_dir << "/" << final_save_name << "_save_tensors_device_" << default_device_index;
    uint64_t unique_index = get_next_file_index(file_key_oss.str(), final_suffix);
    std::vector<at::Tensor> tensor_vec;
    for (const auto& tensor: input) {
        tensor_vec.push_back(tensor);
    }

    auto acl_stream = c10_npu::getCurrentNPUStream();
    SaveTensorsHostFuncArgs* args = new SaveTensorsHostFuncArgs();
    args->tensors = tensor_vec;
    args->save_dir = final_save_dir;
    args->save_name = final_save_name;
    args->suffix = final_suffix;
    args->unique_index = unique_index;

    OPS_CHECK_ERROR(c10_npu::acl::AclrtLaunchHostFunc(acl_stream, save_tensors_callback, args));
}
} // namespace op_api
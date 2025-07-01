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

#ifndef TORCHNPU_TORCH_NPU_UTILS_OP_LOG_UTILS_H_
#define TORCHNPU_TORCH_NPU_UTILS_OP_LOG_UTILS_H_

#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>

#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
namespace logging {
const static size_t SHOW_LIMIT = 20;
const static size_t REPLACE_WORKSPACE = 2;

template<typename T>
inline void append_size_and_elements(std::stringstream& ss, size_t size, const T& array)
{
    ss << " size: " << size;
    if (size > SHOW_LIMIT) {
        ss << ", first " << SHOW_LIMIT << " elements: ";
        for (size_t i = 0; i < SHOW_LIMIT; ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << array[i];
        }
    } else {
        ss << ", elements: " << array;
    }
}

// convert "x\n, weight\n" -> "x, weight\n"
inline void replace_and_append_newline(std::string& str)
{
    size_t count = std::count(str.begin(), str.end(), '\n');
    str.reserve(str.size() + count + 1);
 
    for (size_t pos = 0; (pos = str.find('\n', pos)) != std::string::npos; pos += REPLACE_WORKSPACE) {
        str.replace(pos, 1, ", ");
    }

    str += '\n';
}

// Generate basic log info
inline std::string convert_info(const at::Tensor &at_tensor)
{
    std::stringstream ss;
    if (!at_tensor.defined()) {
        ss << "Undefined Tensor!" << "\n";
        return ss.str();
    }

    if (at_tensor.dim() == 0) {
        if (torch_npu::utils::is_npu(at_tensor)) {
            ss << "NPU scalar Tensor: " << at_tensor << ", dtype: " << at_tensor.dtype();
            std::string res = ss.str();
            replace_and_append_newline(res);
            return res;
        } else {
            ss << "CPU scalar Tensor: " << at_tensor << ", dtype: " << at_tensor.dtype();
            std::string res = ss.str();
            replace_and_append_newline(res);
            return res;
        }
    }

    ss << "Tensor size: "
       << at_tensor.sizes()
       << ", stride: "
       << at_tensor.strides()
       << ", offset: "
       << at_tensor.storage_offset()
       << ", dtype: "
       << at_tensor.dtype()
       << ", device ID: "
       << static_cast<int>(at_tensor.device().index())
       << "\n";

    return ss.str();
}

inline std::string convert_info(const at::Scalar &at_scalar)
{
    std::stringstream ss;
    ss << "Scalar: " << at_scalar << ", type: " << at_scalar.type() << "\n";
    return ss.str();
}

inline std::string convert_info(const at::IntArrayRef &at_array)
{
    std::stringstream ss;
    ss << "IntArrayRef";
    append_size_and_elements(ss, at_array.size(), at_array);
    ss << "\n";
    return ss.str();
}

inline std::string convert_info(const at::ArrayRef<c10::SymInt> &int_array)
{
    std::stringstream ss;
    ss << "ArrayRef<c10::SymInt>";
    auto at_array = c10::asIntArrayRefUnchecked(int_array);
    append_size_and_elements(ss, at_array.size(), at_array);
    ss << "\n";
    return ss.str();
}

// std::array<bool, N> cannot be print derectly using operator<<.
template <std::size_t N>
inline void print_std_array(const std::array<bool, N> &value, std::stringstream& ss)
{
    ss << "[";
    size_t print_end = std::min(N, SHOW_LIMIT);
    for (size_t i = 0; i < print_end; ++i) {
        if (i > 0) ss << ", ";
        ss << value[i];
    }
    ss << "]";
}


template <std::size_t N>
inline std::string convert_info(const std::array<bool, N> &value)
{
    std::stringstream ss;
    if (N == 0) {
        ss << "Empty std::array<bool, N>";
    } else {
        ss << "std::array<bool, N> size: " << N;
        if (N > SHOW_LIMIT) {
            ss << ", first " << SHOW_LIMIT << " elements: ";
        } else {
            ss << ", elements: ";
        }
        print_std_array(value, ss);
    }
    ss << "\n";
    return ss.str();
}

inline std::string convert_info(const at::ArrayRef<bool> &value)
{
    std::stringstream ss;
    if (value.size() == 0) {
        ss << "Empty at::ArrayRef<bool>";
    } else {
        ss << "at::ArrayRef<bool>";
        append_size_and_elements(ss, value.size(), value);
    }
    ss << "\n";
    return ss.str();
}

inline std::string convert_info(const at::TensorList &at_tensor_list)
{
    std::stringstream ss;
    if (at_tensor_list.size() == 0) {
        ss << "Empty TensorList" << "\n";
        return ss.str();
    } else {
        ss << "TensorList size: " << at_tensor_list.size()
           << ", first tensor of tensorlist: " << convert_info(at_tensor_list[0]);
        return ss.str();
    }
}

inline std::string convert_info(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    std::stringstream ss;
    if (at_scalar_list.size() == 0) {
        ss << "Empty ArrayRef<at::Scalar>" << "\n";
    } else {
        ss << "ArrayRef<at::Scalar>";
        append_size_and_elements(ss, at_scalar_list.size(), at_scalar_list);
        ss << ", dtype for first element: " << at_scalar_list[0].type() << "\n";
    }
    return ss.str();
}

inline std::string convert_info(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return convert_info(opt_tensor.value());
    }

    std::stringstream ss;
    ss << "Optional None Tensor" << "\n";
    return ss.str();
}

inline std::string convert_info(const c10::optional<at::TensorList> &opt_at_tensor_list)
{
    if (opt_at_tensor_list.has_value()) {
        return convert_info(opt_at_tensor_list.value());
    }

    std::stringstream ss;
    ss << "Optional None TensorList" << "\n";
    return ss.str();
}

inline std::string convert_info(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return convert_info(opt_array.value());
    }

    std::stringstream ss;
    ss << "Optional None IntArrayRef" << "\n";
    return ss.str();
}

inline std::string convert_info(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return convert_info(opt_array.value());
    }

    std::stringstream ss;
    ss << "Optional None ArrayRef<c10::SymInt>" << "\n";
    return ss.str();
}

inline std::string convert_info(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return convert_info(opt_scalar.value());
    }

    std::stringstream ss;
    ss << "Optional None Scalar" << "\n";
    return ss.str();
}

inline std::string convert_info(const at::ScalarType scalar_type)
{
    std::stringstream ss;
    ss << "ScalarType: " << scalar_type << "\n";
    return ss.str();
}

inline std::string convert_info(int64_t int64_t_value)
{
    std::stringstream ss;
    ss << "int64_t: " << int64_t_value << "\n";
    return ss.str();
}

inline std::string convert_info(int8_t int8_t_value)
{
    std::stringstream ss;
    ss << "int8_t: " << int8_t_value << "\n";
    return ss.str();
}

inline std::string convert_info(int int_value)
{
    std::stringstream ss;
    ss << "int: " << int_value << "\n";
    return ss.str();
}

inline std::string convert_info(bool bool_value)
{
    std::stringstream ss;
    ss << "bool: " << bool_value << "\n";
    return ss.str();
}

inline std::string convert_info(float float_value)
{
    std::stringstream ss;
    ss << "float: " << float_value << "\n";
    return ss.str();
}

inline std::string convert_info(double double_value)
{
    std::stringstream ss;
    ss << "double: " << double_value << "\n";
    return ss.str();
}

inline std::string convert_info(char* char_value)
{
    std::stringstream ss;
    ss << "char*: " << char_value << "\n";
    return ss.str();
}

template <typename T> std::string convert_info(T value)
{
    std::string ss("unknown dtype, please report an issue to op-plugin.\n");
    return ss;
}

// Generate extra debug log info
inline std::string convert_debug_info(const at::Tensor &at_tensor)
{
    std::stringstream ss;
    if (!at_tensor.defined()) {
        ss << "Undefined Tensor!" << "\n";
        return ss.str();
    }

    if (torch_npu::utils::is_npu(at_tensor)) {
        auto at_tensor_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->get_npu_desc();
        if (at_tensor.dim() == 0) {
            ss << "NPU scalar Tensor: "
               << at_tensor
               << ", npu_format: "
               << at_tensor_sizes.npu_format_
               << ", base_sizes: "
               << at_tensor_sizes.base_sizes_
               << ", base_strides: "
               << at_tensor_sizes.base_strides_
               << ", storage_sizes: "
               << at_tensor_sizes.storage_sizes_;
        } else {
            // Min/Max for discontiguous tensor leads to infinite recursion of aclnnInpalceCopy
            if (!at_tensor.is_contiguous()) {
                ss << "Discontiguous tensor npu_format: "
                   << at_tensor_sizes.npu_format_
                   << ", base_sizes: "
                   << at_tensor_sizes.base_sizes_
                   << ", base_strides: "
                   << at_tensor_sizes.base_strides_
                   << ", storage_sizes: "
                   << at_tensor_sizes.storage_sizes_;
                std::string res = ss.str();
                replace_and_append_newline(res);
                return res;
            }
            // To cpu to avoid using aclnnMin/aclnnMax/aclnnMean.
            // To float to avoid problems caused by non-floating-point types, such as int.
            at::Tensor cpu_tensor;
            if (at_tensor.is_floating_point() || at_tensor.is_complex()) {
                cpu_tensor = at_tensor.cpu();
            } else {
                cpu_tensor = at_tensor.cpu().to(at::kFloat);
            }
            ss << "Tensor min: "
               << cpu_tensor.min()
               << ", max: "
               << cpu_tensor.max()
               << ", mean: "
               << cpu_tensor.mean()
               << ", npu_format: "
               << at_tensor_sizes.npu_format_
               << ", base_sizes: "
               << at_tensor_sizes.base_sizes_
               << ", base_strides: "
               << at_tensor_sizes.base_strides_
               << ", storage_sizes: "
               << at_tensor_sizes.storage_sizes_;
        }
        std::string res = ss.str();
        replace_and_append_newline(res);
        return res;
    } else {
        ss << "CPU scalar Tensor: " << at_tensor << ", dtype: " << at_tensor.dtype();
        std::string res = ss.str();
        replace_and_append_newline(res);
        return res;
    }
}

inline std::string convert_debug_info(const at::TensorList &at_tensor_list)
{
    std::stringstream ss;
    if (at_tensor_list.size() == 0) {
        ss << "No extra debug info for this param" << "\n";
        return ss.str();
    } else {
        ss << "Debug info for first tensor of tensorlist: " << convert_debug_info(at_tensor_list[0]);
        return ss.str();
    }
}

inline std::string convert_debug_info(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return convert_debug_info(opt_tensor.value());
    }

    std::stringstream ss;
    ss << "No extra debug info for this param" << "\n";
    return ss.str();
}

inline std::string convert_debug_info(const c10::optional<at::TensorList> &opt_at_tensor_list)
{
    if (opt_at_tensor_list.has_value()) {
        return convert_debug_info(opt_at_tensor_list.value());
    }

    std::stringstream ss;
    ss << "No extra debug info for this param" << "\n";
    return ss.str();
}

template <typename T> std::string convert_debug_info(T value)
{
    std::string ss("No extra debug info for this param\n");
    return ss;
}

// convert "x, weight" -> {"x: ", "weight: "}
inline std::vector<std::string> split_and_processing_args(const char* args)
{
    std::vector<std::string> result;
    std::string s(args);
    size_t pos = 0;
    size_t next;
    
    while ((next = s.find(", ", pos)) != string::npos) {
        result.push_back(s.substr(pos, next - pos) + ": ");
        pos = next + REPLACE_WORKSPACE;
    }
    result.push_back(s.substr(pos) + ": ");
    return result;
}

template <typename Tuple>
inline bool compare_length_vector_tuple(std::vector<std::string>& vec, Tuple& tup)
{
    return vec.size() == std::tuple_size<std::remove_reference_t<decltype(tup)>>::value;
}

template <size_t Index = 0, typename... T>
inline void concat_element_info_impl(const std::tuple<T...>& tpl, const std::vector<std::string>& vec, std::string& result)
{
    if constexpr (Index < sizeof...(T)) {
        result += vec[Index];
        result += std::get<Index>(tpl);
        concat_element_info_impl<Index + 1>(tpl, vec, result);
    }
}

template <typename... T>
inline std::string concat_element_info(const std::vector<std::string>& vec, const std::tuple<T...>& tpl)
{
    std::string result;
    concat_element_info_impl(tpl, vec, result);
    return result;
}

template <typename... Ts> inline std::string generate_log_infos(const char* arg_names, Ts &...args)
{
    std::vector<std::string> split_result = split_and_processing_args(arg_names);
    auto converted_info = std::make_tuple(convert_info(args)...);
    TORCH_CHECK(compare_length_vector_tuple(split_result, converted_info), "Length of arg and info are not equal!");
    std::string log_info = "\n";
    log_info += concat_element_info(split_result, converted_info);
    return log_info;
}

template <typename... Ts> inline std::string generate_debug_log_infos(const char* arg_names, Ts &...args)
{
    std::vector<std::string> split_result = split_and_processing_args(arg_names);
    auto converted_info = std::make_tuple(convert_debug_info(args)...);
    TORCH_CHECK(compare_length_vector_tuple(split_result, converted_info), "Length of arg and info are not equal!");
    std::string log_info = "Detail info:\n";
    log_info += concat_element_info(split_result, converted_info);
    return log_info;
}

}  // namespace utils
}  // namespace op_plugin

#endif //  TORCHNPU_TORCH_NPU_UTILS_OP_LOG_UTILS_H_

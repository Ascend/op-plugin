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

#include "op_api_common.h"

thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;

typedef void(*AddTensorAddrToCachedList) (void *addr);

void add_param_to_buf(const at::Tensor &at_tensor)
{
    static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
    TORCH_CHECK(addTensorAddrToCachedListAddr != nullptr, "GetOpApiFuncAddr failed.");
    AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
        reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
    if (!at_tensor.defined()) {
        MEMCPY_TO_BUF(",", 1);
        return;
    }
    // view shape
    MEMCPY_TO_BUF(at_tensor.sizes().data(), static_cast<int64_t>(at_tensor.sizes().size() * sizeof(int64_t)));
    // data type
    auto st = at_tensor.scalar_type();
    MEMCPY_TO_BUF(&st, sizeof(st));
    // seperator
    MEMCPY_TO_BUF(",", 1);
    // strides
    MEMCPY_TO_BUF(at_tensor.strides().data(), static_cast<int64_t>(at_tensor.sizes().size() * sizeof(int64_t)));
    // offset
    auto so = at_tensor.storage_offset();
    MEMCPY_TO_BUF(&so, sizeof(so));
    // storage shape
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(st);
    c10::SmallVector<int64_t, 5> storageDims;
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }
    MEMCPY_TO_BUF(storageDims.data(), static_cast<int64_t>(storageDims.size() * sizeof(int64_t)));

    addTensorAddrToCachedListFunc(const_cast<void*>(at_tensor.storage().data()));
}

void add_param_to_buf(const at::Scalar &at_scalar)
{
    at::ScalarType scalar_data_type = at_scalar.type();
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            MEMCPY_TO_BUF(&value, sizeof(double));
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            MEMCPY_TO_BUF(&value, sizeof(int64_t));
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            MEMCPY_TO_BUF(&value, sizeof(bool));
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            MEMCPY_TO_BUF(&value, sizeof(value));
            break;
        }
        default: {
            break;
        }
    }
}

void add_param_to_buf(const at::IntArrayRef &at_array)
{
    MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
}

void add_param_to_buf(const at::ArrayRef<bool> &at_array)
{
    MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(bool)));
}

void add_param_to_buf(const at::TensorList &at_tensor_list)
{
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        add_param_to_buf(at_tensor_list[i]);
    }
    auto counter = at_tensor_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        add_param_to_buf(at_scalar_list[i]);
    }
    auto counter = at_scalar_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
    MEMCPY_TO_BUF(",", 1);
}

void add_param_to_buf(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        add_param_to_buf(opt_tensor.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void add_param_to_buf(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        add_param_to_buf(opt_array.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void add_param_to_buf(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        add_param_to_buf(opt_scalar.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void add_param_to_buf(const at::ScalarType scalar_type)
{
    MEMCPY_TO_BUF(&scalar_type, sizeof(scalar_type));
}

void add_param_to_buf(const string& s)
{
    MEMCPY_TO_BUF(s.c_str(), static_cast<int64_t>(s.size()));
}

void add_param_to_buf() {}

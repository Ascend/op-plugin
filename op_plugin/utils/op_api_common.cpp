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

#include <pwd.h>
#include <sys/stat.h>
#include "op_api_common_base.h"
#include "op_api_common.h"

thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;
constexpr int g_rShift33Bits = 33;
constexpr uint64_t MIX_STEP1 = 18397679294719823053LLU;
constexpr uint64_t MIX_STEP2 = 14181476777654086739LLU;
constexpr int OWNER_ROOT_UID = 0;


typedef void(*AddTensorAddrToCachedList) (void *addr);

bool checkOwner(string cusLibPath)
{
    struct stat fileInfo;
    stat(cusLibPath.c_str(), &fileInfo);
    auto cusLibOwnerUid = fileInfo.st_uid;
    auto curOwnerUid = getuid();
    if (curOwnerUid != OWNER_ROOT_UID && cusLibOwnerUid == OWNER_ROOT_UID) {
        TORCH_NPU_WARN_ONCE("A common user is using the files of the root user.");
        return true;
    } else if ((curOwnerUid == OWNER_ROOT_UID && cusLibOwnerUid != OWNER_ROOT_UID) ||
            (curOwnerUid != OWNER_ROOT_UID && (curOwnerUid != cusLibOwnerUid))) {
        TORCH_NPU_WARN_ONCE("The ", cusLibPath,
            " owner does not match current owner or the root user is using the files of a common user, will skip this file.");
        return false;
    }
    return true;
}

static std::vector<std::string> split_str(std::string s, const std::string &del)
{
    size_t end = s.find(del);
    std::vector<std::string> path_list;
    while (end != std::string::npos) {
        path_list.push_back(s.substr(0, end));
        s.erase(s.begin(), s.begin() + end + 1);
        end = s.find(del);
    }
    path_list.push_back(s);
    return path_list;
}

static bool is_file_exist(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
}

std::string real_path(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
        return "";
    }
    return std::string(realPath);
}

std::vector<std::string> get_custom_lib_path()
{
    char *ascend_custom_opppath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    std::vector<std::string> custom_lib_path_list;

    if (ascend_custom_opppath == nullptr) {
        ASCEND_LOGW("ASCEND_CUSTOM_OPP_PATH does not exist");
        return std::vector<std::string>();
    }

    std::string ascend_custom_opppath_str(ascend_custom_opppath);
    // split string with ":"
    custom_lib_path_list = split_str(ascend_custom_opppath_str, ":");
    if (custom_lib_path_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : custom_lib_path_list) {
        it = it + "/op_api/lib/";
    }

    return custom_lib_path_list;
}

std::vector<std::string> get_default_custom_lib_path()
{
    char *ascend_opp_path = std::getenv("ASCEND_OPP_PATH");
    std::vector<std::string> default_vendors_list;

    if (ascend_opp_path == nullptr) {
        ASCEND_LOGW("ASCEND_OPP_PATH does not exist");
        return std::vector<std::string>();
    }

    std::string vendors_path(ascend_opp_path);
    vendors_path = vendors_path + "/vendors";
    std::string vendors_config_file = real_path(vendors_path + "/config.ini");
    if (vendors_config_file.empty()) {
        ASCEND_LOGW("config.ini does not exist");
        return std::vector<std::string>();
    }

    if (!is_file_exist(vendors_config_file)) {
        ASCEND_LOGW("config.ini does not exist or the path length is more than %d", PATH_MAX);
        return std::vector<std::string>();
    }

    std::ifstream ifs(vendors_config_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            break;
        }
    }
    std::string head = "load_priority=";
    line.erase(0, head.length());

    // split string with ","
    default_vendors_list = split_str(line, ",");
    if (default_vendors_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : default_vendors_list) {
        it = real_path(vendors_path + "/" + it + "/op_api/lib/");
    }

    return default_vendors_list;
}

const std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
const std::vector<std::string> g_default_custom_lib_path = get_default_custom_lib_path();

bool hasPrefix(const std::string &str, const std::string &prefix)
{
    return str.compare(0, prefix.size(), prefix) == 0;
}

bool hasSuffix(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> GetAllOpApiSoFiles()
{
    std::vector<std::string> opApiSoFiles;

    const char *ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    if (ascendHomePath == nullptr) {
        ASCEND_LOGW("ASCEND_HOME_PATH does not exist");
        return opApiSoFiles;
    }
    std::string allOpApiSoPath(ascendHomePath);
    allOpApiSoPath = allOpApiSoPath + "/lib64";
    std::string allOpApiSoRealPath = real_path(allOpApiSoPath);
    if (allOpApiSoRealPath.empty()) {
        ASCEND_LOGW("ASCEND_HOME_PATH/lib64 does not exist");
        return opApiSoFiles;
    }
    if (!is_file_exist(allOpApiSoRealPath)) {
        ASCEND_LOGW("ASCEND_HOME_PATH/lib64 does not exist or the path length is more than %d", PATH_MAX);
        return opApiSoFiles;
    }

    try {
        for (const auto& entry : std::filesystem::directory_iterator(allOpApiSoRealPath)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string fileName = entry.path().filename().string();
            if (hasPrefix(fileName, "libopapi_") && hasSuffix(fileName, ".so")) {
                ASCEND_LOGI("%s is found.", fileName.c_str());
                opApiSoFiles.push_back(fileName);
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        ASCEND_LOGW("Filesystem error: %s", e.what());
    } catch (const std::exception& e) {
        ASCEND_LOGW("Error: %s", e.what());
    }

    if (opApiSoFiles.empty()) {
        ASCEND_LOGW("ASCEND_HOME_PATH does not get any libopapi_*.so file");
    }

    return opApiSoFiles;
}

const std::vector<std::string> g_opApiSoFiles = GetAllOpApiSoFiles();

std::vector<void *> GetAllOpApiHandlers()
{
    std::vector<void *> opApiHandlers;

    for (const auto& opApiSoFile : g_opApiSoFiles) {
        auto opApiHandler = GetOpApiLibHandler(opApiSoFile.c_str());
        if (opApiHandler != nullptr) {
            ASCEND_LOGI("%s has got handler.", opApiSoFile.c_str());
        }
        opApiHandlers.push_back(opApiHandler);
    }

    if (opApiHandlers.empty()) {
        ASCEND_LOGW("ASCEND_HOME_PATH does not get any libopapi_*.so handler");
    }

    return opApiHandlers;
}

const std::vector<void *> g_opApiHandlers = GetAllOpApiHandlers();

void add_param_to_buf(const at::Tensor &at_tensor)
{
    static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
    TORCH_CHECK(addTensorAddrToCachedListAddr != nullptr, "GetOpApiFuncAddr failed.", OPS_ERROR(ErrCode::PTR));
    AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
        reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
    if (!at_tensor.defined()) {
        MEMCPY_TO_BUF(",", 1);
        return;
    }
    CheckNpuTensorValid(at_tensor);
    if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
        g_hash_offset = g_hash_buf_max_size;
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
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
            OPS_ERROR(ErrCode::PARAM));
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
            return;
        }
    }
    MEMCPY_TO_BUF(&scalar_data_type, sizeof(at::ScalarType));
}

void add_param_to_buf(const at::IntArrayRef &at_array)
{
    MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
    auto counter = at_array.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf(const at::ArrayRef<c10::SymInt> &int_array)
{
    auto at_array = c10::asIntArrayRefUnchecked(int_array);
    MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
    auto counter = at_array.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf(const at::ArrayRef<bool> &at_array)
{
    MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(bool)));
    auto counter = at_array.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
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

void add_param_to_buf(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        add_param_to_buf(opt_array.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void add_param_to_buf(const c10::OptionalIntArrayRef &opt_array)
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

void add_param_to_buf(char *c)
{
    MEMCPY_TO_BUF(c, strlen(c));
    auto counter = strlen(c);
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf(const char *c)
{
    MEMCPY_TO_BUF(c, strlen(c));
    auto counter = strlen(c);
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf(const TensorWrapper &tensor_r)
{
    static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
    TORCH_CHECK(addTensorAddrToCachedListAddr != nullptr, "GetOpApiFuncAddr failed.", OPS_ERROR(ErrCode::PTR));
    AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
        reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
    const at::Tensor &at_tensor = tensor_r.tensor_;
    if (!at_tensor.defined()) {
        MEMCPY_TO_BUF(",", 1);
        return;
    }
    CheckNpuTensorValid(at_tensor);
    if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
        g_hash_offset = g_hash_buf_max_size;
        return;
    }
    aclDataType acl_data_type = tensor_r.dtype;
    // view shape
    MEMCPY_TO_BUF(at_tensor.sizes().data(), static_cast<int64_t>(at_tensor.sizes().size() * sizeof(int64_t)));
    // data type
    MEMCPY_TO_BUF(&acl_data_type, sizeof(acl_data_type));
    // seperator
    MEMCPY_TO_BUF(",", 1);
    // strides
    MEMCPY_TO_BUF(at_tensor.strides().data(), static_cast<int64_t>(at_tensor.sizes().size() * sizeof(int64_t)));
    // offset
    auto so = at_tensor.storage_offset();
    MEMCPY_TO_BUF(&so, sizeof(so));
    // storage shape
    c10::SmallVector<int64_t, 5> storageDims;
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
            OPS_ERROR(ErrCode::PARAM));
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }
    MEMCPY_TO_BUF(storageDims.data(), static_cast<int64_t>(storageDims.size() * sizeof(int64_t)));

    addTensorAddrToCachedListFunc(const_cast<void*>(at_tensor.storage().data()));
}

void add_param_to_buf(const TensorListWrapper &tensor_list_wrapper)
{
    const at::TensorList &at_tensor_list = tensor_list_wrapper.tensor_list_;
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        add_param_to_buf(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    auto counter = at_tensor_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf() {}

void add_param_to_buf_v2(TensorStructPtr at_tensor)
{
    static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
    TORCH_CHECK(addTensorAddrToCachedListAddr != nullptr, "GetOpApiFuncAddr failed.", OPS_ERROR(ErrCode::PTR));
    AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
        reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
    if (at_tensor == nullptr) {
        MEMCPY_TO_BUF(",", 1);
        return;
    }
    // view shape
    MEMCPY_TO_BUF((*at_tensor).sizes.data(), static_cast<int64_t>((*at_tensor).sizes.size() * sizeof(int64_t)));
    // data type
    auto acl_data_type = (*at_tensor).acl_type;
    MEMCPY_TO_BUF(&acl_data_type, sizeof(acl_data_type));
    // seperator
    MEMCPY_TO_BUF(",", 1);
    // strides
    MEMCPY_TO_BUF((*at_tensor).strides.data(), static_cast<int64_t>((*at_tensor).sizes.size() * sizeof(int64_t)));
    // offset
    auto so = (*at_tensor).storage_offset;
    MEMCPY_TO_BUF(&so, sizeof(so));
    // storage shape
    c10::SmallVector<int64_t, 5> storageDims;
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.",
                    OPS_ERROR(ErrCode::PARAM));
        storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
    }
    MEMCPY_TO_BUF(storageDims.data(), static_cast<int64_t>(storageDims.size() * sizeof(int64_t)));

    addTensorAddrToCachedListFunc((*at_tensor).data_ptr);
}

void add_param_to_buf_v2(const at::Scalar &at_scalar)
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
            return;
        }
    }
    MEMCPY_TO_BUF(&scalar_data_type, sizeof(at::ScalarType));
}

void add_param_to_buf_v2(const std::vector<int64_t> &at_array)
{
    MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
    auto counter = at_array.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf_v2(const std::vector<bool> &at_array)
{
    bool *value_ptr = reinterpret_cast<bool *>(malloc(at_array.size() * sizeof(bool)));
    for (size_t i = 0; i < at_array.size(); i++) {
        value_ptr[i] = at_array[i];
    }
    MEMCPY_TO_BUF(value_ptr, static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
    free(value_ptr);

    auto counter = at_array.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf_v2(const std::vector<TensorStructPtr> &at_tensor_list)
{
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        add_param_to_buf_v2(at_tensor_list[i]);
    }
    auto counter = at_tensor_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf_v2(const std::vector<at::Scalar> &at_scalar_list)
{
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        add_param_to_buf_v2(at_scalar_list[i]);
    }
    auto counter = at_scalar_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
    MEMCPY_TO_BUF(",", 1);
}

void add_param_to_buf_v2(const c10::optional<std::vector<int64_t>> &opt_array)
{
    if (opt_array.has_value()) {
        add_param_to_buf_v2(opt_array.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void add_param_to_buf_v2(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        add_param_to_buf_v2(opt_scalar.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void add_param_to_buf_v2(const at::ScalarType scalar_type)
{
    MEMCPY_TO_BUF(&scalar_type, sizeof(scalar_type));
}

void add_param_to_buf_v2(const string& s)
{
    MEMCPY_TO_BUF(s.c_str(), static_cast<int64_t>(s.size()));
}

void add_param_to_buf_v2(char *c)
{
    MEMCPY_TO_BUF(c, strlen(c));
    auto counter = strlen(c);
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf_v2(const char *c)
{
    MEMCPY_TO_BUF(c, strlen(c));
    auto counter = strlen(c);
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void add_param_to_buf_v2()
{
}

inline uint64_t rotating_left(uint64_t x, uint8_t n)
{
    return (x << n) | (x >> (64 - n));
}

inline uint64_t mixture(uint64_t x)
{
    // constants step1(18397679294719823053) and step2(14181476777654086739) are used to allow
    // hash values to be more evenly distributed after multiplication.
    x ^= x >> g_rShift33Bits;
    x *= MIX_STEP1;
    x ^= x >> g_rShift33Bits;
    x *= MIX_STEP2;
    x ^= x >> g_rShift33Bits;

    return x;
}

// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
uint64_t gen_hash(const void *key, const int len, const uint32_t seed = 0xdeadb0d7)
{
    const uint8_t *data = (const uint8_t *)key;
    // the length of each block is 16 bytes
    const int block_num = len / 16;
    // has and hax are literal appromix to hash, and hax is the return value of this function.
    uint64_t has = seed;
    uint64_t hax = seed;

    // use 9782798678568883157 and 5545529020109919103 for blocking and obfuscation of input data
    const uint64_t c1 = 9782798678568883157LLU;
    const uint64_t c2 = 5545529020109919103LLU;

    const uint64_t *blocks = (const uint64_t *)(data);

    for (int i = 0; i < block_num; i++) {
        int even_num = 2;
        uint64_t tmp1 = blocks[i * even_num];
        uint64_t tmp2 = blocks[i * even_num + 1];

        int8_t bits_31 = 31;
        tmp1 *= c1;
        tmp1  = rotating_left(tmp1, bits_31);
        tmp1 *= c2;
        has ^= tmp1;

        int8_t bits_27 = 27;
        has = rotating_left(has, bits_27);
        has += hax;
        // increase randomness by mul by 5 and adding a constant
        has = has * 5 + 1390208809;

        int8_t bits_33 = 33;
        tmp2 *= c2;
        tmp2  = rotating_left(tmp2, bits_33);
        tmp2 *= c1;
        hax ^= tmp2;

        hax = rotating_left(hax, bits_31);
        hax += has;
        // increase randomness by mul by 5 and adding a constant
        hax = hax * 5 + 944331445;
    }

    // the length of each block is 16 bytes
    const uint8_t *tail = (const uint8_t*)(data + block_num * 16);
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    // because the size of a block is 16, different offsets are calculated for tail blocks
    // for different sizes
    switch (static_cast<uint64_t>(len) & 15) {
        case 15:
            t2 ^= ((uint64_t)tail[14]) << 48;
            [[fallthrough]];;
        case 14:
            t2 ^= ((uint64_t)tail[13]) << 40;
            [[fallthrough]];;
        case 13:
            t2 ^= ((uint64_t)tail[12]) << 32;
            [[fallthrough]];;
        case 12:
            t2 ^= ((uint64_t)tail[11]) << 24;
            [[fallthrough]];;
        case 11:
            t2 ^= ((uint64_t)tail[10]) << 16;
            [[fallthrough]];;
        case 10:
            t2 ^= ((uint64_t)tail[9]) << 8;
            [[fallthrough]];;
        case 9:
            t2 ^= ((uint64_t)tail[8]) << 0;
            t2 *= c2;
            t2 = rotating_left(t2, 33);
            t2 *= c1;
            hax ^= t2;
            [[fallthrough]];;
        case 8:
            t1 ^= ((uint64_t)tail[7]) << 56;
            [[fallthrough]];;
        case 7:
            t1 ^= ((uint64_t)tail[6]) << 48;
            [[fallthrough]];;
        case 6:
            t1 ^= ((uint64_t)tail[5]) << 40;
            [[fallthrough]];;
        case 5:
            t1 ^= ((uint64_t)tail[4]) << 32;
            [[fallthrough]];;
        case 4:
            t1 ^= ((uint64_t)tail[3]) << 24;
            [[fallthrough]];;
        case 3:
            t1 ^= ((uint64_t)tail[2]) << 16;
            [[fallthrough]];;
        case 2:
            t1 ^= ((uint64_t)tail[1]) << 8;
            [[fallthrough]];;
        case 1:
            t1 ^= ((uint64_t)tail[0]) << 0;
            t1 *= c1;
            t1 = rotating_left(t1, 31);
            t1 *= c2;
            has ^= t1;
            [[fallthrough]];;
        default:
            break;
    };

    has ^= static_cast<uint64_t>(len);
    hax ^= static_cast<uint64_t>(len);

    has += hax;
    hax += has;

    has = mixture(has);
    hax = mixture(hax);

    has += hax;
    hax += has;
    return hax;
}

uint64_t calc_hash_id()
{
    if (g_hash_offset == g_hash_buf_max_size) {
        return 0;
    }
    uint64_t hash_id = gen_hash(g_hash_buf, g_hash_offset);
    return hash_id;
}

void *GetOpApiFuncAddrFromFeatureLib(const char *api_name)
{
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_infer_handler, "libaclnn_ops_infer.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_train_handler, "libaclnn_ops_train.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(math_handler, "libaclnn_math.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(sparse_handler, "libaclnn_sparse.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(fft_handler, "libaclnn_fft.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(rand_handler, "libaclnn_rand.so", api_name);
    return nullptr;
}

bool check_aclnn_kernel_available(std::string aclnn_name)
{
    std::string workspace_name = aclnn_name + "GetWorkspaceSize";
    if (GetOpApiFuncAddr(aclnn_name.c_str()) == nullptr || GetOpApiFuncAddr(workspace_name.c_str()) == nullptr) {
        return false;
    }
    return true;
}

inline void CollectB4ShapeInfo(const at::Tensor &at_tensor,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperStride,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperShape)
{
    if (at_tensor.sizes().size() == 1) {
        wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
    } else if (at_tensor.sizes().size() > 1) {
        if (wrapperStride[at_tensor.sizes().size() - 1] == 1) {
            wrapperStride[at_tensor.sizes().size() - PENULTIMATE_DIM] =
                wrapperStride[at_tensor.sizes().size() - PENULTIMATE_DIM] * FP4_IN_INT8;
            wrapperShape[at_tensor.sizes().size() - 1] =
                wrapperShape[at_tensor.sizes().size() - 1] * FP4_IN_INT8;
        } else if (wrapperStride[at_tensor.sizes().size() - PENULTIMATE_DIM] == 1) {
            wrapperStride[at_tensor.sizes().size() - 1] =
                wrapperStride[at_tensor.sizes().size() - 1] * FP4_IN_INT8;
            wrapperShape[at_tensor.sizes().size() - PENULTIMATE_DIM] =
                wrapperShape[at_tensor.sizes().size() - PENULTIMATE_DIM] * FP4_IN_INT8;
        }

        for (auto i = 0; i < at_tensor.sizes().size() - PENULTIMATE_DIM; i++) {
            wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
        }
    } else {
        TORCH_CHECK(false, "unsupported tensor size() in 4-bit dtype.", OPS_ERROR(ErrCode::VALUE));
    }
}

void *GetOpApiFuncAddr(const char *apiName)
{
    if (!g_custom_lib_path.empty()) {
        for (auto &it : g_custom_lib_path) {
            auto cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    // check owner
                    if (!checkOwner(cust_opapi_lib)) {
                        continue;
                    }
                    ASCEND_LOGI("%s is found in %s.", apiName, cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        ASCEND_LOGI("%s is not in custom lib.", apiName);
    }

    if (!g_default_custom_lib_path.empty()) {
        for (auto &it : g_default_custom_lib_path) {
            auto default_cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (default_cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(default_cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    // check owner
                    if (!checkOwner(default_cust_opapi_lib)) {
                        continue;
                    }
                    ASCEND_LOGI("%s is found in %s.", apiName, default_cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        ASCEND_LOGI("%s is not in default custom lib.", apiName);
    }

    if (!g_opApiHandlers.empty()) {
        for (size_t i = 0; i < g_opApiHandlers.size(); ++i) {
            if (g_opApiHandlers[i] != nullptr) {
                auto funcAddr = GetOpApiFuncAddrInLib(g_opApiHandlers[i], g_opApiSoFiles[i].c_str(), apiName);
                if (funcAddr != nullptr) {
                    ASCEND_LOGI("%s is found in %s.", apiName, g_opApiSoFiles[i].c_str());
                    return funcAddr;
                }
            }
        }
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return GetOpApiFuncAddrFromFeatureLib(apiName);
}

aclTensor *ConvertType(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }
    CheckNpuTensorValid(at_tensor);
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(at_tensor)) {
        format = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            if (acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4) {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize() * FP4_IN_INT8);
                CollectB4ShapeInfo(at_tensor, wrapperStride, wrapperShape);
            } else {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
            }
        }
    }

    if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
        c10::Scalar expScalar = at_tensor.item();
        at::Tensor aclInput = at_npu::native::OpPreparation::copy_scalar_to_device(expScalar, scalar_data_type);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double:
            {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long:
            {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool:
            {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble:
            {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }

    return acl_scalar;
}

aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

aclIntArray *ConvertType(const at::ArrayRef<c10::SymInt> &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    auto array = aclCreateIntArray(int_array.data(), int_array.size());
    return array;
}

aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

aclScalarList *ConvertType(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertType(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }

    return nullptr;
}

aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

aclIntArray *ConvertType(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

aclIntArray *ConvertType(const c10::OptionalIntArrayRef &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }

    return nullptr;
}

aclDataType ConvertType(const at::ScalarType scalarType)
{
    return at_npu::native::OpPreparation::convert_to_acl_data_type(scalarType);
}

aclTensor *ConvertType(const TensorWrapper &tensor_r)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor &at_tensor = tensor_r.tensor_;

    if (!at_tensor.defined()) {
        return nullptr;
    }
    CheckNpuTensorValid(at_tensor);

    aclDataType acl_data_type = tensor_r.dtype;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(at_tensor)) {
        format = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                OPS_ERROR(ErrCode::VALUE));
            storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            if (acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4) {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize() * FP4_IN_INT8);
                CollectB4ShapeInfo(at_tensor, wrapperStride, wrapperShape);
            } else {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
            }
        }
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

aclTensorList *ConvertType(const TensorListWrapper &tensor_list_wrapper)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

aclTensor *ConvertTypeV2(TensorStructPtr at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (at_tensor == nullptr) {
        return nullptr;
    }
    aclDataType acl_data_type = (*at_tensor).acl_type;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector((*at_tensor).strides);
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector((*at_tensor).sizes);

    const auto dimNum = (*at_tensor).sizes.size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsBaseFormatType((*at_tensor).acl_format)) {
        format = (*at_tensor).acl_format;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            storageDims = (*at_tensor).storage_sizes;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            if (acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4) {
                storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize * FP4_IN_INT8);
                if ((*at_tensor).sizes.size() == 1) {
                    wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
                } else if ((*at_tensor).sizes.size() > 1 && wrapperStride[(*at_tensor).sizes.size() - 1] == 1) {
                    wrapperStride[(*at_tensor).sizes.size() - PENULTIMATE_DIM] =
                        wrapperStride[(*at_tensor).sizes.size() - PENULTIMATE_DIM] * FP4_IN_INT8;
                    for (auto i = 0; i < (*at_tensor).sizes.size() - PENULTIMATE_DIM; i++) {
                        wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
                    }
                    wrapperShape[(*at_tensor).sizes.size() - 1] =
                        wrapperShape[(*at_tensor).sizes.size() - 1] * FP4_IN_INT8;
                } else if ((*at_tensor).sizes.size() > 1 &&
                           wrapperStride[(*at_tensor).sizes.size() - PENULTIMATE_DIM] == 1) {
                    wrapperStride[(*at_tensor).sizes.size() - 1] =
                        wrapperStride[(*at_tensor).sizes.size() - 1] * FP4_IN_INT8;
                    for (auto i = 0; i < (*at_tensor).sizes.size() - PENULTIMATE_DIM; i++) {
                        wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
                    }
                    wrapperShape[(*at_tensor).sizes.size() - PENULTIMATE_DIM] =
                        wrapperShape[(*at_tensor).sizes.size() - PENULTIMATE_DIM] * FP4_IN_INT8;
                } else {
                    TORCH_CHECK(false, "unsupported tensor wrapper strides in 4-bit dtype.", OPS_ERROR(ErrCode::VALUE));
                }
            } else {
                storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
            }
        }
    }

    auto acl_tensor = aclCreateTensor(
        wrapperShape.data(), (*at_tensor).sizes.size(), acl_data_type, wrapperStride.data(),
        (*at_tensor).storage_offset, format, storageDims.data(), storageDims.size(), (*at_tensor).data_ptr);
    return acl_tensor;
}

TensorStructPtr CopyTypeV2(const at::Tensor &at_tensor)
{
    if (!at_tensor.defined()) {
        return nullptr;
    }
    CheckNpuTensorValid(at_tensor);
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(at_tensor.scalar_type());
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        acl_data_type,
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_,
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_);
}

TensorStructPtr CopyTypeV2(const TensorWrapper &tensor_r)
{
    const at::Tensor &at_tensor = tensor_r.tensor_;
    if (!at_tensor.defined()) {
        return nullptr;
    }
    CheckNpuTensorValid(at_tensor);
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        tensor_r.dtype,
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_,
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_);
}

aclScalar *ConvertTypeV2(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double:
            {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long:
            {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool:
            {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble:
            {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }

    return acl_scalar;
}

aclIntArray *ConvertTypeV2(const std::vector<int64_t> &int_list)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(int_list.data(), int_list.size());
    return array;
}

std::vector<int64_t> CopyTypeV2(const at::IntArrayRef &at_array)
{
    return at_array.vec();
}

std::vector<int64_t> CopyTypeV2(const at::ArrayRef<c10::SymInt> &at_array)
{
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    return int_array.vec();
}

aclBoolArray *ConvertTypeV2(const std::vector<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    bool *value_ptr = reinterpret_cast<bool *>(malloc(value.size() * sizeof(bool)));
    for (size_t i = 0; i < value.size(); i++) {
        value_ptr[i] = value[i];
    }
    auto array = aclCreateBoolArray(value_ptr, value.size());
    free(value_ptr);
    return array;
}

std::vector<bool> CopyTypeV2(const at::ArrayRef<bool> &value)
{
    return value.vec();
}

aclTensorList *ConvertTypeV2(const std::vector<TensorStructPtr> &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertTypeV2(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

std::vector<TensorStructPtr> CopyTypeV2(const at::TensorList &at_tensor_list)
{
    std::vector<TensorStructPtr> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = CopyTypeV2(at_tensor_list[i]);
    }
    return tensor_list;
}

std::vector<TensorStructPtr> CopyTypeV2(const TensorListWrapper &tensor_list_wrapper)
{
    std::vector<TensorStructPtr> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = CopyTypeV2(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    return tensor_list;
}

aclScalarList *ConvertTypeV2(const std::vector<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertTypeV2(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

std::vector<at::Scalar> CopyTypeV2(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    return at_scalar_list.vec();
}

TensorStructPtr CopyTypeV2(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return CopyTypeV2(opt_tensor.value());
    }

    return nullptr;
}

aclIntArray *ConvertTypeV2(const c10::optional<std::vector<int64_t>> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertTypeV2(opt_array.value());
    }

    return nullptr;
}

c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalIntArrayRef &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

aclScalar *ConvertTypeV2(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertTypeV2(opt_scalar.value());
    }

    return nullptr;
}

aclDataType ConvertTypeV2(const at::ScalarType scalarType)
{
    return at_npu::native::OpPreparation::convert_to_acl_data_type(scalarType);
}

char* ConvertTypeV2(const std::string &str)
{
    char* string_ptr = const_cast<char *>(str.c_str());
    return string_ptr;
}

std::string CopyTypeV2(char* str)
{
    std::string result = str;
    return result;
}

void MemcpyToBufImpl(const void* data, size_t size)
{
    if (g_hash_offset + size > g_hash_buf_size) {
        g_hash_offset = g_hash_buf_max_size;
        return;
    }
    memcpy(g_hash_buf + g_hash_offset, data, size);
    g_hash_offset += size;
}

bool CacheParams::GetDeterministicStatus() const
{
    return deterministic_status_;
}

uint32_t CacheParams::GetAicNum() const
{
    return aic_num_;
}

uint32_t CacheParams::GetAivNum() const
{
    return aiv_num_;
}

CacheParams GetCacheParams()
{
    CacheParams params;

    params.deterministic_status_ = at::globalContext().deterministicAlgorithms();

    if (c10_npu::is_core_control_enabled()) {
        params.aic_num_ = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
        params.aiv_num_ = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
    }

    return params;
}

void GetApiFunc(
    const char* api_name,
    const char* workspace_api_name,
    void*& opApiFuncAddr,
    void*& getWorkspaceSizeFuncAddr
)
{
    opApiFuncAddr = GetOpApiFuncAddr(api_name);
    getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspace_api_name);

    TORCH_CHECK(opApiFuncAddr != nullptr && getWorkspaceSizeFuncAddr != nullptr,
        api_name, " or ", workspace_api_name, " not in ", GetOpApiLibName(),
        ", or ", GetOpApiLibName(), " not found.",
        OPS_ERROR(ErrCode::PTR));
}

void InitExecCommonCtx()
{
    void* initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");

    InitHugeMemThreadLocal initMemFunc = initMemAddr ? reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr) : nullptr;

    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }
}

void InitExecSubTheadCtx(aclrtStream acl_stream)
{
    if (c10_npu::check_dequeue_need_use(acl_stream)) {
        c10_npu::UseStreamResInCurrentThread(acl_stream);
    }
}

void UnInitExecCommonCtx()
{
    void* unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    void* releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");

    UnInitHugeMemThreadLocal unInitMemFunc = unInitMemAddr ? reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr) : nullptr;

    if (releaseMemAddr) {
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
        releaseMemFunc(nullptr, false);
    }
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }

    UnInitCacheThreadLocal();
}

aclrtStream GetAclStream()
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    if (c10_npu::check_enqueue_need_use(acl_stream)) {
        c10_npu::UseStreamResInCurrentThread(acl_stream);
    }
    return acl_stream;
}

void SetExecConfig()
{
    at_npu::native::SetDeterministic();
}

void SetExecConfigV2(const CacheParams& cache_params)
{
    at_npu::native::SetDeterministicOps(cache_params.GetDeterministicStatus());
}

void* GetWorkSpaceAddr(
    uint64_t workspace_size)
{
    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        auto workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }
    return workspace_addr;
}

int ExecuteApiFunc(
    const void* opApiFuncAddr,
    aclrtStream acl_stream,
    uint64_t workspace_size,
    aclOpExecutor* executor
)
{
    auto workspace_addr = GetWorkSpaceAddr(workspace_size);
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
    auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);

    return api_ret;
}

bool CheckAndInitFunc(const char* aclnn_api)
{
    static const auto ptaGetExecCacheAddr = GetOpApiFuncAddr("PTAGetExecCache");
    static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
    static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");
    static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");

    PTAGetExecCache ptaGetExecCacheFunc = reinterpret_cast<PTAGetExecCache>(ptaGetExecCacheAddr);
    InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
        reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
    SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);
    CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);

    bool has_valid_funcs = (ptaGetExecCacheFunc != nullptr) &&
                           (initPTACacheThreadLocalFunc != nullptr) &&
                           (setPTAHashKeyFunc != nullptr);
    bool can_use_cache = (canUsePTACacheFunc != nullptr) && canUsePTACacheFunc(aclnn_api);

    bool check_result = has_valid_funcs && can_use_cache;

    if (!check_result) {
        return false;
    } else {
        // 执行初始化相关操作
        initPTACacheThreadLocalFunc();
        g_hash_offset = 0;
    }
    return true;
}

bool CheckAndInitFuncV2(const char* aclnn_api)
{
    static const auto ptaFindExecCacheAddr = GetOpApiFuncAddr("PTAFindExecCache");
    static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
    static const auto setPTACacheHashKeyAddr = GetOpApiFuncAddr("SetPTACacheHashKey");
    static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");

    PTAFindExecCache ptaFindExecCacheFunc = reinterpret_cast<PTAFindExecCache>(ptaFindExecCacheAddr);
    InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
        reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
    SetPTACacheHashKey setPTACacheHashKeyFunc = reinterpret_cast<SetPTACacheHashKey>(setPTACacheHashKeyAddr);
    CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);

    bool has_valid_funcs = (ptaFindExecCacheFunc != nullptr) &&
                           (initPTACacheThreadLocalFunc != nullptr) &&
                           (setPTACacheHashKeyFunc != nullptr);
    bool can_use_cache = (canUsePTACacheFunc != nullptr) && canUsePTACacheFunc(aclnn_api);

    bool check_result = has_valid_funcs && can_use_cache;

    if (!check_result) {
        return false;
    } else {
        // 执行初始化相关操作
        initPTACacheThreadLocalFunc();
        g_hash_offset = 0;
    }
    return true;
}

void AddCacheConfigParams(aclrtStream acl_stream, const CacheParams& cache_params)
{
    bool deterministic_status = cache_params.GetDeterministicStatus();
    uint32_t aic_num = cache_params.GetAicNum();
    uint32_t aiv_num = cache_params.GetAivNum();

    add_param_to_buf(deterministic_status);
    if (aic_num != UINT32_MAX && aiv_num != UINT32_MAX) {
        add_param_to_buf(aic_num);
        add_param_to_buf(aiv_num);
    }
    auto device = c10_npu::current_device();
    add_param_to_buf(device);
    add_param_to_buf(reinterpret_cast<uintptr_t>(acl_stream));
}

void AddCacheConfigParamsV2(aclrtStream acl_stream, const CacheParams& cache_params, const char* aclnn_api)
{
    bool deterministic_status = cache_params.GetDeterministicStatus();
    uint32_t aic_num = cache_params.GetAicNum();
    uint32_t aiv_num = cache_params.GetAivNum();

    add_param_to_buf_v2(deterministic_status);
    if (aic_num != UINT32_MAX && aiv_num != UINT32_MAX) {
        add_param_to_buf_v2(aic_num);
        add_param_to_buf_v2(aiv_num);
    }
    add_param_to_buf_v2(std::string(aclnn_api));
    add_param_to_buf_v2(reinterpret_cast<uintptr_t>(acl_stream));
}

aclOpExecutor* GetCacheExecutorV2(uint64_t* workspace_size)
{
    static const auto ptaFindExecCacheAddr = GetOpApiFuncAddr("PTAFindExecCache");
    static const auto setPTACacheHashKeyAddr = GetOpApiFuncAddr("SetPTACacheHashKey");

    PTAFindExecCache ptaFindExecCacheFunc = reinterpret_cast<PTAFindExecCache>(ptaFindExecCacheAddr);
    SetPTACacheHashKey setPTACacheHashKeyFunc = reinterpret_cast<SetPTACacheHashKey>(setPTACacheHashKeyAddr);

    if (g_hash_offset == g_hash_buf_max_size) {
        setPTACacheHashKeyFunc(nullptr, 0);
    } else {
        setPTACacheHashKeyFunc(reinterpret_cast<uint8_t *>(g_hash_buf), g_hash_offset);
    }
    return ptaFindExecCacheFunc(reinterpret_cast<uint8_t *>(g_hash_buf),
        g_hash_offset, workspace_size);
}

aclOpExecutor* GetCacheExecutor(uint64_t* workspace_size)
{
    uint64_t hashId = calc_hash_id();

    static const auto ptaGetExecCacheAddr = GetOpApiFuncAddr("PTAGetExecCache");
    static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");

    SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);
    PTAGetExecCache ptaGetExecCacheFunc = reinterpret_cast<PTAGetExecCache>(ptaGetExecCacheAddr);

    setPTAHashKeyFunc(hashId);

    return ptaGetExecCacheFunc(hashId, workspace_size);
}

bool ExecuteCachedOp(aclrtStream acl_stream, const char* aclnn_api, void* phrase2)
{
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = GetCacheExecutor(&workspace_size);
    if (executor == nullptr) {
        return false;
    }

    void* workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    auto acl_call = [workspace_addr, workspace_size, acl_stream, executor, phrase2]()->int {
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        NPU_CHECK_ERROR(api_ret, "call failed");
        return api_ret;
    };

    at_npu::native::OpCommand::RunOpApiV2(aclnn_api, acl_call);
    UnInitCacheThreadLocal(); // 清理缓存线程本地资源
    return true;
}

bool ExecuteCachedOpV2(aclrtStream acl_stream, const char* aclnn_api, void* phrase2, int* api_ret)
{
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = GetCacheExecutorV2(&workspace_size);
    if (executor == nullptr) {
        return false;
    }

    void *workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }

    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
    *api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    NPU_CHECK_ERROR(*api_ret, "call failed");
    UnInitCacheThreadLocal();
    return true;
}

void RunAclCall(const string &op_name, const PROC_FUNC &func)
{
    at_npu::native::OpCommand::RunOpApiV2(op_name, func);
}

TORCH_NPU_API uint32_t OpApiGetTaskQueueEnable()
{
    return c10_npu::option::OptionsManager::GetTaskQueueEnable();
}
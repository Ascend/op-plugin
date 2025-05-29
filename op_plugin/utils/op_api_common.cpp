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
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));
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
            break;
        }
    }
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
    auto st = (*at_tensor).scalar_type;
    MEMCPY_TO_BUF(&st, sizeof(st));
    // seperator
    MEMCPY_TO_BUF(",", 1);
    // strides
    MEMCPY_TO_BUF((*at_tensor).strides.data(), static_cast<int64_t>((*at_tensor).sizes.size() * sizeof(int64_t)));
    // offset
    auto so = (*at_tensor).storage_offset;
    MEMCPY_TO_BUF(&so, sizeof(so));
    // storage shape
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(st);
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
            break;
        }
    }
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
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_infer_handler, "libaclnn_ops_infer.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_train_handler, "libaclnn_ops_train.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(math_handler, "libaclnn_math.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(sparse_handler, "libaclnn_sparse.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(fft_handler, "libaclnn_fft.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(rand_handler, "libaclnn_rand.so");
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

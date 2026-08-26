// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include <unordered_map>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr int64_t NZ_16 = 16;
constexpr int64_t NZ_C0_8 = 8;
constexpr int64_t NZ_C0_16 = 16;
constexpr int64_t DIMS_1 = 1;
constexpr int64_t DIMS_2 = 2;
constexpr int64_t DIMS_3 = 3;
constexpr int64_t DIMS_4 = 4;
constexpr int64_t DIMS_MAX = 5;
constexpr int64_t IDX_0 = 0;
constexpr int64_t IDX_1 = 1;
constexpr int64_t IDX_2 = 2;
constexpr int64_t IDX_3 = 3;
constexpr int64_t MX_SCALE_LAST_DIM = 2;

struct QuantContext {
  const at::Tensor& weight;
  const at::Tensor& weight_scale;
  const c10::optional<at::Tensor>& weight_offset;
  const c10::optional<at::Tensor>& bias;
  int64_t x_dtype;
  int64_t weight_dtype;
  c10::optional<int64_t> x_scale_dtype;
  int64_t weight_scale_dtype;
  int64_t k_group_size;
  c10_npu::SocVersion soc_version;
  bool is_weight_trans = false;

  at::Tensor out_weight;
  at::Tensor out_weight_scale;
  at::Tensor out_weight_offset;
  at::Tensor out_bias;
};

using PrepareFunc = void (*)(QuantContext& ctx);
using DataFlowJudgeFunc = bool (*)(QuantContext& ctx);
using DataFlowConfig = std::pair<DataFlowJudgeFunc, std::vector<PrepareFunc>>;

inline int64_t ceil_div(int64_t a, int64_t b) {
  TORCH_CHECK(b != 0, "Division by zero in ceil_div." + OPS_ERROR(ErrCode::VALUE));
  return (a + b - 1) / b;
}

static bool is_transpose_certain_two_dims(const at::Tensor& tensor, int64_t first_dim) {
  TORCH_CHECK(
      first_dim >= 0 && first_dim + 1 < tensor.dim(),
      "first_dim out of bounds: first_dim=",
      first_dim,
      ", tensor.dim()=",
      tensor.dim(),
      OPS_ERROR(ErrCode::PARAM));
  return tensor.stride(first_dim + 1) == tensor.stride(first_dim) * tensor.size(first_dim);
}

// A16W4（INT4 / FP4 E2M1，非 MX）torch 侧统一用 uint8 载体物理打包（每字节 2 个 4-bit 元素），
// 与 aclnnWeightQuantBatchMatmulV2 的 uint8 packed 输入约定一致；其他载体直接拒绝
static void check_a16w4_uint8_carrier(const QuantContext& ctx) {
  TORCH_CHECK(
      ctx.weight.scalar_type() == at::kByte,
      "A16W4 4-bit weight must be packed into a uint8 tensor (2 elements per byte), but got ",
      ctx.weight.scalar_type(),
      OPS_ERROR(ErrCode::PARAM));
}

bool judge_mm_mx_a8w4(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
  aclDataType x_scale_acl_dtype =
      ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;
  aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);

  OP_LOG_DEBUG(
      "judge_mm_mx_a8w4: x_acl_dtype=%d, weight_acl_dtype=%d, x_scale_acl_dtype=%d, "
      "weight_scale_acl_dtype=%d, weight_dim=%d",
      static_cast<int>(x_acl_dtype),
      static_cast<int>(weight_acl_dtype),
      static_cast<int>(x_scale_acl_dtype),
      static_cast<int>(weight_scale_acl_dtype),
      static_cast<int>(ctx.weight.dim()));

  bool dtype_match = (x_acl_dtype == ACL_FLOAT8_E4M3FN) && (weight_acl_dtype == ACL_FLOAT4_E2M1) &&
      (x_scale_acl_dtype == ACL_FLOAT8_E8M0) && (weight_scale_acl_dtype == ACL_FLOAT8_E8M0);
  if (dtype_match && ctx.weight.dim() == DIMS_2) {
    TORCH_CHECK(
        ctx.weight_scale.dim() == DIMS_3,
        "Input weight scale tensor should be 3D, but got ",
        ctx.weight_scale.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        is_transpose_certain_two_dims(ctx.weight, 0),
        "Input weight tensor should be transposed, please check input.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        is_transpose_certain_two_dims(ctx.weight_scale, 0),
        "Input weight scale tensor should be transposed, please check input.",
        OPS_ERROR(ErrCode::PARAM));
    ctx.is_weight_trans = true;
    return true;
  }
  return false;
}

bool judge_gmm_mx_a8w4(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
  aclDataType x_scale_acl_dtype =
      ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;
  aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);

  OP_LOG_DEBUG(
      "judge_gmm_mx_a8w4: x_acl_dtype=%d, weight_acl_dtype=%d, x_scale_acl_dtype=%d, "
      "weight_scale_acl_dtype=%d, weight_dim=%d",
      static_cast<int>(x_acl_dtype),
      static_cast<int>(weight_acl_dtype),
      static_cast<int>(x_scale_acl_dtype),
      static_cast<int>(weight_scale_acl_dtype),
      static_cast<int>(ctx.weight.dim()));

  bool dtype_match = (x_acl_dtype == ACL_FLOAT8_E4M3FN) && (weight_acl_dtype == ACL_FLOAT4_E2M1) &&
      (x_scale_acl_dtype == ACL_FLOAT8_E8M0) && (weight_scale_acl_dtype == ACL_FLOAT8_E8M0);
  if (dtype_match && ctx.weight.dim() == DIMS_3) {
    TORCH_CHECK(
        ctx.weight_scale.dim() == DIMS_4,
        "Input weight scale tensor should be 4D, but got ",
        ctx.weight_scale.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        is_transpose_certain_two_dims(ctx.weight, 1),
        "Input weight tensor should be transposed, please check input.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        is_transpose_certain_two_dims(ctx.weight_scale, 1),
        "Input weight scale tensor should be transposed, please check input.",
        OPS_ERROR(ErrCode::PARAM));
    ctx.is_weight_trans = true;
    return true;
  }
  return false;
}

bool judge_mm_a16s4_per_tensor(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);

  bool x_dtype_match = (x_acl_dtype == ACL_FLOAT16 || x_acl_dtype == ACL_BF16);

  // A16S4 per-tensor：scale 仅含单个元素（{1}/{1,1}），不支持 NZ，一律 ND 直拷（转置/非转置均支持，
  // ND 直拷为物理透传，与转置状态无关）
  if (x_dtype_match && weight_acl_dtype == ACL_INT4 && ctx.weight.dim() == DIMS_2 &&
      ctx.weight_scale.numel() == 1) {
    check_a16w4_uint8_carrier(ctx);
    ctx.is_weight_trans = is_transpose_certain_two_dims(ctx.weight, 0);
    return true;
  }
  return false;
}

bool judge_mm_a16s4_per_channel(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);

  OP_LOG_DEBUG(
      "judge_mm_a16s4_per_channel: x_acl_dtype=%d, weight_acl_dtype=%d, "
      "weight_dim=%d, weight_stride=[%d,%d], weight_size=[%d,%d]",
      static_cast<int>(x_acl_dtype),
      static_cast<int>(weight_acl_dtype),
      static_cast<int>(ctx.weight.dim()),
      ctx.weight.dim() >= 2 ? static_cast<int>(ctx.weight.stride(0)) : -1,
      ctx.weight.dim() >= 2 ? static_cast<int>(ctx.weight.stride(1)) : -1,
      ctx.weight.dim() >= 2 ? static_cast<int>(ctx.weight.size(0)) : -1,
      ctx.weight.dim() >= 2 ? static_cast<int>(ctx.weight.size(1)) : -1);

  bool x_dtype_match = (x_acl_dtype == ACL_FLOAT16 || x_acl_dtype == ACL_BF16);

  if (x_dtype_match && weight_acl_dtype == ACL_INT4 && ctx.weight.dim() == DIMS_2) {
    int64_t scale_dim = ctx.weight_scale.dim();
    bool is_per_channel = (scale_dim == DIMS_1) || (scale_dim == DIMS_2 && ctx.weight_scale.size(0) == 1);
    // 转置状态内部分流：转置 → ND 直拷，非转置 → NZ 转换（prepare_out_weight_a16s4）
    if (is_per_channel && ctx.weight_scale.numel() > 1) {
      check_a16w4_uint8_carrier(ctx);
      ctx.is_weight_trans = is_transpose_certain_two_dims(ctx.weight, 0);
      return true;
    }
  }
  return false;
}

bool judge_mm_a16s4_per_group(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
  aclDataType x_scale_acl_dtype =
      ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;

  bool x_dtype_match = (x_acl_dtype == ACL_FLOAT16 || x_acl_dtype == ACL_BF16);

  // A16S4 per-group：转置状态内部分流，转置 → ND 直拷，非转置 → NZ 转换（prepare_out_weight_a16s4）
  if (x_dtype_match && weight_acl_dtype == ACL_INT4 && x_scale_acl_dtype == ACL_DT_UNDEFINED &&
      ctx.weight.dim() == DIMS_2 && ctx.weight_scale.dim() == DIMS_2 && ctx.weight_scale.size(0) > 1) {
    check_a16w4_uint8_carrier(ctx);
    ctx.is_weight_trans = is_transpose_certain_two_dims(ctx.weight, 0);
    return true;
  }
  return false;
}

bool judge_mm_a16f4_nz_pergroup(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
  aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);
  aclDataType x_scale_acl_dtype =
      ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;

  bool x_dtype_match = (x_acl_dtype == ACL_FLOAT16 || x_acl_dtype == ACL_BF16);
  bool scale_dtype_match = (weight_scale_acl_dtype == ACL_FLOAT16 || weight_scale_acl_dtype == ACL_BF16);

  // A16F4 per-group NZ：FP4 weight + per-group scale [G, N]（G > 1），仅支持非转置 weight
  if (x_dtype_match && scale_dtype_match && weight_acl_dtype == ACL_FLOAT4_E2M1 &&
      x_scale_acl_dtype == ACL_DT_UNDEFINED && ctx.weight.dim() == DIMS_2 && ctx.weight_scale.dim() == DIMS_2 &&
      ctx.weight_scale.size(0) > 1 && !is_transpose_certain_two_dims(ctx.weight, 0)) {
    check_a16w4_uint8_carrier(ctx);
    ctx.is_weight_trans = false;
    return true;
  }
  return false;
}

bool judge_mm_a16f4_mx(QuantContext& ctx) {
  aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
  aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
  aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);
  aclDataType x_scale_acl_dtype =
      ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;

  bool x_dtype_match = (x_acl_dtype == ACL_FLOAT16 || x_acl_dtype == ACL_BF16);

  // A16 MXFP4：FP4 weight + MX scale（E8M0，2D [K/32, N] 连续），仅支持非转置 weight
  if (x_dtype_match && weight_acl_dtype == ACL_FLOAT4_E2M1 && weight_scale_acl_dtype == ACL_FLOAT8_E8M0 &&
      x_scale_acl_dtype == ACL_DT_UNDEFINED && ctx.weight.dim() == DIMS_2 && ctx.weight_scale.dim() == DIMS_2 &&
      !is_transpose_certain_two_dims(ctx.weight, 0)) {
    check_a16w4_uint8_carrier(ctx);
    ctx.is_weight_trans = false;
    return true;
  }
  return false;
}

// 校验镜像 strides 的寻址包络不超出按 numel 最小分配的 buffer；
// 连续/确定转置视图必过，padding/重叠/expand 视图直接拒绝而不是写出界
static void check_strides_envelope(c10::IntArrayRef sizes, c10::IntArrayRef strides, int64_t numel) {
  int64_t max_offset = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    TORCH_CHECK(strides[i] > 0, "expanded/overlapped view is not supported here", OPS_ERROR(ErrCode::PARAM));
    max_offset += (sizes[i] - 1) * strides[i];
  }
  TORCH_CHECK(max_offset < numel, "mirrored strides exceed output allocation: max offset ", max_offset,
              " vs numel ", numel, OPS_ERROR(ErrCode::PARAM));
}

static void prepare_out_weight_scale(QuantContext& ctx) {
  auto scale_view_shape = op_infer::array_to_small_vector(ctx.weight_scale.sizes());
  ctx.out_weight_scale = npu_preparation::apply_tensor_without_format(scale_view_shape, ctx.weight_scale.options());
  check_strides_envelope(ctx.weight_scale.sizes(), ctx.weight_scale.strides(), ctx.out_weight_scale.numel());
  // 保持与输入 scale 相同的 strides（ND per-group 转置场景需配转置 scale），连续输入时为空操作
  ctx.out_weight_scale =
      ctx.out_weight_scale.as_strided_(scale_view_shape, op_infer::array_to_small_vector(ctx.weight_scale.strides()));
}

static void prepare_out_weight_nd(QuantContext& ctx) {
  auto weight_view_shape = op_infer::array_to_small_vector(ctx.weight.sizes());
  ctx.out_weight = npu_preparation::apply_tensor_without_format(weight_view_shape, ctx.weight.options());
  check_strides_envelope(ctx.weight.sizes(), ctx.weight.strides(), ctx.out_weight.numel());
  // ND 直拷是物理透传，out_weight 必须与 weight 保持相同的 sizes/strides（转置场景尤为关键）
  ctx.out_weight = ctx.out_weight.as_strided_(weight_view_shape, op_infer::array_to_small_vector(ctx.weight.strides()));
}

template <bool IsGmm, int64_t NzC0, aclFormat OutWeightFormat>
static void prepare_out_weight_nz(QuantContext& ctx) {
  auto weight_sizes = ctx.weight.sizes();
  int64_t weight_dim = ctx.weight.dim();
  int64_t k = weight_sizes[weight_dim - IDX_2];
  int64_t n = weight_sizes[weight_dim - IDX_1];

  c10::SmallVector<int64_t, DIMS_MAX> storage_shape;
  c10::SmallVector<int64_t, DIMS_MAX> trans_stride;

  if (ctx.is_weight_trans) {
    if constexpr (IsGmm) {
      int64_t group_num = weight_sizes[IDX_0];
      storage_shape = {group_num, ceil_div(k, NzC0), ceil_div(n, NZ_16), NZ_16, NzC0};
      trans_stride = {k * n, 1, k};
    } else {
      storage_shape = {ceil_div(k, NzC0), ceil_div(n, NZ_16), NZ_16, NzC0};
      trans_stride = {1, k};
    }
  } else {
    if constexpr (IsGmm) {
      int64_t group_num = weight_sizes[IDX_0];
      storage_shape = {group_num, ceil_div(n, NzC0), ceil_div(k, NZ_16), NZ_16, NzC0};
    } else {
      storage_shape = {ceil_div(n, NzC0), ceil_div(k, NZ_16), NZ_16, NzC0};
    }
  }

  int64_t storage_size = 1;
  for (auto dim : storage_shape) {
    storage_size *= dim;
  }

  ctx.out_weight = npu_preparation::apply_tensor_without_format({storage_size}, ctx.weight.options());

  auto weight_view_shape = op_infer::array_to_small_vector(ctx.weight.sizes());

  if (ctx.is_weight_trans) {
    ctx.out_weight = ctx.out_weight.as_strided_(weight_view_shape, trans_stride);
  } else {
    ctx.out_weight.unsafeGetTensorImpl()->set_sizes_contiguous(weight_view_shape);
  }

  at_npu::native::StorageDescHelper::SetDesc(
      ctx.out_weight,
      ctx.out_weight.sizes(),
      storage_shape,
      ctx.out_weight.strides(),
      static_cast<aclFormat>(OutWeightFormat));
}

// A16W4（INT4/FP4）NZ（非转置）专用 prepare：torch 侧 uint8 [K, N/2] 非转置输入
// out_weight: torch 侧保持 uint8 物理打包视图 [K, N/2] + NZ_C0_8 标签（C0=8 字节=16 个 4-bit），
// storage 为物理分形 [N/16, K/16, 16, 8]（uint8 字节数精确）；经 ConvertType 统一 ×2 还原为
// ACL 4-bit view [K, N]、storage [N/16, K/16, 16, 16]、format NZ_C0_16，与 ops-math 校验一致。
static void prepare_out_weight_nz_a16w4(QuantContext& ctx) {
  auto weight_sizes = ctx.weight.sizes();
  int64_t weight_dim = ctx.weight.dim();
  int64_t k = weight_sizes[weight_dim - IDX_2];
  int64_t n_packed = weight_sizes[weight_dim - IDX_1]; // uint8 物理打包字节数（N/2）
  int64_t n_logical = n_packed * 2; // 每字节 2 个 INT4

  c10::SmallVector<int64_t, DIMS_MAX> storage_shape = {ceil_div(n_logical, NZ_16), ceil_div(k, NZ_16), NZ_16, NZ_C0_8};

  int64_t storage_size = 1;
  for (auto dim : storage_shape) {
    storage_size *= dim;
  }

  ctx.out_weight = npu_preparation::apply_tensor_without_format({storage_size}, ctx.weight.options());

  // 物理打包视图 [K, N/2]：4-bit 的 dtype/format 语义由 ConvertType 与 fake format 标签承担
  c10::SmallVector<int64_t, DIMS_MAX> out_view_shape = {k, n_packed};
  ctx.out_weight.unsafeGetTensorImpl()->set_sizes_contiguous(out_view_shape);

  at_npu::native::StorageDescHelper::SetDesc(
      ctx.out_weight, ctx.out_weight.sizes(), storage_shape, ctx.out_weight.strides(), ACL_FORMAT_FRACTAL_NZ_C0_8);
}

// A16S4 内部分流：转置 weight → ND 直拷（物理透传，与转置状态无关）；非转置 → NZ 转换
static void prepare_out_weight_a16s4(QuantContext& ctx) {
  if (ctx.is_weight_trans) {
    prepare_out_weight_nd(ctx);
  } else {
    prepare_out_weight_nz_a16w4(ctx);
  }
}

template <bool IsGmm>
static void prepare_out_weight_scale_mx(QuantContext& ctx) {
  constexpr int64_t expected_dim = IsGmm ? DIMS_4 : DIMS_3;
  TORCH_CHECK(
      ctx.weight_scale.dim() == expected_dim,
      "Input weight scale tensor should be ",
      expected_dim,
      "D, but got ",
      ctx.weight_scale.dim(),
      OPS_ERROR(ErrCode::PARAM));

  auto scale_view_shape = op_infer::array_to_small_vector(ctx.weight_scale.sizes());
  ctx.out_weight_scale = npu_preparation::apply_tensor_without_format(scale_view_shape, ctx.weight_scale.options());

  if (ctx.is_weight_trans) {
    int64_t scale_dim = ctx.weight_scale.dim();
    int64_t k_scale = scale_view_shape[scale_dim - IDX_3];
    int64_t n = scale_view_shape[scale_dim - IDX_2];
    c10::SmallVector<int64_t, DIMS_MAX> trans_stride; // 对应内存转置存储
    if constexpr (IsGmm) {
      trans_stride = {
          MX_SCALE_LAST_DIM * k_scale * n,
          MX_SCALE_LAST_DIM,
          MX_SCALE_LAST_DIM * k_scale,
          1}; // 对应内存中的 [group_num, n, k_scale, 2]
    } else {
      trans_stride = {MX_SCALE_LAST_DIM, MX_SCALE_LAST_DIM * k_scale, 1}; // 对应内存中的 [n, k_scale, 2]
    }
    ctx.out_weight_scale = ctx.out_weight_scale.as_strided_(scale_view_shape, trans_stride);
  }
}

static void prepare_out_weight_offset(QuantContext& ctx) {
  if (ctx.weight_offset.has_value() && ctx.weight_offset.value().defined()) {
    auto offset_sizes = op_infer::array_to_small_vector(ctx.weight_offset.value().sizes());
    ctx.out_weight_offset =
        npu_preparation::apply_tensor_without_format(offset_sizes, ctx.weight_offset.value().options());
    check_strides_envelope(
        ctx.weight_offset.value().sizes(), ctx.weight_offset.value().strides(), ctx.out_weight_offset.numel());
    // 与 prepare_out_weight_scale 同理：保持与输入 offset 相同的 strides，
    // 转置场景下游 wqbmmv2 要求 antiquantOffset 与 weight 连续/转置状态一致，连续输入时为空操作
    ctx.out_weight_offset = ctx.out_weight_offset.as_strided_(
        offset_sizes, op_infer::array_to_small_vector(ctx.weight_offset.value().strides()));
  }
}

static void prepare_out_bias(QuantContext& ctx) {
  if (ctx.bias.has_value() && ctx.bias.value().defined()) {
    auto bias_sizes = op_infer::array_to_small_vector(ctx.bias.value().sizes());
    ctx.out_bias = npu_preparation::apply_tensor_without_format(bias_sizes, ctx.bias.value().options());
  }
}

static void execute_prepares(const std::vector<PrepareFunc>& prepares, QuantContext& ctx) {
  for (const auto& prepare : prepares) {
    prepare(ctx);
  }
}

static const std::unordered_map<c10_npu::SocVersion, std::vector<DataFlowConfig>> SOC_DATA_FLOW_CONFIG_MAP = {
    {c10_npu::SocVersion::Ascend950,
     {{judge_mm_mx_a8w4,
       {// FP4 逻辑 C0 为 32，torch 层用 1 个 int8/uint8 元素打包 2 个 FP4，因此物理存储的 C0 维长度为 16
        prepare_out_weight_nz<false, NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_16>,
        prepare_out_weight_scale_mx<false>,
        prepare_out_weight_offset,
        prepare_out_bias}},
      {judge_gmm_mx_a8w4,
       {// FP4 逻辑 C0 为 32，torch 层用 1 个 int8/uint8 元素打包 2 个 FP4，因此物理存储的 C0 维长度为 16
        prepare_out_weight_nz<true, NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_16>,
        prepare_out_weight_scale_mx<true>,
        prepare_out_weight_offset,
        prepare_out_bias}},
      {judge_mm_a16s4_per_tensor,
       {prepare_out_weight_nd,
        prepare_out_weight_scale,
        prepare_out_weight_offset,
        prepare_out_bias}},
      {judge_mm_a16s4_per_channel,
       {prepare_out_weight_a16s4,
        prepare_out_weight_scale,
        prepare_out_weight_offset,
        prepare_out_bias}},
      {judge_mm_a16s4_per_group,
       {prepare_out_weight_a16s4,
        prepare_out_weight_scale,
        prepare_out_weight_offset,
        prepare_out_bias}},
      {judge_mm_a16f4_nz_pergroup,
       {prepare_out_weight_nz_a16w4,
        prepare_out_weight_scale,
        prepare_out_weight_offset,
        prepare_out_bias}},
      {judge_mm_a16f4_mx,
       {prepare_out_weight_nz_a16w4,
        prepare_out_weight_scale,
        prepare_out_weight_offset,
        prepare_out_bias}}}},
};

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_weight_quant_preprocess(
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    int64_t x_dtype,
    int64_t weight_dtype,
    int64_t weight_scale_dtype,
    const c10::optional<at::Tensor>& weight_offset,
    const c10::optional<at::Tensor>& bias,
    c10::optional<int64_t> x_scale_dtype,
    c10::optional<int64_t> k_group_size) {
  auto soc_version = c10_npu::GetSocVersion();
  auto it = SOC_DATA_FLOW_CONFIG_MAP.find(soc_version);
  TORCH_CHECK(
      it != SOC_DATA_FLOW_CONFIG_MAP.end(),
      "Unsupported NPU architecture: " + std::to_string(static_cast<int>(soc_version)) +
          OPS_ERROR(ErrCode::NOT_SUPPORT));

  int64_t k_group_size_real = k_group_size.value_or(0);

  QuantContext ctx{
      weight,
      weight_scale,
      weight_offset,
      bias,
      x_dtype,
      weight_dtype,
      x_scale_dtype,
      weight_scale_dtype,
      k_group_size_real,
      soc_version,
      false,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};

  bool matched = false;
  for (const auto& [judge, prepares] : it->second) {
    if (judge(ctx)) {
      execute_prepares(prepares, ctx);
      matched = true;
      break;
    }
  }
  TORCH_CHECK(matched, "Unsupported data flow combination." + OPS_ERROR(ErrCode::PARAM));

  TensorWrapper weight_wrapper = make_wrapper(ctx.weight, c10::optional<int64_t>(ctx.weight_dtype));
  TensorWrapper scale_wrapper = make_wrapper(ctx.weight_scale, c10::optional<int64_t>(ctx.weight_scale_dtype));
  TensorWrapper out_weight_wrapper = make_wrapper(ctx.out_weight, c10::optional<int64_t>(ctx.weight_dtype));
  TensorWrapper out_scale_wrapper = make_wrapper(ctx.out_weight_scale, c10::optional<int64_t>(ctx.weight_scale_dtype));

  aclDataType x_acl_type = c10_npu::GetAclDataType(x_dtype);
  aclDataType x_scale_acl_type =
      x_scale_dtype.has_value() ? c10_npu::GetAclDataType(x_scale_dtype.value()) : ACL_DT_UNDEFINED;

  at::Tensor weight_offset_tensor = weight_offset.value_or(at::Tensor());
  at::Tensor bias_tensor = bias.value_or(at::Tensor());
  EXEC_NPU_CMD(
      aclnnWeightQuantPreprocess,
      weight_wrapper,
      scale_wrapper,
      weight_offset_tensor,
      bias_tensor,
      x_acl_type,
      x_scale_acl_type,
      ctx.k_group_size,
      out_weight_wrapper,
      out_scale_wrapper,
      ctx.out_weight_offset,
      ctx.out_bias);

  return std::make_tuple(ctx.out_weight, ctx.out_weight_scale, ctx.out_weight_offset, ctx.out_bias);
}

} // namespace op_api

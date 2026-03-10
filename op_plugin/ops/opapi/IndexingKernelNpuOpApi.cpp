// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
static constexpr int32_t K_STRIDED_SLICE_NEW_AXIS = -2;
static constexpr int32_t K_SHRINK_AXIS = -1;
static constexpr int64_t INVALID_IDX = -3L;

struct StridedSliceParams {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    uint64_t begin_mask;
    uint64_t end_mask;
    uint64_t ellipsis_mask;
    uint64_t new_axis_mask;
    uint64_t shrink_axis_mask;
    bool begin_valid;
    bool end_valid;
    bool stride_valid;
    bool real_begin_valid = true;
    bool real_end_valid = true;
    std::vector<int64_t> dy_shape;

    [[nodiscard]] std::string to_string() const
    {
        std::string result = "input_shape:" + op_plugin::utils::get_vector_str(input_shape);
        result += " begin:" + op_plugin::utils::get_vector_str(begin);
        result += " end:" + op_plugin::utils::get_vector_str(end);
        result += " strides:" + op_plugin::utils::get_vector_str(strides);
        result += " begin_mask:" + std::to_string(begin_mask);
        result += " end_mask:" + std::to_string(end_mask);
        result += " ellipsis_mask:" + std::to_string(ellipsis_mask);
        result += " new_axis_mask:" + std::to_string(new_axis_mask);
        result += " shrink_axis_mask:" + std::to_string(shrink_axis_mask);
        result += " begin_valid:" + std::to_string(begin_valid);
        result += " end_valid:" + std::to_string(end_valid);
        result += " stride_valid:" + std::to_string(stride_valid);
        result += " real_begin_valid:" + std::to_string(static_cast<int32_t>(real_begin_valid));
        result += " real_end_valid:" + std::to_string(static_cast<int32_t>(real_end_valid));
        return result;
    }
};

struct ProcessingData {
    std::vector<int64_t> processing_shape;
    std::vector<int64_t> processing_begin;
    std::vector<int64_t> processing_end;
    std::vector<int64_t> processing_strides;

    [[nodiscard]] std::string to_string() const
    {
        std::string result = "processing_shape:" + op_plugin::utils::get_vector_str(processing_shape);
        result += " processing_begin:" + op_plugin::utils::get_vector_str(processing_begin);
        result += " processing_end:" + op_plugin::utils::get_vector_str(processing_end);
        result += " processing_strides:" + op_plugin::utils::get_vector_str(processing_strides);
        return result;
    }
};

struct InputParamUnit {
    int64_t begin;
    int64_t end;
    int64_t stride;
    int64_t dim;
    bool shrink;
};

struct StridedSliceSparseSpec {
    int64_t dims;
    int32_t num_add_axis_after_ellipsis;
    const std::vector<int64_t> begin;
    const std::vector<int64_t> end;
    const std::vector<int64_t> strides;
    const uint64_t begin_mask;
    const uint64_t end_mask;
    uint64_t ellipsis_mask;
    const uint64_t new_axis_mask;
    const uint64_t shrink_axis_mask;
};

struct StridedSliceDenseSpec {
    const int64_t dims;
    uint64_t begin_mask;
    uint64_t end_mask;
    bool begin_valid;
    bool end_valid;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;

    // This vector helps construct the final shape of the slice.
    // The final tensor is reduced in rank whenever a single index e.g. foo[3]
    // is called for. The final tensor increases in rank with tf.newaxis
    // entries. If an index in this array is positive, the size of the dimension
    // is obtained from canonical end-begin. Otherwise, if it is a kNewAxis,
    // it will be 1. A shrunk dimension is skipped.
    std::vector<int64_t> final_shape_gather_indices;

    // The dense indexed shrink mask is which processing dimensions
    // should be shrunk. For example, if foo.shape = (10,10,10,10)
    // foo[3, ..., 5] has sparse_shrink_axis_mask of 0x5 and
    // dense_shrink_axis_mask of 0x9, yielding a final shape (10,10).
    uint64_t shrink_axis_mask;

    [[nodiscard]] std::string to_string() const
    {
        std::string result = "dims:" + std::to_string(dims);
        result += " begin_mask:" + std::to_string(begin_mask);
        result += " end_mask:" + std::to_string(end_mask);
        result += " begin_valid:" + std::to_string(static_cast<int32_t>(begin_valid));
        result += " end_valid:" + std::to_string(static_cast<int32_t>(end_valid));
        result += " begin:" + op_plugin::utils::get_vector_str(begin);
        result += " end:" + op_plugin::utils::get_vector_str(end);
        result += " strides:" + op_plugin::utils::get_vector_str(strides);
        result += " final_shape_gather_indices:" + op_plugin::utils::get_vector_str(final_shape_gather_indices);
        result += " shrink_axis_mask:" + std::to_string(shrink_axis_mask);
        return result;
    }
};

/// 计算第i位的bit值（2^i）
static inline uint64_t bit_1_value(int i)
{
    const uint64_t bit_i = static_cast<uint64_t>(1) << static_cast<uint64_t>(i);
    return bit_i;
}

/// 规范化索引，将负索引转换为正索引
static inline int64_t normalize_index(int64_t x, int64_t dim)
{
    return x < 0 ? dim + x : x;
}

/// 检查索引是否越界
static inline bool fwd_out_of_bound(int64_t fwd, int64_t lower, int64_t upper)
{
    return (fwd < lower) || (fwd >= upper);
}

/// 构建稀疏规格，处理省略号(ellipsis)和省略号后的新轴数量
static void build_sparse_spec(const StridedSliceParams& params, StridedSliceSparseSpec& sparse_spec)
{
    sparse_spec.dims = static_cast<int64_t>(params.strides.size());
    bool ellipsis_seen = false;
    for (int32_t i = 0; i < sparse_spec.dims; i++) {
        const uint64_t bit_i = bit_1_value(i);
        if (ellipsis_seen && (bit_i & params.new_axis_mask) != 0) {
            sparse_spec.num_add_axis_after_ellipsis++;
        }
        if ((bit_i & params.ellipsis_mask) != 0) {
            ellipsis_seen = true;
        }
    }
    // If no ellipsis insert one at the end
    if (!ellipsis_seen) {
        sparse_spec.ellipsis_mask |= bit_1_value(sparse_spec.dims);
        sparse_spec.dims++; // this effects loop iteration below
    }
}

/// 构建密集规格，将稀疏规格转换为完整的索引规格（展开省略号）
static void build_dense_spec(const StridedSliceSparseSpec& sparse, StridedSliceDenseSpec& dense)
{
    // Build expanded begin, end, strides, begin_mask, end_mask
    // to remove any ellipsis
    dense.begin.resize(dense.dims);
    dense.end.resize(dense.dims);
    dense.strides.resize(dense.dims);

    // What indices to get the final shape from.
    dense.begin_mask = 0;
    dense.end_mask = 0;
    dense.shrink_axis_mask = 0;

    int full_index = 0;
    for (int i = 0; i < sparse.dims; i++) {
        const uint64_t bit_i = bit_1_value(i);
        if ((bit_i & sparse.ellipsis_mask) != 0) {
            // Expand the ellipsis into the appropriate indices
            // NOTE: this only works because we guaranteed one ellipsis
            int32_t next_index =
                std::min(dense.dims - (sparse.dims - i) + 1 + sparse.num_add_axis_after_ellipsis, dense.dims);
            for (; full_index < next_index; full_index++) {
                // new_axis' aren't real axis so you have to skip
                dense.begin[full_index] = dense.end[full_index] = 0;
                dense.strides[full_index] = 1;
                dense.begin_mask |= bit_1_value(full_index);
                dense.end_mask |= bit_1_value(full_index);
                dense.final_shape_gather_indices.push_back(full_index);
            }
        } else if ((bit_i & sparse.new_axis_mask) != 0) {
            dense.final_shape_gather_indices.push_back(K_STRIDED_SLICE_NEW_AXIS);
        } else {
            TORCH_CHECK_INDEX(
                static_cast<size_t>(full_index) < dense.begin.size(), "Index out of range using input dim ", full_index,
                "; input has only ", dense.dims, " dims.");

            // Gather slicing spec into appropriate index
            dense.begin[full_index] = sparse.begin[i];
            dense.end[full_index] = sparse.end[i];
            dense.strides[full_index] = sparse.strides[i];

            if ((sparse.begin_mask & bit_i) != 0) {
                dense.begin_mask |= bit_1_value(full_index);
            }
            if ((sparse.end_mask & bit_i) != 0) {
                dense.end_mask |= bit_1_value(full_index);
            }

            // If shrink, record where to get the dimensionality from (i.e.
            // new_axis creates a fake 1 size dimension. Also remember shrink
            // axis (now in dense form) so we can ignore dense.end below.
            if ((sparse.shrink_axis_mask & bit_i) != 0) {
                dense.final_shape_gather_indices.push_back(K_SHRINK_AXIS);
                dense.shrink_axis_mask |= bit_1_value(full_index);
            } else {
                dense.final_shape_gather_indices.push_back(full_index);
            }
            full_index++;
        }
    }
}

/// 构建处理形状，计算中间张量在各维度的形状大小
static void build_processing_shape(
    const StridedSliceDenseSpec& dense_spec, const InputParamUnit& input_param_unit, bool begin_and_end_masked,
    std::vector<int64_t>& processing_shape)
{
    int64_t interval_length;
    bool known_interval = false;
    if (dense_spec.begin_valid && dense_spec.end_valid) {
        interval_length = input_param_unit.end - input_param_unit.begin;
        known_interval = true;
    } else if (input_param_unit.shrink) {
        // The dimension is still known as 1 for the processing_shape, but will be
        // discarded for the final shape.
        interval_length = 1;
        known_interval = true;
    } else if (begin_and_end_masked) {
        // Even if we don't have values for begin or end, we do know that this
        // dimension covers the whole interval. If we have shape information for
        // this dimension, that tells us the interval length.
        if (input_param_unit.dim >= 0) {
            if (input_param_unit.stride < 0) {
                interval_length = -input_param_unit.dim;
            } else {
                interval_length = input_param_unit.dim;
            }
            known_interval = true;
        }
    }
    if (known_interval) {
        int64_t size_i;
        // Hold zero if the interval is degenerate, otherwise account for
        // remainder
        if (interval_length == 0 || ((interval_length < 0) != (input_param_unit.stride < 0))) {
            size_i = 0;
        } else {
            size_i =
                interval_length / input_param_unit.stride + (interval_length % input_param_unit.stride != 0 ? 1 : 0);
        }
        processing_shape.push_back(size_i);
    } else {
        processing_shape.push_back(-1);
    }
}

/// 构建处理数据，规范化begin/end/strides并进行边界检查
static void build_processing_data(
    const StridedSliceDenseSpec& dense_spec, StridedSliceParams& params, ProcessingData& processing_data)
{
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    for (int i = 0; i < static_cast<int>(params.input_shape.size()); ++i) {
        auto& begin_i = params.begin[i];
        auto& end_i = params.end[i];
        auto& stride_i = params.strides[i];
        auto dim_i = params.input_shape[i];
        TORCH_CHECK_VALUE(stride_i != 0, "strides[", i, "] must be non-zero");

        const uint64_t bit_i = bit_1_value(i);
        bool shrink_i = (dense_spec.shrink_axis_mask & bit_i);
        const std::array<uint64_t, 2> masks = {{dense_spec.begin_mask & bit_i, dense_spec.end_mask & bit_i}};
        if (dim_i == -1) {
            processing_data.processing_shape.push_back(shrink_i ? 1 : -1);
            processing_data.processing_begin.push_back(begin_i);
            processing_data.processing_end.push_back(shrink_i ? (begin_i + 1) : end_i);
            processing_data.processing_strides.push_back(shrink_i ? 1 : stride_i);
            continue;
        }

        // 2: begin + end
        const std::array<bool, 2> real_valid = {params.real_begin_valid, params.real_end_valid};
        const std::array<int64_t, 2> valid_range = {{stride_i > 0 ? 0 : -1, stride_i > 0 ? dim_i : dim_i - 1}};

        auto canonical = [stride_i, dim_i, masks, valid_range, real_valid](int64_t x, int c) {
            if (masks[c]) {
                return stride_i > 0 ? valid_range[c] :
                                      valid_range[static_cast<uint64_t>(c + 1) & static_cast<uint64_t>(1)];
            } else {
                if (!real_valid[c]) {
                    return INVALID_IDX; // -3 invalid, diff from valid_range
                }
                int64_t x_fwd = normalize_index(x, dim_i);
                return x_fwd < valid_range[0] ? valid_range[0] : std::min(x_fwd, valid_range[1]);
            }
        };

        TORCH_CHECK_VALUE(!(shrink_i && stride_i <= 0), "only stride 1 allowed on non-range indexing.");
        is_simple_slice = is_simple_slice && (stride_i == 1);

        const bool begin_and_end_masked =
            ((dense_spec.begin_mask & bit_i) != 0) && ((dense_spec.end_mask & bit_i) != 0);
        if (dense_spec.begin_valid && dense_spec.end_valid) {
            if (shrink_i) {
                // If we are shrinking, the end index is now possibly incorrect. In
                // particular foo[-1] produces sparse_begin = -1, sparse_end = 0.
                // and canonical puts these to n-1 and 0, which implies a degenerate
                // interval. Fortunately, it is now safe to re-create end as begin+1.
                if (real_valid[0]) {
                    int64_t x_fwd = normalize_index(begin_i, dim_i);
                    begin_i = x_fwd;
                    end_i = begin_i + 1;
                    TORCH_CHECK_INDEX(
                        !fwd_out_of_bound(x_fwd, 0, dim_i), "slice index ", begin_i, " of dimension ", i,
                        " out of bounds.");
                } else {
                    begin_i = -2; // -2 valid, diff from valid_range
                    end_i = begin_i + 1;
                }
            } else {
                begin_i = canonical(begin_i, 0);
                end_i = canonical(end_i, 1);
            }

            // -3 invalid, diff from valid_range
            TORCH_CHECK_VALUE(
                !((!real_valid[0] || !real_valid[1]) && (begin_i == INVALID_IDX || end_i == INVALID_IDX)),
                "begin_i:", begin_i, " end_i:", end_i, " is invalid while unconst begin or end, shrink_i:", shrink_i,
                " masks:", masks[0], masks[1]);

            processing_data.processing_begin.push_back(begin_i);
            processing_data.processing_end.push_back(end_i);
            processing_data.processing_strides.push_back(stride_i);

            // Update optimization values
            bool take_all_in_dimension = stride_i == 1 && begin_i == 0 && end_i == dim_i;
            is_identity = is_identity && take_all_in_dimension;
            slice_dim0 = slice_dim0 && ((i == 0 && stride_i == 1) || take_all_in_dimension);
        } else {
            is_identity = is_identity && (stride_i == 1 && begin_and_end_masked);
            slice_dim0 = slice_dim0 && ((i == 0 && stride_i == 1) || begin_and_end_masked);
            processing_data.processing_begin.push_back(begin_i);
            processing_data.processing_end.push_back(end_i);
            processing_data.processing_strides.push_back(1);
        }

        // Compute the processing shape (the intermediate Eigen will produce)
        InputParamUnit input_param_unit = {begin_i, end_i, stride_i, dim_i, shrink_i};
        build_processing_shape(dense_spec, input_param_unit, begin_and_end_masked, processing_data.processing_shape);
    }
}

/// 构建最终形状，应用new_axis和shrink操作后计算输出形状
static std::vector<int64_t> build_final_shape(
    const ProcessingData& processing_data, const StridedSliceDenseSpec& dense_spec, StridedSliceParams& params)
{
    params.begin.clear();
    params.end.clear();
    params.strides.clear();
    std::vector<int64_t> out_shape;
    std::vector<int64_t> final_shape_input;
    int shrink_gather_index = 0;
    for (size_t i = 0; i < dense_spec.final_shape_gather_indices.size(); i++) {
        auto gather_index = dense_spec.final_shape_gather_indices[i];
        if (gather_index >= 0) {
            const auto dim_gather_i = processing_data.processing_shape[gather_index];
            out_shape.push_back(dim_gather_i);
            final_shape_input.push_back(params.input_shape[gather_index]);
            params.begin.push_back(processing_data.processing_begin[gather_index]);
            params.end.push_back(processing_data.processing_end[gather_index]);
            params.strides.push_back(processing_data.processing_strides[gather_index]);
            shrink_gather_index = gather_index + 1;
        } else if (gather_index == K_STRIDED_SLICE_NEW_AXIS) {
            out_shape.push_back(1);
            // input is scalar
            if (params.input_shape.empty()) {
                final_shape_input.push_back(1);
                params.begin.push_back(0);
                params.end.push_back(1);
                params.strides.push_back(1);
            }
        } else {
            final_shape_input.push_back(params.input_shape[shrink_gather_index]);
            params.begin.push_back(processing_data.processing_begin[shrink_gather_index]);
            params.end.push_back(processing_data.processing_begin[shrink_gather_index] + 1);
            params.strides.push_back(1);
            shrink_gather_index += 1;
        }
    }

    params.input_shape = final_shape_input;
    return out_shape;
}

/// 内部推断输出形状的核心函数，分四步处理：1)构建稀疏规格 2)构建密集规格 3)构建处理数据 4)构建最终形状
static inline c10::SmallVector<int64_t, SIZE> infer_shape_internal(StridedSliceParams& params)
{
    ASCEND_LOGD("input params: %s", params.to_string().c_str());
    TORCH_CHECK_VALUE(
        params.begin.size() == params.end.size() && params.end.size() == params.strides.size(),
        "Expected begin, end, and strides to be 1D equal size tensors, but got shapes [", params.begin.size(), "], [",
        params.end.size(), "], [", params.strides.size(), "] instead.");

    // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
    // i.e. there exists only no more than one ellipsis
    auto& ellipsis_mask = params.ellipsis_mask;
    TORCH_CHECK_VALUE(
        !(ellipsis_mask != 0 && ((ellipsis_mask & (ellipsis_mask - 1)) != 0)),
        "Multiple ellipses in slice spec not allowed.");

    // Step 1: Account for ellipsis and new axis
    //
    // Check for ellipses and count how many non-newaxis' there are after
    StridedSliceSparseSpec sparse_spec{
        0,
        0,
        params.begin,
        params.end,
        params.strides,
        params.begin_mask,
        params.end_mask,
        params.ellipsis_mask,
        params.new_axis_mask,
        params.shrink_axis_mask};
    build_sparse_spec(params, sparse_spec);

    // Step 2: Make a sparse spec into a full index spec
    //
    // The sparse spec does not correspond to the number of dimensions
    // Make a dense spec that corresponds to the number of dimensions
    //
    // For example suppose foo[...,3:] on foo.shape=(2,2,3) then
    // we need to produce the missing begin_mask for the first two
    // dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
    // we achieve begin_mask=6, end_mask=7
    StridedSliceDenseSpec dense_spec{
        static_cast<int64_t>(params.input_shape.size()),
        0,
        0,
        params.begin_valid,
        params.end_valid,
        params.begin,
        params.end,
        params.strides,
        {},
        0};
    build_dense_spec(sparse_spec, dense_spec);

    ASCEND_LOGD("dense spec: %s", dense_spec.to_string().c_str());

    // Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
    //         and bounds check!
    ProcessingData processing_data;
    params.begin = dense_spec.begin;
    params.end = dense_spec.end;
    params.strides = dense_spec.strides;
    build_processing_data(dense_spec, params, processing_data);

    ASCEND_LOGD("processing data: %s", processing_data.to_string().c_str());

    // Step 4: Compute the final shape
    //
    // new_axis will increase dimension by 1 (with a one-size dimension)
    // slices like foo[3,...] will reduce dimension by 1.
    // This cannot be done earlier, because it depends on Step 3.
    auto out_shape = build_final_shape(processing_data, dense_spec, params);

    ASCEND_LOGD("after infershape params: %s", params.to_string().c_str());
    ASCEND_LOGI("[npu_indexing] output shape: %s", op_plugin::utils::get_vector_str(out_shape).c_str());

    TORCH_CHECK_VALUE(out_shape.size() <= SIZE, "The output tensor cannot be larger than ", SIZE, " dimensions");
    return c10::SmallVector<int64_t, SIZE>(out_shape);
}

/// 计算NPU索引操作的输出大小
static c10::SmallVector<int64_t, SIZE> npu_indexing_output_size(
    const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask,
    int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask)
{
    StridedSliceParams params;

    params.input_shape = self.sizes().vec();

    params.begin = begin.vec();
    params.end = end.vec();
    params.strides = strides.vec();

    // Set mask values
    params.begin_mask = static_cast<uint64_t>(begin_mask);
    params.end_mask = static_cast<uint64_t>(end_mask);
    params.ellipsis_mask = static_cast<uint64_t>(ellipsis_mask);
    params.new_axis_mask = static_cast<uint64_t>(new_axis_mask);
    params.shrink_axis_mask = static_cast<uint64_t>(shrink_axis_mask);

    // Set validity flags
    params.begin_valid = !begin.empty();
    params.end_valid = !end.empty();
    params.stride_valid = !strides.empty();

    return infer_shape_internal(params);
}
} // namespace

at::Tensor& npu_indexing_out(
    const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask,
    int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor& out)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_indexing_out(
            self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out);
    }

    DO_COMPATIBILITY(
        aclnnStridedSlice,
        acl_op::npu_indexing_out(
            self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out));
    auto out_size = npu_indexing_output_size(
        self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
    npu_preparation::check_tensor({self}, out, out.scalar_type(), out_size);
    EXEC_NPU_CMD(
        aclnnStridedSlice, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask, out);
    return out;
}

at::Tensor npu_indexing(
    const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask,
    int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_indexing(
            self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
    }

    DO_COMPATIBILITY(
        aclnnStridedSlice,
        acl_op::npu_indexing(
            self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask));
    auto out_size = npu_indexing_output_size(
        self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
    at::Tensor out = npu_preparation::apply_tensor_without_format(out_size, self.options());
    EXEC_NPU_CMD(
        aclnnStridedSlice, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask, out);
    return out;
}
} // namespace op_api

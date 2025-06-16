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

#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/AdvancedIndex.h"

namespace op_plugin {

namespace {
std::vector<at::Tensor> npu_expand_outplace(at::TensorList to_expand)
{
    // expands a list of Tensors; ignores undefined (null) tensors
    bool first = true;
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
            continue;
        } else if (first) {
            sizes = to_expand[i].sizes().vec();
            first = false;
        } else {
            sizes = at::infer_size(sizes, to_expand[i].sizes());
        }
    }

    std::vector<at::Tensor> result(to_expand.size());
    for (size_t i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
            continue;
        } else if (to_expand[i].sizes().equals(sizes)) {
            result[i] = to_expand[i];
        } else {
            if (to_expand[i].dtype() == at::kLong) {
                result[i] = to_expand[i].to(at::kInt).expand(sizes, true);
            } else {
                result[i] = to_expand[i].expand(sizes, true);
            }
        }
    }
    return result;
}

at::Tensor npu_nonzero_aclop(const at::Tensor &self)
{
    c10::SmallVector<int64_t, SIZE> output_size = {self.dim(), self.numel()};
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor(output_size, self.options().dtype(at::kLong), self);
    c10::SmallVector<int64_t, N> output_sync_idx = {0};
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx).Name("NonZero").Input(self).Output(result).Attr("transpose", true).Run();
    return result;
}

at::Tensor npu_nonzero_aclnn(const at::Tensor &self)
{
    DO_COMPATIBILITY(aclnnNonzeroV2, npu_nonzero_aclop(self));
    c10::SmallVector<int64_t, SIZE> out_size = {self.dim(), self.numel()};
    at::Tensor out =
        at_npu::native::OpPreparation::apply_tensor_without_format(out_size, self.options().dtype(at::kLong));
    static auto aclGetViewShapeAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr);
        return ret;
    }();
    using aclGetViewShapeFuncLocal = int (*)(const aclTensor* tensor, int64_t** view_dims, uint64_t* view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFuncLocal>(aclGetViewShapeAddr);
    OP_EXEC_LOG(aclnnNonzeroV2, "EXEC_NPU_CMD_SYNC", self, out);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnNonzeroV2, self, out);
    int64_t* view_dims = nullptr;
    uint64_t view_dim_num = 0;
    auto ret = aclGetViewShape(npuAclParams.Get<1>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret == 0, "aclGetViewShape failed.");
    c10::SmallVector<int64_t, op_infer::SIZE> output_size(view_dims, view_dims + view_dim_num);
    out = out.resize_(output_size);
    // Need to use delete[] to release memory to avoid memory leakage!
    delete[] view_dims;
    view_dims = nullptr;
    return out;
}
} // namespace

AdvancedIndex::AdvancedIndex(const at::Tensor &src, at::TensorList list_indices)
{
    int64_t before_dims = 0;
    int64_t after_dims = 0;
    int64_t indexed_dims = 0;
    at::IntArrayRef replacement_shape;
    for (size_t dim = 0; dim < list_indices.size(); dim++) {
        if (!list_indices[dim].defined()) {
            if (indexed_dims == 0) {
                before_dims++;
            } else {
                after_dims++;
            }
        } else {
            indexed_dims++;
            replacement_shape = list_indices[dim].sizes();
            indexed_sizes.push_back(src.size(dim));
            indexed_strides.push_back(src.stride(dim));
        }
    }

    // Check if the indexed subspace contains a dim of size 0, but the replacement shape does not.
    // This implies that an index is out of bounds, because there
    // is no number that's a valid index for an empty tensor.
    // Normally, out of bounds is handled in the indexing kernel, but this case fails earlier
    // in restride_src with an unhelpful error message.
    if (std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end() &&
        std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end()) {
        TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0.", OPS_ERROR(ErrCode::PARAM));
    }

    this->dims_before = before_dims;
    this->dims_after = after_dims;
    this->src = AdvanceIndex::restride_src(src, before_dims, indexed_dims, replacement_shape);

    for (auto &index : list_indices) {
        if (index.defined()) {
            indices.push_back(AdvanceIndex::reshape_indexer(index, before_dims, after_dims));
        }
    }
}

bool AdvanceIndex::all_strides_match(at::TensorList tensor_list)
{
    TORCH_CHECK(tensor_list.size() >= 1, OPS_ERROR(ErrCode::PARAM));
    auto strides = tensor_list[0].strides();
    for (auto &tensor : tensor_list.slice(1)) {
        if (!strides.equals(tensor.strides())) {
            return false;
        }
    }
    return true;
}

at::Tensor AdvanceIndex::reshape_indexer(const at::Tensor &index, int64_t dims_before, int64_t dims_after)
{
    auto orig_shape = index.sizes();
    auto shape = at::DimVector();
    shape.append(dims_before, 1);
    shape.append(orig_shape.begin(), orig_shape.end());
    shape.append(dims_after, 1);
    if (index.dtype() == at::kLong) {
        return index.reshape(shape);
    } else {
        return index.reshape(shape).to(at::kLong);
    }
}

at::Tensor AdvanceIndex::restride_src(const at::Tensor &src, int64_t before_dims, int64_t dims_indexed,
                                      at::IntArrayRef replacement_shape)
{
    auto shape = at::DimVector(src.sizes());
    auto strides = at::DimVector(src.strides());
    int64_t end = before_dims + dims_indexed;
    TORCH_CHECK(shape.size() >= end, "end", end, "is overrange shape.size() ", shape.size(), OPS_ERROR(ErrCode::VALUE));
    shape.erase(shape.begin() + before_dims, shape.begin() + end);
    TORCH_CHECK(strides.size() >= end, "end", end, "is overrange strides.size() ", strides.size(),
        OPS_ERROR(ErrCode::VALUE));
    strides.erase(strides.begin() + before_dims, strides.begin() + end);
    shape.insert(shape.begin() + before_dims, replacement_shape.begin(), replacement_shape.end());
    strides.insert(strides.begin() + before_dims, replacement_shape.size(), 0);
    return src.as_strided(shape, strides);
}

std::string AdvanceIndex::shapes_as_str(at::TensorList tensors)
{
    std::ostringstream os;
    bool first = true;
    for (auto &t : tensors) {
        if (t.defined()) {
            if (!first) {
                os << ", ";
            }
            os << t.sizes();
            first = false;
        }
    }
    return os.str();
}

bool AdvanceIndex::checkIndexTensorTypes(const torch::List<c10::optional<at::Tensor>> &indices)
{
    bool needCast = false;
    c10::optional<at::ScalarType> indicesDtype;
    for (c10::optional<at::Tensor> tensor : indices) {
        if (tensor.has_value() && tensor->defined()) {
            auto scalarType = tensor->scalar_type();
            if (scalarType != at::kLong && scalarType != at::kByte &&
                scalarType != at::kBool && scalarType != at::kInt) {
                TORCH_CHECK_INDEX(false, "tensors used as indices must be long, int, byte, or bool tensors",
                    OPS_ERROR(ErrCode::TYPE));
            }
            if (!indicesDtype.has_value()) {
                indicesDtype = scalarType;
            } else if (indicesDtype.value() != scalarType) {
                needCast = true;
            }
        }
    }
    return needCast;
}

AdvancedIndex AdvanceIndex::make_info(at::Tensor self, const torch::List<c10::optional<at::Tensor>> &orig)
{
    AdvanceIndex::checkIndexTensorTypes(orig);
    // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
    auto indices = at::native::expandTensors(self, orig);
    // next broadcast all index tensors together
    try {
        indices = npu_expand_outplace(indices);
    } catch (std::exception &e) {
        TORCH_CHECK_INDEX(false,
                          "shape mismatch: indexing tensors could not be broadcast"
                          " together with shapes ",
                          shapes_as_str(indices),
                          OPS_ERROR(ErrCode::VALUE));
    }
    // add missing null Tensors so that it matches self.dim().
    while (indices.size() < static_cast<size_t>(self.dim())) {
        indices.emplace_back();
    }
    // if the non-null indices are not all adjacent, transpose self
    // and indices together so that they're adjacent at the front
    if (!at::native::hasContiguousSubspace(indices)) {
        std::tie(self, indices) = at::native::transposeToFront(self, indices);
    }
    // Ensure indices are on the same device as self
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i].defined() && indices[i].device() != self.device()) {
            indices[i] = indices[i].to(self.device());
        }
    }
    return AdvancedIndex(self, indices);
}

std::vector<at::Tensor> AdvanceIndex::npu_expand_tensors(const at::Tensor &self,
                                                         const torch::List<c10::optional<at::Tensor>> &indices,
                                                         bool needCast,
                                                         bool flag_aclnn)
{
    // If indices come in as ByteTensor or BoolTensor (masks), expand them into the equivalent indexing by LongTensors
    std::vector<at::Tensor> result;
    for (c10::optional<at::Tensor> index_opt : indices) {
        if (!index_opt.has_value()) {
            result.emplace_back();
        } else {
            at::Tensor index = std::move(*index_opt);
            if (index.defined() && index.device() != self.device()) {
                index = index.to(self.device());
            }
            if (index.scalar_type() == at::kByte || index.scalar_type() == at::kBool) {
                if (index.scalar_type() == at::kByte) {
                    TORCH_WARN("indexing with dtype torch.uint8 is now deprecated,"
                               " please use a dtype torch.bool instead.");
                }
                // The sizes of the ByteTensor mask or bool tensor must match the sizes of the corresponding dimensions
                // in self
                for (uint64_t j = 0; j < static_cast<uint64_t>(index.dim()); j++) {
                    uint64_t srcIdx = result.size() + j;
                    TORCH_CHECK_INDEX(index.size(j) == self.size(srcIdx), "The shape of the mask ", index.sizes(),
                                      " at index ", j, " does not match the shape of the indexed tensor ", self.sizes(),
                                      " at index ", srcIdx, OPS_ERROR(ErrCode::VALUE));
                }
                at::Tensor nonzero;
                // Replace with nonzeros
                nonzero = flag_aclnn ? npu_nonzero_aclnn(index) : npu_nonzero_aclop(index);
                for (int64_t j = 0; j < index.dim(); j++) {
                    result.emplace_back(nonzero.select(0, j));
                }
            } else {
                result.emplace_back(std::move(index));
            }
        }
    }
    if (needCast) {
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i].defined() && result[i].dtype() == at::kInt) {
                result[i] = result[i].to(at::kLong);
            }
        }
    }
    return result;
}

std::vector<at::Tensor> AdvanceIndex::npu_broadcast_tensors(std::vector<at::Tensor> to_broadcast)
{
    // Broadcast a list of Tensors, ignoring undefined (null) tensors.
    bool first = true;
    std::vector<int64_t> sizes;
    for (uint64_t i = 0; i < to_broadcast.size(); ++i) {
        if (!to_broadcast[i].defined()) {
            continue;
        } else if (first) {
            // The initial value of sizes is the first defined tensor's shape.
            sizes = to_broadcast[i].sizes().vec();
            first = false;
        } else {
            sizes = at::infer_size(sizes, to_broadcast[i].sizes());
        }
    }

    std::vector<at::Tensor> result(to_broadcast.size());
    for (uint64_t i = 0; i < to_broadcast.size(); ++i) {
        if (!to_broadcast[i].defined()) {
            continue;
        } else if (to_broadcast[i].sizes().equals(sizes)) {
            result[i] = to_broadcast[i];
        } else {
            result[i] = op_plugin::npu_broadcast(to_broadcast[i], sizes);
        }
    }
    return result;
}

bool AdvanceIndex::is_expandable_to(c10::IntArrayRef shape, c10::IntArrayRef desired)
{
    // True if `shape` can be broadcasted to `desired`
    size_t ndim = shape.size();
    size_t target_dim = desired.size();
    if (ndim > target_dim) {
        return false;
    }
    for (size_t i = 0; i < ndim; i++) {
        int64_t size = shape[ndim - i - 1];
        int64_t target = desired[target_dim - i - 1];
        if (size != target && size != 1) {
            return false;
        }
    }
    return true;
}

} // namespace op_plugin

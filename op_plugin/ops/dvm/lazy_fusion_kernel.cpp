// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/ops/dvm/lazy_fusion_kernel.h"
#include <algorithm>
#include <linux/limits.h>
#include <sys/stat.h>
#include <fstream>
#include <ATen/record_function.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace lazy_fusion {
static inline bool DumpEnabled() {
  static const bool v = g_lazy_fusion_manager.flags_.dump_as_text;
  return v;
}

// Whether bf16 inputs must be cast to fp32 before entering DVM. Required on
// Ascend910B-class chips; on Ascend950 (A5) DVM handles bf16 natively, so the
// cast (which costs an extra round-trip in the fused kernel) is skipped.
static inline bool NeedBf16ToFp32Cast() {
  static const bool need = (c10_npu::GetSocVersion() != c10_npu::SocVersion::Ascend950);
  return need;
}
namespace {
class LazyFusionDump {
 public:
  static LazyFusionDump &Instance() {
    static LazyFusionDump instance;
    return instance;
  }

  void DumpGraphInfo(std::stringstream &buf) {
    std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + "_graph.txt";
    std::string file_path = dump_dir_ + "/" + file_name;
    DumpToFile(file_path, buf);
  }

  void DumpKernelInfo(std::stringstream &buf) {
    std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + "_kernel.txt";
    std::string file_path = dump_dir_ + "/" + file_name;
    DumpToFile(file_path, buf);
  }

 private:
  void DumpToFile(const std::string &file_path, std::stringstream &buf) {
    if (!enable_dump_) {
      return;
    }
    ChangeFileMode(file_path, S_IWUSR);
    std::ofstream of(file_path, std::ios::app);
    if (!of.is_open()) {
      ASCEND_LOGW("Open dump file '%s' failed!", file_path.c_str());
      ChangeFileMode(file_path, S_IRUSR);
      return;
    }
    of << buf.str() << "\n";
    of.close();
    ChangeFileMode(file_path, S_IRUSR);
    buf.str("");
  }
  LazyFusionDump() { CreateDumpDir(); }
  ~LazyFusionDump() = default;

  void ChangeFileMode(const std::string &file_name, mode_t mode) {
    if (access(file_name.c_str(), F_OK) == -1) {
      return;
    }
    try {
      if (chmod(file_name.c_str(), mode) != 0) {
        ASCEND_LOGW("Change file '%s' to mode %d fail", file_name.c_str(), mode);
      }
    } catch (std::exception &e) {
      ASCEND_LOGW("File '%s' change mode failed! May be not exist.", file_name.c_str())
    }
  }

  bool IsFileExist(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return access(path.c_str(), F_OK) == 0;
  }

  bool IsDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    struct stat st = {0};
    int ret = lstat(path.c_str(), &st);
    if (ret != 0) {
      return false;
    }
    return S_ISDIR(st.st_mode);
  }

  bool CreateDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    if (IsFileExist(path)) {
      return IsDir(path);
    }
    size_t pos = 0;
    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
      std::string base_dir = path.substr(0, ++pos);
      if (IsFileExist(base_dir)) {
        if (IsDir(base_dir)) {
          continue;
        } else {
          return false;
        }
      }
      if (mkdir(base_dir.c_str(), 0750) != 0) {
        return false;
      }
    }
    return mkdir(path.c_str(), 0750) == 0;
  }

  void CreateDumpDir() {
    if (!g_lazy_fusion_manager.flags_.dump_as_text) {
      return;
    }
    enable_dump_ = CreateDir(g_lazy_fusion_manager.flags_.dump_dir);
    if (!enable_dump_) {
      ASCEND_LOGW("Failed to create dump directory: %s", g_lazy_fusion_manager.flags_.dump_dir.c_str());
      return;
    }
    auto path = g_lazy_fusion_manager.flags_.dump_dir.c_str();
    char real_path[PATH_MAX] = {0};
    if (strlen(path) >= PATH_MAX || realpath(path, real_path) == nullptr) {
      ASCEND_LOGW("Get realpath failed, path: %s", path);
      enable_dump_ = false;
      return;
    }
    dump_dir_ = real_path;
    ASCEND_LOGW("dvm dump directory: %s", dump_dir_.c_str());
  }

  bool enable_dump_{false};
  std::string dump_dir_;
};

}  // namespace

Manager g_lazy_fusion_manager;

Manager::~Manager() {
  while (!pool_.empty()) {
    auto top = pool_.front();
    delete top;
    pool_.pop();
  }
}

LazyFusionKernel *Manager::NewKernel() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!pool_.empty()) {
      auto k = pool_.front();
      pool_.pop();
      return k;
    }
  }
  return new LazyFusionKernel();
}

LazyFusionKernel::LazyFusionKernel() {
  dvm::Kernel::Reset(dvm::KernelType::kEager, dvm::KernelFlag::kUnifyWS);
  inputs_.reserve(32);
  outputs_.reserve(16);
}

LazyFusionKernel::~LazyFusionKernel() {
  for (auto load : inputs_) {
    delete load;
  }
  for (auto s : cached_shape_) {
    delete s;
  }
  for (auto op : dvm_ops_) {
    delete op;
  }
}

void LazyFusionKernel::CacheTensorMeta(TensorMeta *meta, const at::Tensor &tensor) {
  meta->tensor_impl = tensor.unsafeGetTensorImpl();
  meta->data_ptr = tensor.data_ptr();
  meta->storage_offset = tensor.storage_offset();
  meta->dtype = tensor.scalar_type();
  meta->dim = tensor.dim();
  meta->sizes.resize(meta->dim);
  meta->strides.resize(meta->dim);
  for (size_t i = 0; i < meta->sizes.size(); ++i) {
    meta->sizes[i] = tensor.size(i);
    meta->strides[i] = tensor.stride(i);
  }
}

// Exact-match reuse intentionally includes shape/stride metadata so sibling views
// that share one storage do not reuse the wrong NDObject.
bool LazyFusionKernel::MatchTensorMeta(const TensorMeta &meta, const at::Tensor &tensor) {
  auto dim = tensor.dim();
  if (meta.tensor_impl != tensor.unsafeGetTensorImpl() || meta.data_ptr != tensor.data_ptr() ||
      meta.storage_offset != tensor.storage_offset() || meta.dtype != tensor.scalar_type() || meta.dim != dim) {
    return false;
  }
  for (size_t i = 0; i < meta.sizes.size(); ++i) {
    if (meta.sizes[i] != tensor.size(i) || meta.strides[i] != tensor.stride(i)) {
      return false;
    }
  }
  return true;
}

// `shape` may override the tensor's physical sizes for broadcast-like DVM inputs.
// The issue is not inside one op call itself, but across later reuse in the same
// pending fusion. For example:
//   y0 = batch_norm_elemt(x, weight, bias, mean, invstd, eps)
//   y1 = add(mean, other)
//
// batch_norm_elemt feeds `mean` to DVM as [1, C, 1, 1] through new_shape_ref,
// while `mean.sizes()` is still [C]. Because batch_norm_elemt does not flush at
// the end, a later fused add(mean, other) may try to reuse the cached NDObject
// for `mean`. If reuse only compares at::Tensor::sizes(), the [1, C, 1, 1]
// version could be wrongly reused when add expects [C], and the later DVM
// shape/broadcast inference would be wrong.
void LazyFusionKernel::CacheDvmShape(DvmOp *dvm_op, dvm::ShapeRef *shape, c10::IntArrayRef default_shape) {
  auto req_shape = shape != nullptr ? c10::IntArrayRef(shape->data, shape->size) : default_shape;
  dvm_op->dvm_shape.resize(req_shape.size());
  for (size_t i = 0; i < req_shape.size(); ++i) {
    dvm_op->dvm_shape[i] = req_shape[i];
  }
}

bool LazyFusionKernel::MatchDvmShape(const DvmOp *dvm_op, dvm::ShapeRef *shape, c10::IntArrayRef default_shape) {
  auto req_shape = shape != nullptr ? c10::IntArrayRef(shape->data, shape->size) : default_shape;
  if (dvm_op->dvm_shape.size() != req_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < req_shape.size(); ++i) {
    if (dvm_op->dvm_shape[i] != req_shape[i]) {
      return false;
    }
  }
  return true;
}

bool LazyFusionKernel::MatchDvmOp(const DvmOp *dvm_op, const at::Tensor &tensor, dvm::ShapeRef *shape) {
  return MatchTensorMeta(dvm_op->tensor_meta, tensor) && MatchDvmShape(dvm_op, shape, tensor.sizes());
}

void LazyFusionKernel::CacheDvmOp(torch_npu::NPUStorageImpl *storage, const at::Tensor &tensor, const TensorMeta *tensor_meta,
                                  dvm::NDObject *obj, dvm::ShapeRef *shape, bool reloadable_from_gm) {
  if (dvm_ops_used_ == dvm_ops_.size()) {
    dvm_ops_.push_back(new DvmOp());
  }
  auto p = dvm_ops_[dvm_ops_used_++];
  p->op = obj;
  if (tensor_meta != nullptr) {
    p->tensor_meta = *tensor_meta;
  } else {
    CacheTensorMeta(&(p->tensor_meta), tensor);
  }
  CacheDvmShape(p, shape, tensor.sizes());
  p->reloadable_from_gm = reloadable_from_gm;
  storage->lazy_fusion_data_ = p;
}

// Input-side flush is decided before graph mutation; by the time Input() runs,
// a non-reloadable alias miss should already have forced a flush.
bool LazyFusionKernel::NeedFlushForInput(const at::Tensor &x, dvm::ShapeRef *shape) const {
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(x);
  if (storage->lazy_fusion_data_ == nullptr) {
    return false;
  }
  auto dvm_op = static_cast<DvmOp *>(storage->lazy_fusion_data_);
  if (dvm_op->reloadable_from_gm) {
    return false;
  }
  return !MatchDvmOp(dvm_op, x, shape);
}

// Writable tensors are even stricter: anything other than the exact same tensor
// is conservatively flushed to avoid aliasing/overlap write hazards.
bool LazyFusionKernel::NeedFlushForWritableOutput(const at::Tensor &tensor) const {
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(tensor);
  if (storage->lazy_fusion_data_ == nullptr) {
    return false;
  }
  return !MatchDvmOp(static_cast<DvmOp *>(storage->lazy_fusion_data_), tensor, nullptr);
}

dvm::NDObject *LazyFusionKernel::Input(const at::Tensor &x, bool enable_cast, dvm::ShapeRef *shape) {
  auto input_type = TransType(x.scalar_type());
  bool cast_to_fp32 = (enable_cast && input_type == dvm::DType::kBFloat16 && NeedBf16ToFp32Cast());
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(x);
  if (storage->lazy_fusion_data_ != nullptr) {
    auto dvm_op = static_cast<DvmOp *>(storage->lazy_fusion_data_);
    if (MatchDvmOp(dvm_op, x, shape)) {
      ASCEND_LOGD("dvm Input: cache hit, reuse NDObject for tensor %p, shape=%s",
                  x.unsafeGetTensorImpl(), ToString(x).c_str());
      auto op = dvm_op->op;
      if (cast_to_fp32) {
        op = Cast(op, dvm::DType::kFloat32);
      }
      return op;
    }
    ASCEND_LOGD("dvm Input: cache miss (reloadable=%d) for tensor %p, shape=%s",
                dvm_op->reloadable_from_gm, x.unsafeGetTensorImpl(), ToString(x).c_str());
    TORCH_INTERNAL_ASSERT(dvm_op->reloadable_from_gm, "non-reloadable dvm input alias must flush before Input");
  }
  if (input_used_ == inputs_.size()) {
    inputs_.push_back(new Load());
  }
  auto load = inputs_[input_used_++];
  if (shape != nullptr) {
    load->shape = *shape;
  } else {
    auto ptr = GetShapeRef(x.sizes());
    load->shape = *ptr;
  }
  load->data_ptr = x.data_ptr();
  load->tensor = x;
  // Three cases:
  //   1) shape != nullptr  : caller provided a physical-shape override, e.g. matmul's
  //                          transpose-of-contig path (uses trans flag downstream).
  //                          → simple Load (no stride).
  //   2) shape == nullptr + tensor contig
  //                          → simple Load (no stride).
  //   3) shape == nullptr + tensor non-contig (e.g. PyTorch view from chunk/slice)
  //                          → strided Load using the tensor's own strides.
  dvm::NDObject *load_op = nullptr;
  if (shape == nullptr && !x.is_contiguous()) {
    auto stride_ptr = GetShapeRef(x.strides());
    load->stride = *stride_ptr;
    load->has_stride = true;
    load_op = dvm::Kernel::Load(load->data_ptr, &(load->shape), &(load->stride), input_type);
    ASCEND_LOGD("dvm Input: new StridedLoad p%zu, tensor %p, addr=%p, shape=%s, dtype=%s",
                input_used_ - 1, x.unsafeGetTensorImpl(), x.data_ptr(),
                ToString(x).c_str(), c10::toString(x.scalar_type()));
  } else {
    load->has_stride = false;
    load_op = dvm::Kernel::Load(load->data_ptr, &(load->shape), input_type);
    ASCEND_LOGD("dvm Input: new Load p%zu, tensor %p, addr=%p, shape=%s, dtype=%s",
                input_used_ - 1, x.unsafeGetTensorImpl(), x.data_ptr(),
                ToString(x).c_str(), c10::toString(x.scalar_type()));
  }
  if (DumpEnabled()) {
    dump_buf_ << "p" << (input_used_ - 1) << ": " << ToString(x) << "\n";
  }
  CacheDvmOp(storage, x, nullptr, load_op, &(load->shape), true);
  return cast_to_fp32 ? Cast(load_op, dvm::DType::kFloat32) : load_op;
}

dvm::NDObject *LazyFusionKernel::ViewInput(const at::Tensor &base, void *data_ptr, dvm::ShapeRef *shape,
                                             dvm::ShapeRef *stride, bool enable_cast) {
  auto input_type = TransType(base.scalar_type());
  bool cast_to_fp32 = (enable_cast && input_type == dvm::DType::kBFloat16 && NeedBf16ToFp32Cast());

  if (input_used_ == inputs_.size()) {
    inputs_.push_back(new Load());
  }
  auto load = inputs_[input_used_++];
  load->shape = *shape;
  load->stride = *stride;
  load->has_stride = true;
  load->data_ptr = data_ptr;
  load->tensor = base;  // hold base tensor to keep storage alive until Flush
  auto load_op = dvm::Kernel::Load(load->data_ptr, &(load->shape), &(load->stride), input_type);
  ASCEND_LOGD("dvm ViewInput: new StridedLoad p%zu, base=%p, addr=%p, dtype=%s",
              input_used_ - 1, base.unsafeGetTensorImpl(), data_ptr,
              c10::toString(base.scalar_type()));
  if (DumpEnabled()) {
    dump_buf_ << "p" << (input_used_ - 1) << ": ViewInput(base=" << ToString(base) << ")\n";
  }

  // Register a DvmOp on base.storage->lazy_fusion_data_ with the *view's* metadata.
  // This protects against the read/write race scenario:
  //
  //   y = npu_swiglu(x, -1)   // ViewInput on x.storage at two offsets, no flush yet
  //   x.mul_(2)               // inplace write to same storage
  //                           // PrepareWritableOutput(x) checks storage->lazy_fusion_data_:
  //                           //   - if null/missing → would NOT flush → race in fused kernel
  //                           //   - now non-null with view metadata → strict mismatch
  //                           //     against full-x metadata → triggers flush ✓
  //
  // reloadable_from_gm=true because data is on GM and any later Input()/ViewInput()
  // can re-Load with new shape/stride without flushing.
  auto req_shape = c10::IntArrayRef(shape->data, shape->size);
  auto req_stride = c10::IntArrayRef(stride->data, stride->size);
  TensorMeta view_meta;
  view_meta.tensor_impl = base.unsafeGetTensorImpl();
  view_meta.data_ptr = data_ptr;
  view_meta.storage_offset = static_cast<int64_t>(
      (static_cast<const char *>(data_ptr) - static_cast<const char *>(base.storage().data())) /
      base.element_size());
  view_meta.dtype = base.scalar_type();
  view_meta.dim = static_cast<int64_t>(req_shape.size());
  view_meta.sizes.assign(req_shape.begin(), req_shape.end());
  view_meta.strides.assign(req_stride.begin(), req_stride.end());
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(base);
  CacheDvmOp(storage, base, &view_meta, load_op, shape, true);

  return cast_to_fp32 ? Cast(load_op, dvm::DType::kFloat32) : load_op;
}

void LazyFusionKernel::Output(const at::Tensor &tensor, dvm::NDObject *obj, bool inplace) {
  auto tensor_type = TransType(tensor.scalar_type());
  if (dvm::Kernel::GetDType(obj) != tensor_type) {
    obj = Cast(obj, tensor_type);
  }
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(tensor);
  if (inplace && storage->lazy_fusion_data_ != nullptr) {
    // %0 = Mul(p0, p1)
    // %1 = InplaceAddExt(%0, p2, 1)
    // here Inplace op's first input come from another op, should not emit Store for %0,
    // 2 Store with overlapped GM address will cause precision issue in DVM.
    for (int64_t i = static_cast<int64_t>(outputs_.size()) - 1; i >= 0; --i) {
      auto idx = static_cast<size_t>(i);
      if (MatchTensorMeta(outputs_[idx].tensor_meta, tensor)) {
        outputs_[idx].skip = true;
        ASCEND_LOGD("dvm Output: inplace skip %%%zu, addr=%p", idx, tensor.data_ptr());
        break;
      }
    }
  }
  auto &store = outputs_.emplace_back(obj, inplace, tensor);
  CacheTensorMeta(&(store.tensor_meta), tensor);
  ASCEND_LOGD("dvm Output: %%%zu, tensor %p, addr=%p, shape=%s, dtype=%s, inplace=%d",
              outputs_.size() - 1, tensor.unsafeGetTensorImpl(), tensor.data_ptr(),
              ToString(tensor).c_str(), c10::toString(tensor.scalar_type()), inplace);
  if (DumpEnabled()) {
    dump_buf_ << "%" << (outputs_.size() - 1) << ": " << ToString(tensor) << "\n";
  }
  CacheDvmOp(storage, tensor, &(store.tensor_meta), obj, nullptr, false);
}

void *LazyFusionKernel::Alloc(size_t size) {
  static const bool use_workspace_allocator =
      c10_npu::option::OptionsManager::GetTaskQueueEnable() == 2;
  at::Tensor ws_tensor = use_workspace_allocator
      ? at_npu::native::allocate_workspace(size, stream_)
      : at_npu::native::OpPreparation::unsafe_empty_workspace(size);
  void *addr = const_cast<void *>(ws_tensor.storage().data());
  workspace_.emplace_back(std::move(ws_tensor));
  if (DumpEnabled()) {
    dump_buf_ << "workspace: " << addr << " " << size << "\n";
  }
  return addr;
}

void LazyFusionKernel::Flush() {
  if (flushed_) {
    return;
  }
  if (outputs_.empty()) {
    ASCEND_LOGD("dvm Flush: empty graph, skip (id=%zu)", id_);
    Clear();
    return;
  }
  ASCEND_LOGI("dvm Flush: id=%zu, inputs=%zu, outputs=%zu, ops=%zu",
              id_, input_used_, outputs_.size(), dump_ops_.size());
  RECORD_FUNCTION("DvmFlush", {});
  flushed_ = true;
  // Emit Store
  for (auto &out : outputs_) {
    if (out.skip) {
      continue;
    }
    if (!out.storage.lock()) {
      out.skip = true;
      continue;
    }
    auto store = dvm::Kernel::Store(out.tensor_meta.data_ptr, out.op);
    if (out.inplace) {
      dvm::Kernel::SetStoreInplace(store);
    }
  }
  if (DumpEnabled()) {
    DumpGraph();
  }

  static const bool codegen_in_task_queue =
      c10_npu::option::OptionsManager::GetTaskQueueEnable() == 2;
  if (codegen_in_task_queue) {
    // Level 2: codegen + workspace alloc/free + launch all run on the TaskQueue thread.
    ClearGraphRefs();
    at_npu::native::OpCommand::RunOpApiV2("Dvm", [this]() -> int {
      CodeGenAndDump();
      auto ret = Launch();
      workspace_.clear();
      ClearRuntimeState();
      return ret;
    });
    return;
  }
  // Level 1: codegen (which calls Alloc for the N-pad workspace) runs on the caller
  // thread. Run it *before* ClearGraphRefs so the inputs are still alive while Alloc
  // picks a block -- otherwise ClearGraphRefs frees a short-lived input (e.g. a freshly
  // computed grad_output whose only ref is the lazy cache) and Alloc immediately hands
  // that same block back as the workspace; the matmul's AIV pre-pass then copies the
  // input into a buffer that overlaps it -> garbled rhs -> wrong grad -> grad_norm explodes.
  CodeGenAndDump();
  ClearGraphRefs();
  workspace_.clear();
  at_npu::native::OpCommand::RunOpApiV2("Dvm", [this]() -> int {
    auto ret = Launch();
    ClearRuntimeState();
    return ret;
  });
}

void LazyFusionKernel::CodeGenAndDump() {
  if (DumpEnabled()) {
    LazyFusionDump::Instance().DumpGraphInfo(dump_buf_);
    dump_buf_ << "[lazy_fusion before split](" << id() << ", " << this << ") {\n";
    dump_buf_ << Dump() << "}\n";
    LazyFusionDump::Instance().DumpKernelInfo(dump_buf_);
  }
  dvm::Kernel::CodeGen(nullptr, 0, this);
  if (DumpEnabled()) {
    dump_buf_ << "[lazy_fusion after split](" << id() << ", " << this << ") {\n";
    dump_buf_ << Dump() << "}\n";
    dump_buf_ << Das() << "\n";
    LazyFusionDump::Instance().DumpKernelInfo(dump_buf_);
  }
}

int LazyFusionKernel::Launch() {
  ASCEND_LOGI("start dvm Launch: id=%zu", id_);
  if (c10_npu::check_dequeue_need_use(stream_)) {
    NPU_CHECK_ERROR(c10_npu::UseStreamResInCurrentThread(stream_));
  }
  auto ret = dvm::Kernel::Launch(stream_);
  if (g_lazy_fusion_manager.flags_.synchronize) {
    auto err = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream_);
    if (err != ACL_ERROR_NONE) {
      ASCEND_LOGE("SyncStream failed for dvm kernel, id = %zu, error = %d", id_, err);
    }
  }
  ASCEND_LOGI("end dvm Launch: id=%zu, ret=%d", id_, ret);
  return ret;
}

void LazyFusionKernel::ClearRuntimeState() {
  dvm::Kernel::Clear();
  dump_buf_.str("");
  flushed_ = false;
  g_lazy_fusion_manager.FreeKernel(this);
}

std::string LazyFusionKernel::ToString(const at::OptionalIntArrayRef &t) {
  if (t.has_value()) {
    std::stringstream ss;
    ss << t.value();
    return ss.str();
  }
  return "None";
}

std::string LazyFusionKernel::ToString(const at::Scalar &t) {
  auto type = t.type();
  std::string type_str = c10::toString(type);
  if (type == at::ScalarType::Long) {
    return type_str + "(" + std::to_string(t.toLong()) + ")";
  }
  if (type == at::ScalarType::Double) {
    return type_str + "(" + std::to_string(t.toDouble()) + ")";
  }
  return type_str + "(?)";
}

std::string LazyFusionKernel::ToString(const at::Tensor &t) {
  std::stringstream ss;
  ss << "Tensor(shape=" << t.sizes() << " dtype=" << c10::toString(t.scalar_type()) << " ";
  if (t.dim() == 0 && !torch_npu::utils::is_npu(t)) {
    ss << "value=" << ToString(t.item()) << " ";
  }
  ss << "strides=" << t.strides() << " is_contiguous=" << t.is_contiguous();
  ss << " tensor_impl=" << t.unsafeGetTensorImpl();
  auto impl = t.storage().unsafeGetStorageImpl();
  if (impl != nullptr) {
    auto ptr = static_cast<torch_npu::NPUStorageImpl *>(impl);
    ss << " storage=" << impl << " ptr=" << ptr->data() << " offset=" << t.storage_offset() << " addr=" << t.data_ptr();
  }
  ss << ")";
  return ss.str();
}

std::string LazyFusionKernel::ToString(const at::Tensor &t, bool verbose) {
  if (verbose || (t.dim() == 0 && !torch_npu::utils::is_npu(t))) {
    return ToString(t);
  }
  auto impl = t.unsafeGetTensorImpl();
  for (int64_t i = static_cast<int64_t>(input_used_) - 1; i >= 0; --i) {
    if (inputs_[static_cast<size_t>(i)]->tensor.unsafeGetTensorImpl() == impl) {
      return "p" + std::to_string(i);
    }
  }
  for (int64_t i = static_cast<int64_t>(outputs_.size()) - 1; i >= 0; --i) {
    if (outputs_[static_cast<size_t>(i)].tensor_meta.tensor_impl == impl) {
      return "%" + std::to_string(i);
    }
  }
  return ToString(t);
}

void LazyFusionKernel::DumpGraph() {
  dump_buf_ << "lazy_fusion_graph";
  for (const auto &op : dump_ops_) {
    dump_buf_ << "_" << op.name;
  }
  dump_buf_ << "(" << id() << ", " << this << ") {\n";
  size_t output_idx = 0;
  for (const auto &op : dump_ops_) {
    dump_buf_ << "  ";
    for (size_t i = 0; i < op.output_num; ++i) {
      if (i != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << "%" << output_idx;
      output_idx += 1;
    }
    dump_buf_ << " = " << op.name << "(";
    for (size_t i = 0; i < op.inputs.size(); ++i) {
      if (i != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << op.inputs[i];
    }
    dump_buf_ << ")\n";
  }
  dump_buf_ << "  return(";
  output_idx = 0;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (!outputs_[i].skip) {
      if (output_idx != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << "%" << i;
      output_idx += 1;
    }
  }
  dump_buf_ << ")\n}\n";
}

bool IsEnabled(const std::string &op, Level required) {
  const auto &flags = g_lazy_fusion_manager.flags_;
  // Each op declares the minimum level it needs. The user-visible default is
  // kO2 (everything on); internal debug can drop to kO1 via
  // `TORCH_NPU_LAZY_FUSION="level=O1"` to disable the heavier ops
  // (matmul / sum / npu_swiglu / native_batch_norm_backward / ViewLoad).
  if (flags.level < required) {
    return false;
  }
  const auto &disable_ops = flags.disable_ops;
  const auto &enable_ops = flags.enable_ops;
  const auto &enable_ops_only = flags.enable_ops_only;
  bool enable = false;
  if (!enable_ops_only.empty()) {
    enable = std::find(enable_ops_only.begin(), enable_ops_only.end(), op) != enable_ops_only.end();
  } else if (flags.enabled) {
    enable = std::find(disable_ops.begin(), disable_ops.end(), op) == disable_ops.end();
  } else {
    enable = std::find(enable_ops.begin(), enable_ops.end(), op) != enable_ops.end();
  }
  ASCEND_LOGI("op [%s], dvm enabled=%d, level=O%d, op enable=%d",
              op.c_str(), flags.enabled, static_cast<int>(flags.level), enable);
  return enable;
}
}  // namespace lazy_fusion

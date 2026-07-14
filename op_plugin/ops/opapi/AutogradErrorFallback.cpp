 #include <torch/library.h>
  #include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
  #include "op_plugin/utils/Version.h"

  // 以下算子没有反向实现。为它们注册 PyTorch 内置的 autogradNotImplementedFallback：
  // 前向正常执行，只有在对依赖这些算子的张量调用 .backward() 时，才会抛出
  // "derivative for 'npu::xxx' is not implemented" 异常。
  // 该 fallback 是 boxed 内核，对任意算子 schema 通用，无需匹配各自签名。
  TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
      m.impl("npu_add_rms_norm", torch::autograd::autogradNotImplementedFallback());
      m.impl("npu_interleave_rope", torch::autograd::autogradNotImplementedFallback());
      m.impl("npu_moe_gating_top_k_softmax", torch::autograd::autogradNotImplementedFallback());
      // npu_apply_rotary_pos_emb 仅在 v2.7+ 暴露，低版本未注册该算子，需加版本保护。
  #if VERSION_BETWEEN(V2R7, VERSION_NEWEST)
      m.impl("npu_apply_rotary_pos_emb", torch::autograd::autogradNotImplementedFallback());
  #endif
  }

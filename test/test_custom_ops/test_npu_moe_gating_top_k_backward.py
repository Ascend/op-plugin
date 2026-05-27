import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

from typing import Optional
import unittest


EPS: float = 1e-7
GOLDEN_CPU_DTYPE = torch.float64
DEFAULT_PRECISION_LEVEL = "L2"

precision_levels_config = {
    "L0": {"MARE": 10.0, "MERE": 2.0, "RMSE": 2.0},
    "L1": {"MARE": 5.0, "MERE": 1.5, "RMSE": 1.5},
    "L2": {"MARE": 2.0, "MERE": 1.2, "RMSE": 1.2},
}

small_value_config = {
    torch.float16: {"threshold": 2**-11, "error": 2**-16},
    torch.bfloat16: {"threshold": 2**-8, "error": 2**-16},
    torch.float32: {"threshold": 2**-14, "error": 2**-30},
}

error_thd_config = {
    torch.float16: 2 ** -11,  # 4.88e-4
    torch.bfloat16: 2 ** -8,  # 3.91e-3
    torch.float32: 2 ** -14,  # 6.10e-5
}


def absolute_error(actual: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    return torch.abs(actual - golden)


def relative_error(actual: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    return torch.abs(actual - golden) / (torch.abs(golden) + EPS)


def max_relative_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    re: torch.Tensor = relative_error(actual, golden)
    return re.max().item()


def mean_relative_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    re: torch.Tensor = relative_error(actual, golden)
    return re.mean().item()


def root_mean_squared_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    diff: torch.Tensor = actual - golden
    mse: torch.Tensor = torch.mean(diff * diff)
    rmse: torch.Tensor = torch.sqrt(mse)
    return rmse.item()


def calc_error_metrics(actual: torch.Tensor, golden: torch.Tensor) -> dict:
    assert actual.shape == golden.shape
    assert golden.dtype == GOLDEN_CPU_DTYPE

    return {
        "MARE": max_relative_error(actual, golden),
        "MERE": mean_relative_error(actual, golden),
        "RMSE": root_mean_squared_error(actual, golden),
    }


def split_normal_and_small_domain(
    actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
):
    threshold = small_value_config[dtype]["threshold"]

    abs_golden = torch.abs(golden)
    normal_mask = abs_golden >= threshold
    small_mask = abs_golden < threshold

    return (
        actual[normal_mask],
        golden[normal_mask],
        actual[small_mask],
        golden[small_mask],
    )


def normal_domain_pass(
    npu_normal: torch.Tensor,
    gpu_normal: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
    precision_level: str = DEFAULT_PRECISION_LEVEL,
) -> bool:
    npu_metrics = calc_error_metrics(npu_normal, golden)
    gpu_metrics = calc_error_metrics(gpu_normal, golden)

    err_thd = error_thd_config[dtype]

    #ratio = {k: npu_metrics[k] / max(gpu_metrics[k] + 1e-7, err_thd) for k in npu_metrics}
    ratio = {k: npu_metrics[k] / (gpu_metrics[k] + 1e-7) for k in npu_metrics}

    limits = precision_levels_config[precision_level]

    print(f"[NormalDomain] ratio: {ratio}")
    print(f"[NormalDomain] required level: {precision_level}")

    for k, limit in limits.items():
        if ratio[k] > limit:
            print(f"[NormalDomain] FAIL on {k}: {ratio[k]} > {limit}")
            return False

    print("[NormalDomain] PASS")
    return True


def small_domain_pass(
    npu_actual: torch.Tensor,
    gpu_actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
) -> bool:
    cfg = small_value_config[dtype]
    threshold = cfg["threshold"]
    err = cfg["error"]

    def error_count(actual):
        mask = (torch.abs(golden) < threshold) & (torch.abs(actual - golden) > err)
        return int(mask.sum().item())

    npu_cnt = error_count(npu_actual)
    gpu_cnt = error_count(gpu_actual)

    ratio = npu_cnt / max(gpu_cnt, 1)

    print(
        f"[SmallDomain] NPU ErrorCount={npu_cnt}, "
        f"GPU ErrorCount={gpu_cnt}, Ratio={ratio:.3f}"
    )

    return ratio <= 2.0


def precision_check(
    npu_out: torch.Tensor,
    gpu_out: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
    precision_level: str = DEFAULT_PRECISION_LEVEL,
) -> bool:
    (npu_normal, golden_normal, npu_small, golden_small) = (
        split_normal_and_small_domain(npu_out, golden, dtype)
    )

    (gpu_normal, _, gpu_small, _) = split_normal_and_small_domain(
        gpu_out, golden, dtype
    )

    normal_ok = True
    if golden_normal.numel() > 0:
        normal_ok = normal_domain_pass(
            npu_normal, gpu_normal, golden_normal, dtype, precision_level
        )

    small_ok = True
    if golden_small.numel() > 0:
        small_ok = small_domain_pass(npu_small, gpu_small, golden_small, dtype)

    print(f"[Final] normal_pass={normal_ok}, small_pass={small_ok}")
    return normal_ok and small_ok


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    norm_topk_prob: bool = False,
):
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk):
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            if expert_bias is not None:
                scores += expert_bias
            probs, top_indices = compute_topk(scores, topk)
        else:
            if expert_bias is not None:
                logits += expert_bias
            scores, top_indices = compute_topk(logits, topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk)
        probs = scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if topk > 1 and norm_topk_prob:
        denominator = probs.sum(dim=-1, keepdim=True) + 1e-20
        probs = probs / denominator

    if scaling_factor:
        probs = probs * scaling_factor

    probs = probs.type_as(logits)
    return probs, top_indices


# ---------- test cases ----------

class TestNpuMoeGatingTopKBackward(TestCase):
    def _run_backward_test(self, M, N, K, grad_y_dtype,
                           routed_scaling_factor=1.0, eps=1e-20, seed=44):
        torch.manual_seed(seed)

        # 0. Generate base deterministic data on CPU
        logits_base = torch.randn(M, N, dtype=torch.float32)
        bias_base = torch.randn(N, dtype=torch.float32) * 0.1
        grad_output_base = torch.randn(M, K, dtype=torch.float32)

        # =====================================================================
        # 1. NPU Native Autograd (gpu_golden)
        # =====================================================================
        logits_gpu = logits_base.to(grad_y_dtype).npu().requires_grad_(True)
        bias_gpu = bias_base.to(grad_y_dtype).npu()
        
        y_gpu, idx_gpu = topk_routing_with_score_function(
            logits_gpu, 
            topk=K, 
            score_function="sigmoid", 
            expert_bias=bias_gpu,
            norm_topk_prob=True, 
            scaling_factor=routed_scaling_factor
        )
        
        grad_out_gpu = grad_output_base.to(grad_y_dtype).npu()
        y_gpu.backward(grad_out_gpu)
        gpu_golden = logits_gpu.grad.detach().clone()

        # =====================================================================
        # 2. CPU FP64 Native Autograd (cpu_golden)
        # =====================================================================
        logits_cpu = logits_base.to(torch.float64).requires_grad_(True)
        bias_cpu = bias_base.to(torch.float64)
        
        y_cpu, _ = topk_routing_with_score_function(
            logits_cpu, 
            topk=K, 
            score_function="sigmoid", 
            expert_bias=bias_cpu,
            norm_topk_prob=True, 
            scaling_factor=routed_scaling_factor
        )
        
        grad_out_cpu = grad_output_base.to(torch.float64)
        y_cpu.backward(grad_out_cpu)
        cpu_golden = logits_cpu.grad.detach().clone()

        # =====================================================================
        # 3. Custom Ascend C Operator (npu_result)
        # =====================================================================
        # Manually compute unnormalized sigmoid in fp32 for the custom operator
        x_norm_npu = torch.sigmoid(logits_base.to(torch.float32)).npu()
        
        idx_npu = idx_gpu.detach().to(torch.int32)
        grad_out_npu = grad_out_gpu.detach()
        
        npu_result = torch_npu.npu_moe_gating_top_k_backward(
            x_norm_npu,
            grad_out_npu,
            idx_npu,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
        )
        torch.npu.synchronize()

        # =====================================================================
        # 4. Compare Results
        # =====================================================================
        # Move everything to CPU and cast to FP64 for the precision check
        npu_out_fp64 = npu_result.cpu().to(torch.float64)
        gpu_out_fp64 = gpu_golden.cpu().to(torch.float64)
        
        ok = precision_check(npu_out_fp64, gpu_out_fp64, cpu_golden, grad_y_dtype)
        self.assertTrue(ok, "Precision check failed")


    @unittest.skip("Temporarily skipping")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_gating_top_k_backward_bf16_2048_192_10(self):
        self._run_backward_test(2048, 192, 10, torch.bfloat16)


if __name__ == "__main__":
    run_tests()

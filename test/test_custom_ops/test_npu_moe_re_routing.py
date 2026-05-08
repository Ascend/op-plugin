import importlib.util
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def numpy_hifloat8():
    try:
        from en_dtypes import hifloat8
        return hifloat8
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support hifloat8 dtype!!! "
                        "Please install with `pip3 install en-dtypes`")
    except ImportError:
        raise RuntimeError("Please upgrade en_dtypes to v0.0.3 at least to support hifloat8 dtype!!! "
                        "Command is `pip3 install --upgrade en-dtypes`")


def numpy_float4_e2m1fn():
    try:
        from ml_dtypes import float4_e2m1fn
        return float4_e2m1fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float4_e2m1fn dtype!!! "
                        "Please install with `pip3 install ml-dtypes`")
    except ImportError:
        raise RuntimeError("Please upgrade ml_dtypes to support float4_e2m1fn dtype!!! "
                        "Command is `pip3 install --upgrade ml-dtypes`")


def numpy_to_torch(np_arr):
    FP8_DTYPE_MAP_NUMPY_TO_TORCH = {
        "bfloat16": torch.bfloat16,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e8m0": None if not hasattr(torch, "float8_e8m0fnu") else getattr(torch, "float8_e8m0fnu"),
        "hifloat8": torch_npu.hifloat8,
    }

    def _bitcast_float8_to_torch(np_arr):
        np_dtype = np_arr.dtype
        torch_dtype = FP8_DTYPE_MAP_NUMPY_TO_TORCH[str(np_dtype)]
        np_uint8 = np_arr.view(np.uint8)
        t_uint8 = torch.from_numpy(np_uint8)
        return t_uint8.view(torch_dtype)
    
    if str(np_arr.dtype) in list(FP8_DTYPE_MAP_NUMPY_TO_TORCH.keys()):
        return _bitcast_float8_to_torch(np_arr)
    return torch.from_numpy(np_arr)


def assert_tensors_close(x: torch.Tensor, y: torch.Tensor, rtol=1e-3, atol=1e-5, label="Tensor"):
    if x.device.type != "cpu":
        x = x.cpu()
    if y.device.type != "cpu":
        y = y.cpu()

    x_f32 = x.to(torch.float32)
    y_f32 = y.to(torch.float32)

    nan_mask_x = torch.isnan(x_f32)
    nan_mask_y = torch.isnan(y_f32)
    if not torch.equal(nan_mask_x, nan_mask_y):
        raise AssertionError(
            f"[{label}] NaN mismatch: x has NaNs at {torch.where(nan_mask_x)}, y has NaNs at {torch.where(nan_mask_y)}")

    valid_mask = ~nan_mask_x
    x_valid = x_f32[valid_mask]
    y_valid = y_f32[valid_mask]

    inf_mask_x = torch.isinf(x_valid)
    inf_mask_y = torch.isinf(y_valid)
    if not torch.equal(inf_mask_x, inf_mask_y):
        raise AssertionError(f"[{label}] Inf mismatch.")

    valid_mask_no_inf = ~inf_mask_x
    x_final = x_valid[valid_mask_no_inf]
    y_final = y_valid[valid_mask_no_inf]

    if x_final.numel() == 0:
        return

    diff = torch.abs(x_final - y_final)
    tolerance = atol + (rtol * torch.abs(y_final))

    failure_mask = diff > tolerance

    if torch.any(failure_mask):
        max_diff = diff.max().item()
        max_idx = torch.argmax(diff).item()

        y_safe = y_final.clone()
        y_safe[y_safe == 0] = 1e-12
        rel_error = (diff / torch.abs(y_safe)).max().item()

        raise AssertionError(
            f"[{label}] Tensors not close!\n"
            f"  Max absolute diff: {max_diff:.6e} at index {max_idx}\n"
            f"  Max relative error: {rel_error:.6e}\n"
            f"  Tolerance: atol={atol}, rtol={rtol}\n"
            f"  Shape: {x.shape}")


class TestMoeReRouting(TestCase):

    def generate_inputs(self, bs, hidden_dim, dtype, expert_num=16, rank_num=2):
        expert_token_num_per_rank = np.zeros((rank_num, expert_num), dtype=np.int64)
        for i in range(rank_num):
            for j in range(expert_num):
                expert_token_num_per_rank[i, j] = (bs // expert_num) + np.random.randint(0, 5)
        
        if dtype == torch_npu.float4_e2m1fn_x2:
            fp4_dtype = numpy_float4_e2m1fn()
            tokens = np.random.randn(bs, hidden_dim // 2).astype(fp4_dtype)
        elif dtype == torch_npu.hifloat8:
            tokens = np.random.randn(bs, hidden_dim).astype(np.float16)
            tokens = tokens.astype(numpy_hifloat8())
        else:
            tokens = np.random.randn(bs, hidden_dim).astype(np.float16)
        
        return tokens, expert_token_num_per_rank

    def custom_op_exec(self, tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu, 
                        expert_token_num_type, idx_type, tokens_dtype=None):
        return torch_npu.npu_moe_re_routing(
            tokens_npu,
            expert_token_num_per_rank_npu,
            per_token_scales=per_token_scales_npu,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type,
            tokens_dtype=tokens_dtype
        )

    def golden_calc(self, tokens, expert_token_num_per_rank, expert_token_num_type=1):
        bs = tokens.shape[0]
        hidden_dim = tokens.shape[1]
        rank_num = expert_token_num_per_rank.shape[0]
        expert_num = expert_token_num_per_rank.shape[1]
        
        permute_tokens = np.zeros((bs, hidden_dim), dtype=tokens.dtype)
        permute_per_token_scales = np.ones((bs,), dtype=np.float32)
        permute_token_idx = np.arange(bs, dtype=np.int32)
        
        if str(tokens.dtype) == "hifloat8":
            permute_tokens = tokens.copy()
        else:
            permute_tokens = tokens.copy()
        
        expert_token_num = expert_token_num_per_rank[0, :expert_num].copy()
        
        return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_re_routing_fp16(self, device="npu"):
        bs_list = [32, 128]
        hidden_dim_list = [4096]
        dtype_list = [torch.float16]
        
        for bs, hidden_dim, dtype in zip(bs_list, hidden_dim_list, dtype_list):
            tokens, expert_token_num_per_rank = self.generate_inputs(bs, hidden_dim, dtype)
            tokens_npu = torch.from_numpy(tokens).npu()
            expert_token_num_per_rank_npu = torch.from_numpy(expert_token_num_per_rank).npu()
            
            permute_tokens, permute_scales, permute_idx, expert_token_num = \
                self.custom_op_exec(tokens_npu, expert_token_num_per_rank_npu, None, 1, 0)
            
            golden_tokens, golden_scales, golden_idx, golden_expert_num = \
                self.golden_calc(tokens, expert_token_num_per_rank)
            
            self.assertEqual(permute_tokens.shape[0], bs)
            self.assertEqual(permute_tokens.shape[1], hidden_dim)

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_re_routing_bf16(self, device="npu"):
        bs_list = [32]
        hidden_dim_list = [4096]
        dtype_list = [torch.bfloat16]
        
        for bs, hidden_dim, dtype in zip(bs_list, hidden_dim_list, dtype_list):
            tokens, expert_token_num_per_rank = self.generate_inputs(bs, hidden_dim, dtype)
            tokens_npu = torch.from_numpy(tokens).npu()
            expert_token_num_per_rank_npu = torch.from_numpy(expert_token_num_per_rank).npu()
            
            permute_tokens, permute_scales, permute_idx, expert_token_num = \
                self.custom_op_exec(tokens_npu, expert_token_num_per_rank_npu, None, 1, 0)
            
            self.assertEqual(permute_tokens.shape[0], bs)
            self.assertEqual(permute_tokens.shape[1], hidden_dim)

    @unittest.skipIf(
        importlib.util.find_spec("en_dtypes") is None,
        "Unittest for hif8 need package en_dtypes"
    )
    @SupportedDevices(['Ascend950'])
    def test_npu_moe_re_routing_hif8(self, device="npu"):
        bs_list = [32, 128]
        hidden_dim_list = [4096, 7168]
        expert_token_num_type_list = [1]
        idx_type_list = [0, 1]
        
        for bs, hidden_dim, expert_token_num_type, idx_type in zip(
                bs_list, hidden_dim_list, expert_token_num_type_list, idx_type_list):
            tokens, expert_token_num_per_rank = self.generate_inputs(bs, hidden_dim, torch_npu.hifloat8)
            
            tokens_uint8 = tokens.view(np.uint8)
            tokens_npu = torch.from_numpy(tokens_uint8).npu()
            expert_token_num_per_rank_npu = torch.from_numpy(expert_token_num_per_rank).npu()
            
            permute_tokens, permute_scales, permute_idx, expert_token_num = \
                self.custom_op_exec(tokens_npu, expert_token_num_per_rank_npu, None, 
                                    expert_token_num_type, idx_type, tokens_dtype=290)
            
            self.assertEqual(permute_tokens.dtype, torch.uint8)
            self.assertEqual(permute_tokens.shape[0], bs)
            self.assertEqual(permute_tokens.shape[1], hidden_dim)
            
            golden_tokens, golden_scales, golden_idx, golden_expert_num = \
                self.golden_calc(tokens, expert_token_num_per_rank, expert_token_num_type)
            
            permute_tokens_cpu = permute_tokens.cpu()
            golden_tokens_torch = numpy_to_torch(golden_tokens)
            
            assert_tensors_close(permute_tokens_cpu, golden_tokens_torch, rtol=1e-2, atol=1e-2, 
                                 label=f"hif8_tokens_bs={bs}_h={hidden_dim}")

    @unittest.skipIf(
        importlib.util.find_spec("en_dtypes") is None,
        "Unittest for hif8 need package en_dtypes"
    )
    @SupportedDevices(['Ascend950'])
    def test_npu_moe_re_routing_hif8_with_scales(self, device="npu"):
        bs_list = [32]
        hidden_dim_list = [4096]
        expert_token_num_type = 1
        idx_type = 0
        
        for bs, hidden_dim in zip(bs_list, hidden_dim_list):
            tokens, expert_token_num_per_rank = self.generate_inputs(bs, hidden_dim, torch_npu.hifloat8)
            
            per_token_scales = np.random.randn(bs).astype(np.float32)
            
            tokens_uint8 = tokens.view(np.uint8)
            tokens_npu = torch.from_numpy(tokens_uint8).npu()
            expert_token_num_per_rank_npu = torch.from_numpy(expert_token_num_per_rank).npu()
            per_token_scales_npu = torch.from_numpy(per_token_scales).npu()
            
            permute_tokens, permute_scales, permute_idx, expert_token_num = \
                self.custom_op_exec(tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu,
                                    expert_token_num_type, idx_type, tokens_dtype=290)
            
            self.assertEqual(permute_tokens.dtype, torch.uint8)
            self.assertEqual(permute_scales.dtype, torch.float32)
            self.assertEqual(permute_idx.dtype, torch.int32)
            
            self.assertEqual(permute_tokens.shape[0], bs)
            self.assertEqual(permute_tokens.shape[1], hidden_dim)

    @unittest.skipIf(
        importlib.util.find_spec("ml_dtypes") is None,
        "Unittest for fp4 need package ml_dtypes"
    )
    @SupportedDevices(['Ascend950'])
    def test_npu_moe_re_routing_fp4_e2m1(self, device="npu"):
        bs_list = [32, 128]
        hidden_dim_list = [4096, 7168]
        expert_token_num_type = 1
        idx_type = 0
        
        for bs, hidden_dim in zip(bs_list, hidden_dim_list):
            tokens, expert_token_num_per_rank = self.generate_inputs(bs, hidden_dim, torch_npu.float4_e2m1fn_x2)
            
            tokens_uint8 = tokens.view(np.uint8)
            tokens_npu = torch.from_numpy(tokens_uint8).npu()
            expert_token_num_per_rank_npu = torch.from_numpy(expert_token_num_per_rank).npu()
            
            permute_tokens, permute_scales, permute_idx, expert_token_num = \
                self.custom_op_exec(tokens_npu, expert_token_num_per_rank_npu, None,
                                    expert_token_num_type, idx_type, tokens_dtype=296)
            
            self.assertEqual(permute_tokens.dtype, torch.uint8)
            self.assertEqual(permute_tokens.shape[0], bs)
            self.assertEqual(permute_tokens.shape[1], hidden_dim // 2)

    @unittest.skipIf(
        importlib.util.find_spec("ml_dtypes") is None,
        "Unittest for fp4 need package ml_dtypes"
    )
    @SupportedDevices(['Ascend950'])
    def test_npu_moe_re_routing_fp4_e2m1_with_scales(self, device="npu"):
        bs_list = [32]
        hidden_dim_list = [4096]
        expert_token_num_type = 1
        idx_type = 0
        
        for bs, hidden_dim in zip(bs_list, hidden_dim_list):
            tokens, expert_token_num_per_rank = self.generate_inputs(bs, hidden_dim, torch_npu.float4_e2m1fn_x2)
            
            per_token_scales = np.random.randn(bs).astype(np.float32)
            
            tokens_uint8 = tokens.view(np.uint8)
            tokens_npu = torch.from_numpy(tokens_uint8).npu()
            expert_token_num_per_rank_npu = torch.from_numpy(expert_token_num_per_rank).npu()
            per_token_scales_npu = torch.from_numpy(per_token_scales).npu()
            
            permute_tokens, permute_scales, permute_idx, expert_token_num = \
                self.custom_op_exec(tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu,
                                    expert_token_num_type, idx_type, tokens_dtype=296)
            
            self.assertEqual(permute_tokens.dtype, torch.uint8)
            self.assertEqual(permute_scales.dtype, torch.float32)
            self.assertEqual(permute_idx.dtype, torch.int32)
            
            self.assertEqual(permute_tokens.shape[0], bs)
            self.assertEqual(permute_tokens.shape[1], hidden_dim // 2)


if __name__ == "__main__":
    run_tests()
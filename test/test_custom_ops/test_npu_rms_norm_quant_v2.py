import unittest
import math

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

DST_TYPE_MAP = {1: "int8", 16: "int4", 290: "hifloat8", 291: "float8_e5m2", 292: "float8_e3m4fn"}

class TestNPURmsNormQuant(TestCase):
    def numpy_float8_e4m3fn(self):
        try:
            from ml_dtypes import float8_e4m3fn
            return float8_e4m3fn
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_hifloat8(self):
        try:
            from en_dtypes import hifloat8
            return hifloat8
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support hifloat8 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")
        except ImportError:
            raise RuntimeError("Please upgrade en_dtypes to v0.0.3 at least to support hifloat8 dtype!!! "
                               "Command is `pip3 install --upgrade en-dtypes`")

    def numpy_float8_e5m2(self):
        try:
            from ml_dtypes import float8_e5m2
            return float8_e5m2
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def quant_process(self, y_quant, y_type):
        y_quant = y_quant if y_type in ("hifloat8", "float8_e5m2", "float8_e4m3fn") else np.round(y_quant, 0)
        if y_type == "int8":
            y_quant = np.clip(y_quant, -128, 127)
            y_quant = y_quant.astype("int8", copy=False)
        elif y_type == "float8_e5m2":
            y_quant = y_quant.astype(self.numpy_float8_e5m2(), copy=False)
        elif y_type == "float8_e4m3fn":
            y_quant = y_quant.astype(self.numpy_float8_e4m3fn(), copy=False)
        elif y_type == "hifloat8":
            y_quant = y_quant.astype(self.numpy_hifloat8(), copy=False)

        return y_quant

    def pack_ml_int4_to_int32(self, int4_array):
        bit8_data = np.asarray(int4_array, dtype=np.uint8)

        reshaped = bit8_data.reshape(-1, 8).astype(np.uint32)
        packed = np.zeros(reshaped.shape[0], dtype=np.uint32)
        for i in range(8):
            shift = 4 * i
            val = reshaped[:, i] & 0xF
            packed |= (val << shift).astype(np.uint32)
        return packed.astype(np.int32)


    def compare(self, a, b, benchmark):

        diff_abs = torch.abs(a - b)
        max_diff_abs, _ = torch.max(diff_abs, dim=0)

        if max_diff_abs.item() < benchmark:
            return True
        else:
            rel_error = 0
            abs_error = 0
            for i in range(a.shape[0]):
                yes_no = (a[i] == 0 and b[i].item() != 0)
                no_yes = (a[i] != 0 and b[i].item() == 0)
                if a[i].item() == 0 and b[i].item() == 0:
                    diff_rel_item = 0
                elif yes_no or no_yes:
                    diff_rel_item = 1
                elif a[i] != 0 and b[i].item() != 0:
                    diff_rel_item = diff_abs[i].item() / abs(a[i].item())

                if abs(a[i].item()) < 1 and diff_abs[i].item() > benchmark:
                    abs_error += 1
                elif abs(a[i].item()) >= 1 and diff_rel_item > benchmark:
                    rel_error += 1
                if (rel_error + abs_error) > 10:
                    break
            if (rel_error + abs_error) > 0:
                return False
            else:
                return True

    def npu_rms_norm_quant_v2_golden(self, x, gamma, scale,
                                    offset, beta, epsilon=1e-06, div_mode=True, dst_dtype=1):

        x_fp32 = x.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32).reshape(-1)
        scale_fp32 = scale.to(torch.float32).expand(gamma_fp32.shape)

        offset_fp32 = offset.to(torch.float32).expand(gamma_fp32.shape) if offset is not None else None
        beta_fp32 = beta.to(torch.float32).reshape(-1) if beta is not None else None

        len_shape_x = len(x_fp32.shape)
        len_shape_gamma = len(gamma.shape)
        axis = len_shape_x - len_shape_gamma
        variance = torch.mean(torch.pow(x_fp32, 2), axis=axis, keepdims=True)
        std = torch.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd
        y_array = result_mid * gamma_fp32 + beta_fp32 if beta_fp32 is not None else result_mid * gamma_fp32
        y = y_array.type(torch.float32)

        if div_mode:
            y1 = y / scale_fp32
            y1 = y1 + offset_fp32 if offset_fp32 is not None else y1
        else:
            y1 = y * scale_fp32
            y1 = y1 + offset_fp32 if offset_fp32 is not None else y1

        y1_np = y1.cpu().numpy()
        dst_type = DST_TYPE_MAP[dst_dtype]

        y1_np = self.quant_process(y1_np, dst_type)
        if dst_type == "int4":
            packed = self.pack_ml_int4_to_int32(y1_np)
            packed = packed.reshape(*y.shape[:-1], y.shape[-1] // 8)
            y1_np = packed
        y = torch.tensor(y1_np.reshape(x.shape).astype(np.float32))
        return y, rstd


    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_quant_int8(self):
        x_shape = [4, 32]
        quant_shape = [32]
        x = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        scale = (torch.rand(1, dtype=torch.float16) * 0.8 + 0.2)  # [0.2, 1.0)
        offset = torch.randint(-5, 6, (1,), dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        eps = 1e-6
        div_mode = True
        dst_dtype = 1
        output_rstd = True

        x_npu = x.npu().requires_grad_(output_rstd)
        gamma_npu = gamma.npu()
        scale_npu = scale.npu()
        offset_npu = offset.npu()
        beta_npu = beta.npu()

        y_ref, rstd_ref = self.npu_rms_norm_quant_v2_golden(x, gamma, scale, offset, beta, epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)

        y_npu, rstd_npu = torch_npu.npu_rms_norm_quant_v2(x_npu, gamma_npu, scale_npu, offset=offset_npu, beta=beta_npu,
                                                epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)
        if dst_dtype == 290:
            y_npu_flat = torch_npu.npu_dtype_cast(y_npu, torch.float32, torch_npu.hifloat8)
        else:
            y_npu_flat = y_npu.to(torch.float32)
        benchmark_float32 = 1e-6

        y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
        y_npu_flat = y_npu_flat.to(torch.float32).reshape(1, y_npu.numel())[0].cpu()
        rstd_ref_flat = rstd_ref.cpu()
        rstd_npu_flat = rstd_npu.cpu()

        self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_float32))
        self.assertTrue(self.compare(rstd_ref_flat, rstd_npu_flat, benchmark_float32))

    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_quant_hifloat8(self):
        x_shape = [4, 32]
        quant_shape = [32]
        x = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        scale = (torch.rand(1, dtype=torch.float16) * 0.8 + 0.2)  # [0.2, 1.0)
        offset = torch.randint(-5, 6, (1,), dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        eps = 1e-6
        div_mode = True
        dst_dtype = 290
        output_rstd = True

        x_npu = x.npu().requires_grad_(output_rstd)
        gamma_npu = gamma.npu()
        scale_npu = scale.npu()
        offset_npu = offset.npu()
        beta_npu = beta.npu()

        y_ref, rstd_ref = self.npu_rms_norm_quant_v2_golden(x, gamma, scale, offset, beta, epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)

        y_npu, rstd_npu = torch_npu.npu_rms_norm_quant_v2(x_npu, gamma_npu, scale_npu, offset=offset_npu, beta=beta_npu,
                                                epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)
        if dst_dtype == 290:
            y_npu_flat = torch_npu.npu_dtype_cast(y_npu, torch.float32, torch_npu.hifloat8)
        else:
            y_npu_flat = y_npu.to(torch.float32)
        benchmark_float32 = 1e-6

        y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
        y_npu_flat = y_npu_flat.to(torch.float32).reshape(1, y_npu.numel())[0].cpu()
        rstd_ref_flat = rstd_ref.cpu()
        rstd_npu_flat = rstd_npu.cpu()

        self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_float32))
        self.assertTrue(self.compare(rstd_ref_flat, rstd_npu_flat, benchmark_float32))


    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_quant_float8_e5m2(self):
        x_shape = [4, 32]
        quant_shape = [32]
        x = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        scale = (torch.rand(1, dtype=torch.float16) * 0.8 + 0.2)  # [0.2, 1.0)
        offset = torch.randint(-5, 6, (1,), dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        eps = 1e-6
        div_mode = True
        dst_dtype = 291
        output_rstd = True

        x_npu = x.npu().requires_grad_(output_rstd)
        gamma_npu = gamma.npu()
        scale_npu = scale.npu()
        offset_npu = offset.npu()
        beta_npu = beta.npu()

        y_ref, rstd_ref = self.npu_rms_norm_quant_v2_golden(x, gamma, scale, offset, beta, epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)

        y_npu, rstd_npu = torch_npu.npu_rms_norm_quant_v2(x_npu, gamma_npu, scale_npu, offset=offset_npu, beta=beta_npu,
                                                epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)
        if dst_dtype == 290:
            y_npu_flat = torch_npu.npu_dtype_cast(y_npu, torch.float32, torch_npu.hifloat8)
        else:
            y_npu_flat = y_npu.to(torch.float32)
        benchmark_float32 = 1e-6

        y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
        y_npu_flat = y_npu_flat.to(torch.float32).reshape(1, y_npu.numel())[0].cpu()
        rstd_ref_flat = rstd_ref.cpu()
        rstd_npu_flat = rstd_npu.cpu()

        self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_float32))
        self.assertTrue(self.compare(rstd_ref_flat, rstd_npu_flat, benchmark_float32))

    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_quant_gamma_multi_float8_e5m2(self):
        x_shape = [4, 32]
        quant_shape = [1, 32]
        x = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        scale = (torch.rand(1, dtype=torch.float16) * 0.8 + 0.2)  # [0.2, 1.0)
        offset = torch.randint(-5, 6, (1,), dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        eps = 1e-6
        div_mode = True
        dst_dtype = 1
        output_rstd = True

        x_npu = x.npu().requires_grad_(output_rstd)
        gamma_npu = gamma.npu()
        scale_npu = scale.npu()
        offset_npu = offset.npu()
        beta_npu = beta.npu()

        y_ref, rstd_ref = self.npu_rms_norm_quant_v2_golden(x, gamma, scale, offset, beta, epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)

        y_npu, rstd_npu = torch_npu.npu_rms_norm_quant_v2(x_npu, gamma_npu, scale_npu, offset=offset_npu, beta=beta_npu,
                                                epsilon=eps, div_mode=div_mode, dst_dtype=dst_dtype)
        if dst_dtype == 290:
            y_npu_flat = torch_npu.npu_dtype_cast(y_npu, torch.float32, torch_npu.hifloat8)
        else:
            y_npu_flat = y_npu.to(torch.float32)
        benchmark_float32 = 1e-6

        y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
        y_npu_flat = y_npu_flat.to(torch.float32).reshape(1, y_npu.numel())[0].cpu()
        rstd_ref_flat = rstd_ref.cpu()
        rstd_npu_flat = rstd_npu.cpu()

        self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_float32))
        self.assertTrue(self.compare(rstd_ref_flat, rstd_npu_flat, benchmark_float32))

if __name__ == "__main__":
    run_tests()
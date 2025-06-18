import unittest
import math

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUAddRmsNormQuant(TestCase):

    def compare(self, a, b, benchmark):
        diff_abs = torch.abs(a - b)
        max_diff_abs, _ = torch.max(diff_abs, dim=0)

        if max_diff_abs.item() == 0:
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

    # pylint:disable = huawei-too-many-arguments
    def npu_add_rms_norm_quant_golden(self, input_x1, input_x2, input_gamma, input_scales1,
                                      input_zero_points1, input_scales2=None, input_zero_points2=None, epsilon=1e-06):
        torchType32 = torch.float32
        len_shape_x = len(input_x1.shape)
        len_shape_gamma = len(input_gamma.shape)
        axis = len_shape_x - len_shape_gamma
        divMode = True

        n = len(input_x1.shape) - len(input_gamma.shape)
        input_gamma = input_gamma.reshape(np.multiply.reduce(np.array(input_gamma.shape)))

        input_scales1 = input_scales1.reshape(np.multiply.reduce(np.array(input_scales1.shape)))
        if input_scales2 is not None:
            input_scales2 = input_scales2.reshape(np.multiply.reduce(np.array(input_scales2.shape)))
        if input_zero_points1 is not None:
            input_zero_points1 = input_zero_points1.reshape(np.multiply.reduce(np.array(input_zero_points1.shape)))
        if input_zero_points2 is not None:
            input_zero_points2 = input_zero_points2.reshape(np.multiply.reduce(np.array(input_zero_points2.shape)))

        x1_shape = input_x1.shape[0:n] + input_gamma.shape
        input_x1 = input_x1.reshape(x1_shape)
        input_x2 = input_x2.reshape(x1_shape)
        len_shape_x = len(input_x1.shape)
        len_shape_gamma = len(input_gamma.shape)
        axis = len_shape_x - len_shape_gamma
        input_x_dtype = input_x1.dtype
        input_scales_dtype = input_scales1.dtype
        if input_zero_points1 is not None:
            input_zp_dtype = input_zero_points1.dtype
        if input_zero_points2 is not None:
            input_zp_dtype = input_zero_points2.dtype

        if (input_x_dtype == torchType32):
            add_x = input_x1 + input_x2
        elif (input_x_dtype == torch.float16):
            add_x = (input_x1.type(torchType32) + input_x2.type(torchType32))
        else:
            add_x = (input_x1.type(torchType32) + input_x2.type(torchType32))

        if input_scales_dtype is not torchType32:
            input_scales1 = input_scales1.type(torchType32)
        if input_scales2 is not None:
            if input_scales_dtype is not torchType32:
                input_scales2 = input_scales2.type(torchType32)
        if input_zero_points1 is not None:
            if input_zp_dtype == torch.int32:
                input_zero_points1 = input_zero_points1.type(torchType32)
            elif input_scales_dtype is not torchType32:
                input_zero_points1 = input_zero_points1.type(torchType32)
        if input_zero_points2 is not None:
            if input_zp_dtype == torch.int32:
                input_zero_points2 = input_zero_points2.type(torchType32)
            elif input_scales_dtype is not torchType32:
                input_zero_points2 = input_zero_points2.type(torchType32)

        x_fp32 = add_x.type(torchType32)
        variance = torch.mean(torch.pow(x_fp32, 2), axis=axis, keepdims=True)
        std = torch.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd

        if input_x_dtype == torchType32:
            y_array = result_mid * input_gamma
        elif input_x_dtype == torch.float16:
            input_gamma_fp32 = input_gamma.type(torchType32)
            y_array = result_mid * input_gamma_fp32
        else:
            input_gamma_fp32 = input_gamma.type(torchType32)
            y_array = result_mid * input_gamma_fp32

        tensor_scales1 = input_scales1.to(torch.float32)

        if input_zero_points1 is None:
            tensor_zero_points1 = torch.zeros(input_scales1.shape, dtype=torch.float32)
        else:
            tensor_zero_points1 = input_zero_points1
        if input_scales2 is None:
            tensor_scales2 = torch.ones(input_scales1.shape, dtype=torch.float32)
        else:
            tensor_scales2 = input_scales2
        if input_zero_points2 is None:
            tensor_zero_points2 = torch.zeros(input_scales1.shape, dtype=torch.float32)
        else:
            tensor_zero_points2 = input_zero_points2

        if not divMode:
            tensor_scales1 = 1.0 / tensor_scales1
            if input_scales2 is not None:
                tensor_scales2 = 1.0 / tensor_scales2

        y = y_array.type(torch.float32)
        y1 = torch.quantize_per_channel(y, tensor_scales1, tensor_zero_points1, axis, torch.qint8)
        y1_np = y1.int_repr().detach().clone().cpu().numpy()

        y2 = torch.quantize_per_channel(y, tensor_scales2, tensor_zero_points2, axis, torch.qint8)
        y2_np = y2.int_repr().detach().clone().cpu().numpy()

        return torch.tensor(y1_np).type(torch.int8), torch.tensor(y2_np).type(torch.int8), \
            add_x.type(input_x_dtype).reshape(input_x1.shape)

    @unittest.skip("Skip test_npu_add_rms_norm_quant due to low version of cann")
    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_quant(self):
        shape_list = [[[16, ], [16, ]],
                      [[2, 16], [16, ]],
                      [[2, 16], [2, 16]],
                      [[16, 32], [16, 32]],
                      [[16, 32], [32, ]]]
        for item in shape_list:
            x_shape = item[0]
            quant_shape = item[1]
            x1 = torch.randn(x_shape, dtype=torch.float16)
            x2 = torch.randn(x_shape, dtype=torch.float16)
            gamma = torch.randn(quant_shape, dtype=torch.float16)
            scales1 = torch.randn(quant_shape, dtype=torch.float32)
            zero_points1 = torch.randint(-10, 10, quant_shape, dtype=torch.int32)

            x1_npu = x1.npu()
            x2_npu = x2.npu()
            gamma_npu = gamma.npu()
            scales1_npu = scales1.npu()
            zero_points1_npu = zero_points1.npu()

            y1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
            y1_cpu, _, x_out_cpu = self.npu_add_rms_norm_quant_golden(x1, x2, gamma, scales1, zero_points1)

            benchmark = math.pow(2, -7)
            benchmark_int8 = 1
            y1_cpu_data = y1_cpu.reshape(1, y1_cpu.numel())[0].cpu()
            y1_npu_data = y1.reshape(1, y1.numel())[0].cpu()
            self.assertTrue(self.compare(y1_cpu_data, y1_npu_data, benchmark_int8))

            x_cpu_data = x_out_cpu.reshape(1, x_out_cpu.numel())[0].cpu()
            x_npu_data = x_out.reshape(1, x_out.numel())[0].cpu()
            self.assertTrue(self.compare(x_cpu_data, x_npu_data, benchmark))
    
    @unittest.skip("Skip test_npu_add_rms_norm_quant due to low version of cann")
    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_quant_bf16(self):
        shape_list = [[[16, ], [16, ]],
                      [[2, 16], [16, ]],
                      [[2, 16], [2, 16]],
                      [[16, 32], [16, 32]],
                      [[16, 32], [32, ]]]
        for item in shape_list:
            x_shape = item[0]
            quant_shape = item[1]
            x1 = torch.randn(x_shape, dtype=torch.bfloat16)
            x2 = torch.randn(x_shape, dtype=torch.bfloat16)
            gamma = torch.randn(quant_shape, dtype=torch.bfloat16)
            # Don't support scales with 0.
            scales1 = torch.randn(quant_shape, dtype=torch.bfloat16).uniform_(0.1, 1)
            zero_points1 = torch.randn(quant_shape, dtype=torch.bfloat16)

            x1_npu = x1.npu()
            x2_npu = x2.npu()
            gamma_npu = gamma.npu()
            scales1_npu = scales1.npu()
            zero_points1_npu = zero_points1.npu()

            y1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
            y1_cpu, _, x_out_cpu = self.npu_add_rms_norm_quant_golden(x1, x2, gamma, scales1, zero_points1)

            benchmark = math.pow(2, -7)
            benchmark_int8 = 1
            y1_cpu_data = y1_cpu.reshape(1, y1_cpu.numel())[0].cpu()
            y1_npu_data = y1.reshape(1, y1.numel())[0].cpu()
            self.assertTrue(self.compare(y1_cpu_data, y1_npu_data, benchmark_int8))

            x_cpu_data = x_out_cpu.reshape(1, x_out_cpu.numel())[0].cpu()
            x_npu_data = x_out.reshape(1, x_out.numel())[0].cpu()
            self.assertTrue(self.compare(x_cpu_data, x_npu_data, benchmark))


if __name__ == "__main__":
    run_tests()

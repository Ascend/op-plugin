import unittest

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUAddRmsNormQuant(TestCase):

    # pylint:disable = huawei-too-many-arguments
    def npu_add_rms_norm_quant_golden(self, input_x1, input_x2, input_gamma, input_scales1, input_zero_points1, epsilon=1e-06):
        len_shape_x = len(input_x1.shape)
        len_shape_gamma = len(input_gamma.shape)
        axis = len_shape_x - len_shape_gamma
        input_dtype = input_x1.dtype

        if (input_dtype == np.float32) or (input_dtype == np.float16):
            add_x = input_x1 + input_x2
        else:
            add_x = (input_x1.astype(np.float32) + input_x2.astype(np.float32))

        x_fp32 = add_x.astype(np.float32)
        variance = np.mean(np.power(x_fp32, 2), axis=axis, keepdims=True)
        std = np.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd

        if input_dtype == np.float32:
            y_array = result_mid * input_gamma
        elif input_dtype == np.float16:
            result_mid_fp16 = result_mid.astype(np.float16)
            y_array = result_mid_fp16 * input_gamma
        else:
            result_cast = result_mid.astype(input_dtype)
            result_mid = result_cast.astype(np.float32)

            input_gamma_fp32 = input_gamma.astype(np.float32)
            result = result_mid * input_gamma_fp32
            output_dtype = input_gamma.dtype
            y_array = result.astype(output_dtype)

        tensor_scales1 = torch.from_numpy(input_scales1)
        
        if input_zero_points1 is None:
            tensor_zero_points1 = torch.zeros(input_scales1.shape, dtype=torch.int32)
        else:
            tensor_zero_points1 = torch.from_numpy(input_zero_points1)
        
        y = torch.from_numpy(y_array).type(torch.float32)
        y1 = torch.quantize_per_channel(y, tensor_scales1, tensor_zero_points1, axis, torch.qint8)
        y1_np = y1.int_repr().detach().clone().cpu()
        return y1_np, y1_np, torch.from_numpy(add_x.astype(input_dtype))

    @unittest.skip("Skip test_npu_add_rms_norm_quant due to low version of cann")
    @SupportedDevices(['Ascend910B'])
    def test_npu_add_rms_norm_quant(self):
        dtype_set_list = [[torch.float16, torch.float32, torch.int32],
                          [torch.bfloat16, torch.bfloat16, torch.bfloat16]]
        shape_set_list = [[[16, ], [16, ]], [[2, 16], [16, ]], [[1024, 11264], [11264, ]]]
        shape_dtype_list = [[shape, dtype]
                            for shape in shape_set_list
                            for dtype in dtype_set_list]
        for item in shape_dtype_list:
            x_shape = item[0][0]
            quant_shape = item[0][1]
            x_dtype = item[1][0]
            scales_dtype = item[1][1]
            zero_dtype = item[1][2]
            x1 = torch.randn(x_shape, dtype=x_dtype)
            x2 = torch.randn(x_shape, dtype=x_dtype)
            gamma = torch.randn(quant_shape, dtype=x_dtype)
            scales1 = torch.randn(quant_shape, dtype=scales_dtype)
            if zero_dtype == torch.int32:
                zero_points1 = torch.randint(-10, 10, quant_shape, dtype=zero_dtype)
            else:
                zero_points1 = torch.randn(quant_shape, dtype=zero_dtype)

            x1_npu = x1.npu()
            x2_npu = x2.npu()
            gamma_npu = gamma.npu()
            scales1_npu = scales1.npu()
            zero_points1_npu = zero_points1.npu()

            if x_dtype == torch.bfloat16:
                x1 = x1.to(torch.float32)
                x2 = x2.to(torch.float32)
                gamma = gamma.to(torch.float32)
                scales1 = scales1.to(torch.float32)
                zero_points1 = zero_points1.to(torch.float32)
            npu_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
            golden_out = self.npu_add_rms_norm_quant_golden(x1.numpy(), x2.numpy(), gamma.numpy(), scales1.numpy(), zero_points1.numpy())

            if x_dtype == torch.bfloat16:
                self.assertRtolEqual(npu_out[0].cpu(), golden_out[0], prec=2**(-7))
                self.assertRtolEqual(npu_out[2].to(torch.float32).cpu(), golden_out[2], prec=2**(-7))
            else:
                self.assertRtolEqual(npu_out[0].cpu(), golden_out[0])
                self.assertRtolEqual(npu_out[2].cpu(), golden_out[2])


if __name__ == "__main__":
    run_tests()

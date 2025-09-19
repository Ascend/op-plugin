import unittest
import math

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPURmsNormQuant(TestCase):

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

    def npu_rms_norm_quant_golden(self, x, gamma, beta, scale,
                                    offset, epsilon=1e-06):
    
        x_fp32 = x.float()
        input_gamma_fp32 = gamma.float()
        input_beta_fp32 = beta.float()
        tensor_scales = scale.float()
        offset = offset.float()
        ori_shape = x.shape
    
        len_shape_x = len(x_fp32.shape)
        len_shape_gamma = len(gamma.shape)
        axis = len_shape_x - len_shape_gamma
        variance = torch.mean(torch.pow(x_fp32, 2), axis=-1, keepdims=True)
        std = torch.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd
        y_array = result_mid * input_gamma_fp32 + input_beta_fp32
        y = y_array.type(torch.float32)
        y1 = torch.quantize_per_tensor(y, tensor_scales, offset, torch.qint8)
        y1_np = y1.int_repr().detach().clone().cpu().numpy()
        return torch.tensor(y1_np).type(torch.float16).type(torch.int8).reshape(ori_shape)

    @unittest.skip("skip until CANN is updated to support aclnnRmsNormQuant")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rms_norm_quant(self):
        torch.manual_seed(42) 
        np.random.seed(42)  
        shape_list = [
            [[16, ], [16, ]],    
            [[16, ], [1, 16]],   
            [[1, 16], [16, ]],     
            [[1, 16], [1, 16]],   
            [[1, 1, 16], [16, ]], 
            [[1, 1, 16], [1, 16]],
            [[1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 1, 1, 16], [1, 16]],     
            [[1, 1, 1, 1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 1, 1, 1, 16], [1, 16]],  
        ]

        benchmark_int8 = 1              

        for x_shape, quant_shape in shape_list:
            D = x_shape[-1]

            x = torch.randn(x_shape, dtype=torch.float16)

            if quant_shape == [D, ]:
                gamma = torch.randn(D, dtype=torch.float16)
                beta = torch.randn(D, dtype=torch.float16)
            elif quant_shape == [1, D]:
                gamma = torch.randn(1, D, dtype=torch.float16)
                beta = torch.randn(1, D, dtype=torch.float16)

            scale = (torch.rand(1, dtype=torch.float16) * 0.8 + 0.2)  
            offset = torch.randint(-5, 6, (1, ), dtype=torch.int8)
            x_npu = x.npu()     
            gamma_npu = gamma.npu() 
            beta_npu = beta.npu() 
            scale_npu = scale.npu() 
            offset_npu = offset.npu() 
            
            y_ref = self.npu_rms_norm_quant_golden(x, gamma, beta, scale, offset, epsilon=1e-6)
            y_npu = torch_npu.npu_rms_norm_quant(x_npu, gamma_npu, beta_npu, scale_npu, offset_npu, epsilon=1e-6)
            y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
            y_npu_flat = y_npu.reshape(1, y_npu.numel())[0].cpu()
            self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_int8))
    
    @unittest.skip("skip until CANN is updated to support aclnnRmsNormQuant")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rms_norm_quant_bf16(self):
        shape_list = [
            [[16, ], [16, ]],    
            [[16, ], [1, 16]],   
            [[1, 16], [16, ]],     
            [[1, 16], [1, 16]],   
            [[1, 1, 16], [16, ]], 
            [[1, 1, 16], [1, 16]],
            [[1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 1, 1, 16], [1, 16]],    
            [[1, 1, 1, 1, 1, 1, 1, 16], [16, ]], 
            [[1, 1, 1, 1, 1, 1, 1, 16], [1, 16]],  
        ]

        benchmark_int8 = 1              

        for x_shape, quant_shape in shape_list:
            D = x_shape[-1]
            x = torch.randn(x_shape, dtype=torch.bfloat16)
            if quant_shape == [D, ]:
                gamma = torch.randn(D, dtype=torch.bfloat16)
                beta = torch.randn(D, dtype=torch.bfloat16)
            elif quant_shape == [1, D]:
                gamma = torch.randn(1, D, dtype=torch.bfloat16)
                beta = torch.randn(1, D, dtype=torch.bfloat16)

            scale = (torch.rand(1, dtype=torch.bfloat16) * 0.8 + 0.2)  # (0.2, 1.0]
            offset = torch.randint(-5, 6, (1, ), dtype=torch.int8)

            x_npu = x.npu()     
            gamma_npu = gamma.npu() 
            beta_npu = beta.npu() 
            scale_npu = scale.npu() 
            offset_npu = offset.npu() 

            y_ref = self.npu_rms_norm_quant_golden(x, gamma, beta, scale, offset, epsilon=1e-6)
            y_npu = torch_npu.npu_rms_norm_quant(x_npu, gamma_npu, beta_npu, scale_npu, offset_npu, epsilon=1e-6)

            y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
            y_npu_flat = y_npu.reshape(1, y_npu.numel())[0].cpu()
           
            self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_int8))

if __name__ == "__main__":
    run_tests()
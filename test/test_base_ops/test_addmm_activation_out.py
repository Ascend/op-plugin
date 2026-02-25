import copy
import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAddmmActivationOut(TestCase):
    def generate_scalar(self, dtype, min_x, max_x):
        if min_x >= max_x:
            raise ValueError("min_x should be less then max_x")
        if dtype == "float32" or "float16":
            scalar = np.random.uniform(min_x, max_x)
        if dtype == "int32":
            scalar = np.random.randint(min_x, max_x)
        return scalar

    def cpu_reference_out(self, input1, input2, input3, beta, alpha, use_gelu, out):
        # perform addmm into out on CPU, then apply activation in-place
        torch._addmm_activation(input1, input2, input3, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out)
        return out.numpy()

    def npu_op_exec_out(self, input1, input2, input3, beta, alpha, use_gelu, out):
        input1 = input1.to('npu')
        input2 = input2.to('npu')
        input3 = input3.to('npu')
        out_npu = out.to('npu')
        torch._addmm_activation(input1, input2, input3, beta=beta, alpha=alpha, use_gelu=use_gelu, out=out_npu)
        out_res = out_npu.to('cpu').numpy()
        return out_res

    def _run_test(self, dtype, shape_a, shape_b, shape_c, tensor_range, scalar_range, use_gelu):
        """
        :param dtype: np.float32 或 np.float16
        :param shape_a, shape_b, shape_c: 输入张量形状
        :param tensor_range: create_common_tensor 的取值范围 (min, max)
        :param scalar_range: generate_scalar 的取值范围 (min, max)
        :param use_gelu: False 表示 ReLU, True 表示 GeLU
        """
        cpu_input1, npu_input1 = create_common_tensor([dtype, 0, shape_a], 0, tensor_range)
        cpu_input2, npu_input2 = create_common_tensor([dtype, 0, shape_b], 0, tensor_range)
        cpu_input3, npu_input3 = create_common_tensor([dtype, 0, shape_c], 0, tensor_range)
        cpu_out, npu_out = create_common_tensor([dtype, 0, shape_a], 0, tensor_range)

        dtype_str = "float32" if dtype == np.float32 else "float16"
        beta = self.generate_scalar(dtype_str, scalar_range[0], scalar_range[1])
        alpha = self.generate_scalar(dtype_str, scalar_range[0], scalar_range[1])

        # FP16 特殊处理：CPU 计算在 FP32 下进行，然后转回 FP16 与 NPU 行为保持一致
        if dtype == np.float16:
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input3 = cpu_input3.to(torch.float32)
            cpu_out = cpu_out.to(torch.float32)

        cpu_out_copy = copy.deepcopy(cpu_out)
        cpu_ref = self.cpu_reference_out(cpu_input1, cpu_input2, cpu_input3, 
                                        beta, alpha, use_gelu, cpu_out_copy)
        
        if dtype == np.float16:
            cpu_ref = cpu_ref.astype(np.float16)
            
        npu_res = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, 
                                      beta, alpha, use_gelu, npu_out)
        self.assertRtolEqual(cpu_ref, npu_res, prec=1.e-3, prec16=1.e-3)

    def test_addmm_activation_out_fp32(self):
        shape_a = (3, 3)
        shape_b = (3, 4)
        shape_c = (4, 3)
        
        # ReLU 测试
        self._run_test(np.float32, shape_a, shape_b, shape_c, 1, (0, 2), False)
        # GeLU 测试
        self._run_test(np.float32, shape_a, shape_b, shape_c, 1, (0, 2), True)

    def test_addmm_activation_out_fp16(self):
        shape_a = (3, 3)
        shape_b = (3, 4)
        shape_c = (4, 3)
        
        # ReLU 测试
        self._run_test(np.float16, shape_a, shape_b, shape_c, 2, (0, 10), False)
        # GeLU 测试
        self._run_test(np.float16, shape_a, shape_b, shape_c, 2, (0, 10), True)


if __name__ == '__main__':
    run_tests()
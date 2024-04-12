import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMm(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.mm(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.mm(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_mm_split_k_fp16(self):
        torch.npu.set_compile_mode(jit_compile=True)
        # dtype, format, shape, transpose
        shape_info1 = [
            [np.float16, 2, [55296, 640], True]
        ]
        shape_info2 = [
            [np.float16, 2, [55296, 640], False]
        ]
        for idx, item in enumerate(shape_info1):
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(shape_info2[idx], -1, 1)
            if item[0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            trans_a = item[-1]
            trans_b = shape_info2[idx][-1]
            if trans_a:
                cpu_input1 = cpu_input1.t()
                npu_input1 = npu_input1.t()
            if trans_b:
                cpu_input2 = cpu_input2.t()
                npu_input2 = npu_input2.t()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

if __name__ == "__main__":
    run_tests()

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import random
import numpy as np

class TestDropout(TestCase):
    def _test_native_dropout_backward(self, dtype, p):
        grad_output = torch.tensor(1.2,dtype=dtype)
        b = np.random.randint(0,100,size=(0)).astype(np.uint8)
        mask = torch.tensor(b).to(torch.uint8)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, mask, p)
        output_npu = torch.ops.aten.native_dropout_backward(grad_output.npu(), mask.npu(), p) 
        self.assertEqual(output_cpu, output_npu)
    
    
    def test_native_dropout_backward_fp32(self):
        self._test_native_dropout_backward(torch.float32, 2)

    def test_native_dropout_backward_fp16(self):
        self._test_native_dropout_backward(torch.float16, 3)

if __name__ == '__main__':
    run_tests()
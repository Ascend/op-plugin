import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
torch.npu.set_compile_mode(jit_compile=False)


class TestNPUDropoutGenMask(TestCase):
    def cpu_op_exec_p0(self, x):
        BYTE_LEN = 8
        DATA_ALIGN = 128
        size = x.shape
        numels = np.prod(size)
        length = (numels + DATA_ALIGN - 1) // DATA_ALIGN * DATA_ALIGN // BYTE_LEN
        output = np.zeros(length, dtype=np.uint8)
        res_len = (numels + BYTE_LEN - 1) // BYTE_LEN
        for i in range(res_len):
            output[i] = 255
        return output

    def npu_op_exec(self, x, p, seed=1, offset=0, parallel=False):
        x1 = x.to("npu")
        size = x.shape
        output = torch_npu._npu_dropout_gen_mask(x1, size, p, seed, offset, parallel=parallel)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_dropout_gen_full_mask(self):
        h, w = 32, 16
        x = torch.randn(h, w, dtype=torch.float16)
        prob = 0.0
        res = self.npu_op_exec(x, prob)
        res_cpu = self.cpu_op_exec_p0(x)
        self.assertRtolEqual(res, res_cpu)
    
    def test_dropout_gen_mask(self):
        h, w = 17, 19
        x = torch.randn(h, w, dtype=torch.float32)
        prob = 0.4
        res = self.npu_op_exec(x, prob)

    def test_gen_mask_enable_parallel(self):
        h, w = 17, 19
        x = torch.randn(h, w, dtype=torch.float32)
        prob = 0.7
        res = self.npu_op_exec(x, prob, 2, 100, True)
        res1 = self.npu_op_exec(x, prob, 2, 100, False)
        self.assertRtolEqual(res, res1)


if __name__ == "__main__":
    run_tests()

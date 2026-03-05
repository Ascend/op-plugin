import torch
from torch_npu.testing.testcase import TestCase, run_tests
import op_extension


class TestCustomAdd(TestCase):

    def test_add_custom_ops(self):
        length = [8, 2048]
        x = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)
        y = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)

        x_npu = x.npu()
        y_npu = y.npu()
        output = op_extension.ops.custom_add(x_npu, y_npu)
        cpuout = torch.add(x, y)

        self.assertRtolEqual(output, cpuout)


class TestCustomTrig(TestCase):
    def test_trig_custom_ops(self):
        length = [8, 2048]
        x = torch.rand(length, device='npu', dtype=torch.float32)
        out_sin = torch.empty_like(x)
        out_cos = torch.empty_like(x)
        x_npu = x.npu()
        out_sin_npu = out_sin.npu()
        out_cos_npu = out_cos.npu()

        out_tan = op_extension.ops.custom_trig(x_npu, out_sin_npu, out_cos_npu)
        out_sin_cpu = torch.sin(x)
        out_cos_cpu = torch.cos(x)
        out_tan_cpu = torch.tan(x)
        self.assertRtolEqual(out_sin_npu, out_sin_cpu)
        self.assertRtolEqual(out_cos_npu, out_cos_cpu)
        self.assertRtolEqual(out_tan, out_tan_cpu)


if __name__ == "__main__":
    run_tests()

import copy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


class TestCustomAdd(TestCase):
    def test_add_custom(self):
        x_cpu = torch.randn([8, 2048], dtype=torch.float16)
        y_cpu = torch.randn([8, 2048], dtype=torch.float16)
        x_npu, y_npu = copy.deepcopy(x_cpu).npu(), copy.deepcopy(y_cpu).npu()
        x_cpu.requires_grad = True
        y_cpu.requires_grad = True
        x_npu.requires_grad = True
        y_npu.requires_grad = True

        # calculate on npu
        output = custom_ops.add_custom(x_npu, y_npu)
        output.backward(output)

        # calculate on cpu
        cpuout = torch.add(x_cpu, y_cpu)
        cpuout.backward(cpuout)

        # compare result
        self.assertRtolEqual(output, cpuout)
        self.assertRtolEqual(x_npu.grad, x_cpu.grad)
        self.assertRtolEqual(y_npu.grad, y_cpu.grad)


    def test_add_custom_meta(self):
        input1 = torch.randn([8, 2048], dtype=torch.float16)
        input2 = torch.randn([8, 2048], dtype=torch.float16)

        x_input1 = input1.to("meta")
        y_input1 = input2.to("meta")
        x_input1.requires_grad = True
        y_input1.requires_grad = True
        custom_out = custom_ops.add_custom(x_input1, y_input1)
        custom_out.backward(custom_out)

        x_input2 = input1.to("meta")
        y_input2 = input2.to("meta")
        x_input2.requires_grad = True
        y_input2.requires_grad = True
        cpuout = torch.add(x_input2, y_input2)
        cpuout.backward(cpuout)

        self.assertTrue(custom_out.is_meta)
        self.assertRtolEqual(custom_out.size(), cpuout.size())
        self.assertRtolEqual(x_input1.grad.size(), x_input2.grad.size())
        self.assertRtolEqual(y_input1.grad.size(), y_input2.grad.size())


if __name__ == "__main__":
    run_tests()

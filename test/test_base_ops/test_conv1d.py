import torch
import numpy as np
import torch.nn as nn
import torch_npu

from torch.nn.functional import conv1d

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


class TestConv2d(TestCase):
    
    def op_exec_cpu(self, input1, input2, stride):
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            cpu_output = conv1d(input1, input2, stride=stride)
            cpu_output = cpu_output.to(torch.float16)
        else:
            cpu_output = conv1d(input1, input2, stride=stride)
        return cpu_output
    
    def op_exec_npu(self, input1, input2, stride):
        npu_output = conv1d(input1, input2, stride=stride)
        return npu_output.cpu()
    
    def op_exec_backward_cpu(self, input1, input2, stride):
        input1.requires_grad_()
        input2.requires_grad_()
        cpu_output = conv1d(input1, input2, stride=stride)
        cpu_output.sum().backward()
        return input1.grad, input2.grad
    
    def op_exec_backward_npu(self, input1, input2, stride):
        input1.requires_grad_()
        input2.requires_grad_()
        npu_output = conv1d(input1, input2, stride=stride)
        npu_output.sum().backward()
        return input1.grad.cpu(), input2.grad.cpu()

    def test_conv1d_special_case_forward_fp16(self):
        hop_length = [128, 256]
        magnification = [2, 4, 8, 10, 40]
        combine_list = [[length, mag] for length in hop_length for mag in magnification]

        for item in combine_list:
            input1_shape = [4, 1, item[0] * item[1]]
            input2_shape = [24, 1, item[0]]
            attr1_list = [np.float16, 0, input1_shape]
            attr2_list = [np.float16, 0, input2_shape]
            input1_cpu, input1_npu = create_common_tensor(attr1_list, -1, 1)
            input2_cpu, input2_npu = create_common_tensor(attr2_list, -1, 1)

            cpu_output = self.op_exec_cpu(input1_cpu, input2_cpu, item[0])
            npu_output = self.op_exec_npu(input1_npu, input2_npu, item[0])
            self.assertRtolEqual(cpu_output, npu_output)
    
    @SupportedDevices(['Ascend910B'])
    def test_conv1d_special_case_forward_fp32(self):
        hop_length = [128, 256]
        magnification = [2, 4, 8, 10, 40]
        combine_list = [[length, mag] for length in hop_length for mag in magnification]

        for item in combine_list:
            input1_shape = [4, 1, item[0] * item[1]]
            input2_shape = [24, 1, item[0]]
            attr1_list = [np.float32, 0, input1_shape]
            attr2_list = [np.float32, 0, input2_shape]
            input1_cpu, input1_npu = create_common_tensor(attr1_list, -1, 1)
            input2_cpu, input2_npu = create_common_tensor(attr2_list, -1, 1)

            cpu_output = self.op_exec_cpu(input1_cpu, input2_cpu, item[0])
            npu_output = self.op_exec_npu(input1_npu, input2_npu, item[0])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_conv1d_special_case_backward_fp16(self):
        hop_length = [128, 256]
        magnification = [2, 4, 8, 10, 40]
        combine_list = [[length, mag] for length in hop_length for mag in magnification]

        for item in combine_list:
            input1_shape = [4, 1, item[0] * item[1]]
            input2_shape = [24, 1, item[0]]
            attr1_list = [np.float16, 0, input1_shape]
            attr2_list = [np.float16, 0, input2_shape]
            input1_cpu, input1_npu = create_common_tensor(attr1_list, -1, 1)
            input2_cpu, input2_npu = create_common_tensor(attr2_list, -1, 1)

            cpu_grad1, cpu_grad2 = self.op_exec_backward_cpu(input1_cpu.to(torch.float32), 
                                                                input2_cpu.to(torch.float32),
                                                                item[0])
            npu_grad1, npu_grad2 = self.op_exec_backward_npu(input1_npu, input2_npu, item[0])
            self.assertRtolEqual(cpu_grad1.to(torch.float16), npu_grad1)
            self.assertRtolEqual(cpu_grad2.to(torch.float16), npu_grad2)
    
    @SupportedDevices(['Ascend910B'])
    def test_conv1d_special_case_backward_fp32(self):
        hop_length = [128, 256]
        magnification = [2, 4, 8, 10, 40]
        combine_list = [[length, mag] for length in hop_length for mag in magnification]

        for item in combine_list:
            input1_shape = [4, 1, item[0] * item[1]]
            input2_shape = [24, 1, item[0]]
            attr1_list = [np.float32, 0, input1_shape]
            attr2_list = [np.float32, 0, input2_shape]
            input1_cpu, input1_npu = create_common_tensor(attr1_list, -1, 1)
            input2_cpu, input2_npu = create_common_tensor(attr2_list, -1, 1)

            cpu_grad1, cpu_grad2 = self.op_exec_backward_cpu(input1_cpu, input2_cpu, item[0])
            npu_grad1, npu_grad2 = self.op_exec_backward_npu(input1_npu, input2_npu, item[0])
            self.assertRtolEqual(cpu_grad1, npu_grad1)
            self.assertRtolEqual(cpu_grad2, npu_grad2)


if __name__ == "__main__":
    run_tests()

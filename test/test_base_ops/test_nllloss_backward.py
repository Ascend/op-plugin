import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNlllossbackward(TestCase):

    def cpu_op_exec_new(self, input1, target, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100
        input1.requires_grad_(True)
        output = torch.nn.functional.nll_loss(input1, target, reduction=reduction, ignore_index=ignore_index)
        input_cpu = output.detach().numpy()
        output.backward(torch.ones_like(output))
        res = input1.grad
        res = res.numpy()
        return input_cpu, res

    def npu_op_exec_new(self, input1, target, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100
        target = target.to(torch.int32)
        target = target.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.nll_loss(input1, target, reduction=reduction, ignore_index=ignore_index)
        output.backward(torch.ones_like(output))
        input_npu = output.to("cpu")
        input_npu = input_npu.detach().numpy()
        res = input1.grad.to("cpu")
        res = res.numpy()
        return input_npu, res

    def test_nllloss_shape_format_fp32(self):
        # Currently, only positive numbers are supported.
        # If np.sum(ignore_index == np_target) == 0, ignore_index can be set to any value.
        ignore_index = 1
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float32, 0, [256, 100]], reduction, None],
                [[np.float32, 3, [256, 100]], reduction, ignore_index],
                [[np.float32, 0, [4800, 3003]], reduction, ignore_index],
                [[np.float32, 3, [4800, 3003]], reduction, ignore_index],
                [[np.float32, 0, [4800, 3003]], reduction, None],
            ]
            for item in shape_format:
                np_target = np.random.randint(0, item[0][2][1], (item[0][2][0])).astype(np.int64)
                target = torch.from_numpy(np_target)
                cpu_input, _ = create_common_tensor(item[0], -100, 100)
                npu_input = cpu_input.detach().clone().npu()
                cpu_output, cpu_grad = self.cpu_op_exec_new(cpu_input, target, item[1], item[2])
                npu_output, npu_grad = self.npu_op_exec_new(npu_input, target, item[1], item[2])
                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad, npu_grad)

    def test_nllloss_shape_format_fp16(self):
        # Currently, only positive numbers are supported.
        # If np.sum(ignore_index == np_target) == 0, ignore_index can be set to any value.
        ignore_index = 1
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float16, 0, [256, 100]], reduction, ignore_index],
                [[np.float16, 3, [256, 100]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], reduction, ignore_index],
                [[np.float16, 3, [4800, 3003]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], reduction, None]
            ]
            for item in shape_format:
                np_target = np.random.uniform(0, item[0][2][1], (item[0][2][0])).astype(np.int64)
                target = torch.from_numpy(np_target)
                cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
                cpu_input = cpu_input.to(torch.float32)
                cpu_output, cpu_grad = self.cpu_op_exec_new(cpu_input, target, item[1], item[2])
                npu_output, npu_grad = self.npu_op_exec_new(npu_input, target, item[1], item[2])
                cpu_output = cpu_output.astype(np.float16)
                cpu_grad = cpu_grad.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad, npu_grad)

    def test_nllloss_target_0d(self):
        # Currently, only positive numbers are supported.
        # If np.sum(ignore_index == np_target) == 0, ignore_index can be set to any value.
        ignore_index = 1
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float32, 0, [256]], reduction, ignore_index],
                [[np.float32, 0, [4800]], reduction, ignore_index],
                [[np.float32, 0, [4800]], reduction, None],
                [[np.float16, 0, [256]], reduction, ignore_index],
                [[np.float16, 0, [4800]], reduction, ignore_index],
                [[np.float16, 0, [4800]], reduction, None]
            ]
            for item in shape_format:
                np_target = np.random.uniform(1, 100)
                target = torch.tensor(np_target).long()
                cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
                if item[0][0] == np.float16:
                    cpu_input = cpu_input.to(torch.float32)
                cpu_output, cpu_grad = self.cpu_op_exec_new(cpu_input, target, item[1], item[2])
                npu_output, npu_grad = self.npu_op_exec_new(npu_input, target, item[1], item[2])
                if item[0][0] == np.float16:
                    cpu_output = cpu_output.astype(np.float16)
                    cpu_grad = cpu_grad.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)
                self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()

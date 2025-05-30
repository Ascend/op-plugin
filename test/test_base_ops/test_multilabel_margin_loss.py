import numpy as np

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMultilabelMarginLoss(TestCase):
    def cpu_op_exec(self, data, target, reduction):
        output = torch.nn.functional.multilabel_margin_loss(input=data, target=target, reduction=reduction)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def npu_op_exec(self, data, target, reduction):
        output = torch.nn.functional.multilabel_margin_loss(input=data, target=target, reduction=reduction)
        output = output.to("cpu")
        output = output.to(torch.float32)
        output = output.detach().numpy()
        return output

    def cpu_op_exec_out(self, data, target, c_out, reduction):
        output = torch._C._nn.multilabel_margin_loss(input=data, target=target, reduction=reduction, out=c_out)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def npu_op_exec_out(self, data, target, c_out, reduction):
        output = torch._C._nn.multilabel_margin_loss(input=data, target=target, reduction=reduction, out=c_out)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def test_multilabel_margin_loss_1(self):
        data = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        target = torch.Tensor([[3, 0, -1, 1], [0, 1, 3, -1]]).to(torch.int64)

        for reduction in ["mean", "none", "sum"]:
            data_npu = data.to("npu")
            target_npu = target.to(torch.int32).to("npu")
            cpu_output = self.cpu_op_exec(data, target, reduction)
            npu_output = self.npu_op_exec(data_npu, target_npu, reduction)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_2(self):
        data = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        target = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1]]).to(torch.int64)

        for reduction in ["mean", "none", "sum"]:
            data_npu = data.to("npu")
            target_npu = target.to(torch.int32).to("npu")
            cpu_output = self.cpu_op_exec(data, target, reduction)
            npu_output = self.npu_op_exec(data_npu, target_npu, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_3(self):
        data = torch.Tensor([[0.1, 0.2, 0.4, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.2, 0.4, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]]).to(torch.float32)
        target = torch.Tensor([[3, 0, 7, 8, 1, -1, 1, 2, 2], [4, 5, -1, 1, 1, 1, 1, 2, 2]]).to(torch.int64)

        for reduction in ["mean", "none", "sum"]:
            data_npu = data.to("npu")
            target_npu = target.to(torch.int32).to("npu")
            cpu_output = self.cpu_op_exec(data, target, reduction)
            npu_output = self.npu_op_exec(data_npu, target_npu, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_out(self):
        data = torch.tensor([[-0.4191, 0.6214],
                             [-0.3765, -0.4781],
                             [0.2881, 0.4888]]).to(torch.float32)
        target = torch.tensor([[1, -1],
                               [0, -1],
                               [1, -1]]).to(torch.int64)

        for reduction in range(3):
            data_npu = data.to("npu")
            target_npu = target.to(torch.int32).to("npu")
            c_out = torch.randn(1, 2, 3).float()
            cpu_output = self.cpu_op_exec_out(data, target, c_out, reduction)
            c_out = torch.randn(1, 2, 3).float()
            c_npu = c_out.to("npu")
            npu_output = self.npu_op_exec_out(data_npu, target_npu, c_npu, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_out_2(self):
        a = np.random.uniform(-0.001, 0.0, (97, 128)).astype(np.float32)
        b = np.random.randint(-1, 64, (97, 128)).astype(np.int32)
        out = np.random.uniform(-0.001, 0.0, (97, 128)).astype(np.float32)
        a1 = torch.from_numpy(a)
        b1 = torch.from_numpy(b).to(torch.int64)
        out1 = torch.from_numpy(out)

        a2 = torch.from_numpy(a).npu()
        b2 = torch.from_numpy(b).npu()
        out2 = torch.from_numpy(out).npu()

        reduction = 2
        output_data = self.cpu_op_exec_out(a1, b1, out1, reduction)
        output_data_npu = self.npu_op_exec_out(a2, b2, out2, reduction)
        self.assertRtolEqual(output_data, output_data_npu)

    def test_multilabel_margin_loss_float16_1(self):
        data = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        target = torch.Tensor([[3, 0, -1, 1], [0, 1, 3, -1]]).to(torch.int64)

        for reduction in ["mean", "none", "sum"]:
            data_npu = data.to(torch.float16).to("npu")
            target_npu = target.to(torch.int32).to("npu")
            cpu_output = self.cpu_op_exec(data, target, reduction)
            npu_output = self.npu_op_exec(data_npu, target_npu, reduction)

            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_multilabel_margin_loss_float16_2(self):
        data = torch.Tensor([[0.1, 0.2, 0.4, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.2, 0.4, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]]).to(torch.float32)
        target = torch.Tensor([[3, 0, 7, 8, 1, -1, 1, 2, 2], [4, 5, -1, 1, 1, 1, 1, 2, 2]]).to(torch.int64)

        for reduction in ["mean", "none", "sum"]:
            data_npu = data.to(torch.float16).to("npu")
            target_npu = target.to(torch.int32).to("npu")
            cpu_output = self.cpu_op_exec(data, target, reduction)
            npu_output = self.npu_op_exec(data_npu, target_npu, reduction)

            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

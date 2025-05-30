import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestBatchNormBackwardElemt(TestCase):
    def test_batch_norm_backward_elemt_4d(self):
        grad_output = torch.ones([2, 3, 1, 4]).npu()
        input1 = torch.ones([2, 3, 1, 4]).npu()
        mean = torch.tensor([8.0, 5.0, 9.0]).npu()
        invstd = torch.tensor([2.0, 1.0, 2.0]).npu()
        weight = torch.tensor([1.0, 1.0, 4.0]).npu()
        mean_dy = torch.tensor([2.0, 2.0, 6.0]).npu()
        mean_dy_xmn = torch.tensor([2.0, 3.0, 11.0]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(
            grad_output,
            input1,
            mean,
            invstd,
            weight,
            mean_dy,
            mean_dy_xmn,
            count_tensor,
        )
        cuda_expect_out = torch.tensor(
            [
                [
                    [[9.2000, 9.2000, 9.2000, 9.2000]],
                    [[1.6667, 1.6667, 1.6667, 1.6667]],
                    [[192.5333, 192.5333, 192.5333, 192.5333]],
                ],
                [
                    [[9.2000, 9.2000, 9.2000, 9.2000]],
                    [[1.6667, 1.6667, 1.6667, 1.6667]],
                    [[192.5333, 192.5333, 192.5333, 192.5333]],
                ],
            ]
        )
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)

    def test_batch_norm_backward_elemt_2d(self):
        grad_output = torch.ones([2, 3]).npu()
        input1 = torch.ones([2, 3]).npu()
        mean = torch.tensor([8.0, 5.0, 9.0]).npu()
        invstd = torch.tensor([2.0, 1.0, 2.0]).npu()
        weight = torch.tensor([1.0, 1.0, 4.0]).npu()
        mean_dy = torch.tensor([2.0, 2.0, 6.0]).npu()
        mean_dy_xmn = torch.tensor([2.0, 3.0, 11.0]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(
            grad_output,
            input1,
            mean,
            invstd,
            weight,
            mean_dy,
            mean_dy_xmn,
            count_tensor,
        )
        cuda_expect_out = torch.tensor(
            [[9.2000, 1.6667, 192.5333], [9.2000, 1.6667, 192.5333]]
        )
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)

    def test_batch_norm_backward_elemt_2d_fp(self):
        grad_output = torch.ones([2, 3]).npu()
        input1 = torch.ones([2, 3]).npu()
        mean = torch.tensor([8.123456, 5.147125, 9.365778]).npu()
        invstd = torch.tensor([2.65485, 1.36541, 2.25879]).npu()
        weight = torch.tensor([1.36987, 1.36944, 4.25774]).npu()
        mean_dy = torch.tensor([2.0, 2.0, 6.0]).npu()
        mean_dy_xmn = torch.tensor([2.0, 3.0, 11.0]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(
            grad_output,
            input1,
            mean,
            invstd,
            weight,
            mean_dy,
            mean_dy_xmn,
            count_tensor,
        )
        cuda_expect_out = torch.tensor(
            [[27.4980, 4.5119, 306.8037], [27.4980, 4.5119, 306.8037]]
        )
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)


if __name__ == "__main__":
    run_tests()

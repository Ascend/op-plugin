import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.decorator import Dtypes, instantiate_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


@instantiate_tests
class TestSlice(TestCase):
    def npu_op_exec(self, input1, offset, sizes):
        output = torch_npu.npu_slice(input1, offset, sizes)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @Dtypes(torch.float, torch.half, torch.int32, torch.uint8, torch.int8, torch.int16, torch.long)
    def test_slice(self, device, dtype):
        input_data = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).npu().to(dtype)
        exoutput = torch.tensor([[1, 2], [6, 7]]).to(dtype)
        output = self.npu_op_exec(input_data, [0, 0], [2, 2])
        self.assertRtolEqual(exoutput.numpy(), output)

    @skipIfUnsupportMultiNPU(2)
    def test_npu_slice_multinpu(self):
        a = torch.randn((8, 500, 500), dtype=torch.float16).npu()
        b = a[:, :, :2]
        b = b.to(device="npu:1", non_blocking=False)
        self.assertRtolEqual(b, a[:, :, :2])


if __name__ == "__main__":
    run_tests()

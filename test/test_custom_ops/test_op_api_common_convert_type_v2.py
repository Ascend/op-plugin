import os

# ConvertTypeV2 is selected only by task queue v2. Set this before importing
# torch_npu so OptionsManager sees it before any op_api macro caches the value.
os.environ["TASK_QUEUE_ENABLE"] = "2"

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestOpApiCommonConvertTypeV2(TestCase):
    @SupportedDevices(["Ascend950"])
    def test_compact_mx_a8w4_grouped_matmul_weight_nz_task_queue_v2(self):
        torch.manual_seed(0)

        m = 16
        k = 256
        n = 512
        expert_num = 2
        group_size = 32
        group_list = torch.tensor([8, 8], dtype=torch.int64).npu()

        x = torch.ones((m, k), dtype=torch.float8_e4m3fn).npu()
        # Compact W4 layout stores two 4-bit values in each int8 element, packed
        # along the K axis. The logical weight is (E, N, K), so the compact
        # storage is (E, N, K // 2). Cast the int8 compact tensor directly to NZ
        # with explicit logical float4 input dtype so ConvertTypeV2 handles a
        # non-base-format TensorWrapper whose logical dtype is 4-bit.
        weight_compact = torch.zeros((expert_num, n, k // 2), dtype=torch.int8).npu()
        weight_nz = torch_npu.npu_format_cast(
            weight_compact,
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        # npu_grouped_matmul consumes the weight transposed on its last two axes
        # (is_weight_trans must be true so the N axis is not doubled by FP4_IN_INT8).
        weight_nz = weight_nz.transpose(-1, -2)
        # mxA8W4 requires a 4-D pergroup antiquant scale shaped
        # (E, N, K // (2 * group_size), 2): K is halved again for the compact W4
        # packing and the trailing 2 carries the e8m0 scale pair. Transpose the
        # middle two axes (N and the group axis) to match the transposed weight.
        antiquant_scale = torch.full(
            (expert_num, n, k // (2 * group_size), 2), 127, dtype=torch.uint8
        ).npu()
        antiquant_scale = antiquant_scale.transpose(1, 2)
        # The per-token scale follows the same compact-W4 group layout as the
        # antiquant scale: a 3-D scale shaped (M, K // (2 * group_size), 2) where
        # K is halved for the compact W4 packing and the trailing 2 carries the
        # e8m0 scale pair.
        pertoken_scale = torch.full((m, k // (2 * group_size), 2), 127, dtype=torch.uint8).npu()

        output = torch_npu.npu_grouped_matmul(
            [x],
            [weight_nz],
            bias=None,
            scale=None,
            offset=None,
            antiquant_scale=[antiquant_scale],
            antiquant_offset=None,
            per_token_scale=[pertoken_scale],
            group_list=group_list,
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=3,
            group_type=0,
            group_list_type=1,
            act_type=0,
            output_dtype=torch.float16,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )

        torch_npu.npu.synchronize()
        expected = torch.zeros((m, n), dtype=torch.float16)
        self.assertEqual(output[0].shape, expected.shape)
        self.assertRtolEqual(expected.numpy(), output[0].cpu().numpy(), 0.001)


if __name__ == "__main__":
    run_tests()

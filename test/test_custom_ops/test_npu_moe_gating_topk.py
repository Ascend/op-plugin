import itertools
import numpy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuMoeGatingTopK(TestCase):
    def moe_gating_top_k_numpy(self, x: torch.tensor, k: int, *, bias: torch.tensor = None, k_group: int = 1, group_count: int = 1,
                            group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, out_flag: bool = False,
                            routed_scaling_factor: float = 1.0, eps: float = 1e-20) -> tuple:
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            bias = bias.to(dtype=torch.float32)

        x = x.numpy()
        bias = bias.numpy()
        if norm_type == 0:
            x = numpy.exp(x - numpy.expand_dims(numpy.log(numpy.sum(numpy.exp(x),
                        axis=-1, keepdims=True)), axis=-1))  # softmax
        else:
            x = 1 / (1 + numpy.exp(-x))  # sigmoid
        original_x = x
        if bias is not None:
            x = x + bias
        
        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = numpy.amax(x, axis=-1)
            else:
                group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
        indices = numpy.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]  # Indices of top-k_group

        mask = numpy.ones((x.shape[0], group_count), dtype=bool)  # Create a mask with all 1
        mask[numpy.arange(x.shape[0])[:, None], indices] = False  # Set to false at the indices
        x = numpy.where(mask[..., None], float('-inf'), x)  # Fill with -inf when mask value is true
        x = x.reshape(x.shape[0], -1)

        indices = numpy.argsort(-x, axis=-1, kind='stable')
        indices = indices[:, :k]
        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type == 1:
            y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor
        if out_flag:
            out = original_x
        else:
            out = None

        y = torch.tensor(y, dtype=dtype)
        return y, indices.astype(numpy.int32), out

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_gating_topk_1(self, device="npu"):
        x = numpy.random.uniform(-2, 2, (8, 256)).astype(numpy.float32)
        bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        bias_tensor = torch.tensor(bias, dtype=torch.float32)

        k = 6
        k_group = 4
        group_count = 8
        group_select_mode = 1
        renorm = 0
        norm_type = 1
        out_flag = False
        routed_scaling_factor = 1.0
        eps = 1e-20

        y, expert_idx, out = self.moe_gating_top_k_numpy(
            x_tensor,
            k,
            bias=bias_tensor,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps)

        y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(
            x_tensor.npu(),
            k,
            bias=bias_tensor.npu(),
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps)

        self.assertRtolEqual(y, y_npu.cpu())
        self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_gating_topk_2(self, device="npu"):
        x = numpy.random.uniform(-2, 2, (1002, 256)).astype(numpy.float32)
        bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)

        x_tensor = torch.tensor(x, dtype=torch.float16)
        bias_tensor = torch.tensor(bias, dtype=torch.float16)

        k = 1
        k_group = 4
        group_count = 8
        group_select_mode = 1
        renorm = 0
        norm_type = 1
        out_flag = False
        routed_scaling_factor = 1.0
        eps = 1e-20

        y, expert_idx, out = self.moe_gating_top_k_numpy(
            x_tensor,
            k,
            bias=bias_tensor,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps)

        y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(
            x_tensor.npu(),
            k,
            bias=bias_tensor.npu(),
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps)

        self.assertRtolEqual(y, y_npu.cpu())
        self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_gating_topk_3(self, device="npu"):
        x = numpy.random.uniform(-2, 2, (128, 256)).astype(numpy.float32)
        bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)

        x_tensor = torch.tensor(x, dtype=torch.bfloat16)
        bias_tensor = torch.tensor(bias, dtype=torch.bfloat16)

        k = 16
        k_group = 4
        group_count = 8
        group_select_mode = 1
        renorm = 0
        norm_type = 1
        out_flag = False
        routed_scaling_factor = 1.0
        eps = 1e-20

        y, expert_idx, out = self.moe_gating_top_k_numpy(
            x_tensor,
            k,
            bias=bias_tensor,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps)

        y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(
            x_tensor.npu(),
            k,
            bias=bias_tensor.npu(),
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps)

        self.assertRtolEqual(y, y_npu.cpu())
        self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())

if __name__ == "__main__":
    run_tests()

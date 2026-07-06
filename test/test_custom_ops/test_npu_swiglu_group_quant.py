import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestSwigluGroupQuant(TestCase):

    def golden_swiglu_group_quant(self, x, weight=None, group_index=None, scale=None,
                                  dst_type=296, quant_mode=0, block_size=0, round_scale=False,
                                  clamp_limit=-1.0, dst_type_max=0.0, output_origin=False):
        input_dtype = x.dtype
        original_shape = x.shape
        H = x.shape[-1] // 2

        x = x.float()
        x = x.reshape(-1, 2 * H)
        x0 = x[:, :H]
        x1 = x[:, H:]

        if clamp_limit != 0:
            x0 = torch.clamp(x0, max=clamp_limit)
            x1 = torch.clamp(x1, -clamp_limit, clamp_limit)

        y = torch.nn.functional.silu(x0) * x1

        output_shape = original_shape[:-1] + (H,)
        if weight is not None:
            weight = weight.reshape(-1, 1)
            y = y * weight

        y_origin = y.to(input_dtype).reshape(output_shape) if output_origin else torch.empty(0, device=x.device)

        eps = 1e-8
        if group_index is None:
            amax = torch.max(torch.abs(y))
            amax = torch.clamp(amax, min=eps)
            print(f"max {amax}")
            y_scale = amax / dst_type_max
            y_quant = torch_npu.npu_dtype_cast(y.npu() / y_scale.npu(), torch_npu.hifloat8)
            y_quant = y_quant.reshape(output_shape)
            return y_quant, y_scale.unsqueeze(0), y_origin
        else:
            num_groups = group_index.shape[0]
            total_tokens = y.shape[0]
            y_scale = torch.zeros(num_groups, dtype=torch.float32, device=x.device)
            y_quant = torch.zeros(total_tokens, H, dtype=torch.uint8, device=x.device)
            start = 0
            for g in range(num_groups):
                end = start + group_index[g].item()
                y_g = y[start:end, :]
                amax_g = torch.max(torch.abs(y_g))
                amax_g = torch.clamp(amax_g, min=eps)
                scale_g = amax_g / dst_type_max
                y_scale[g] = scale_g
                y_g_quant = torch_npu.npu_dtype_cast(y_g.npu() / scale_g.npu(), torch_npu.hifloat8)
                y_quant[start:end, :] = y_g_quant
                start = end
            y_quant = y_quant.reshape(output_shape)
            return y_quant, y_scale, y_origin


    @SupportedDevices(['Ascend950'])
    def test_swiglu_group_quant(self):
        x = torch.randn([8, 16], dtype=torch.float32)
        weight = torch.randn([8, 1], dtype=torch.float32)
        group_index = torch.randint(1, 10, (2,), dtype=torch.int64)
        scale = torch.randn([2], dtype=torch.float32)

        quant_mode = 3
        round_scale = False
        clamp_limit = 0.01
        dst_type_max = 15.0
        output_origin = True

        y_golden, y_scale_golden, y_origin_golden = self.golden_swiglu_group_quant(x,
                                                                                   weight=weight,
                                                                                   group_index=group_index,
                                                                                   scale=scale,
                                                                                   quant_mode=quant_mode,
                                                                                   round_scale=round_scale,
                                                                                   clamp_limit=clamp_limit,
                                                                                   dst_type_max=dst_type_max,
                                                                                   output_origin=output_origin)

        x_npu = x.npu()
        weight_npu = weight.npu()
        group_index_npu = group_index.npu()
        scale_npu = scale.npu()

        y_npu, y_scale_npu, y_origin_npu = torch_npu.npu_swiglu_group_quant(x_npu,
                                                                            weight=weight_npu,
                                                                            group_index=group_index_npu,
                                                                            scale=scale_npu,
                                                                            quant_mode=quant_mode,
                                                                            round_scale=round_scale,
                                                                            clamp_limit=clamp_limit,
                                                                            dst_type_max=dst_type_max,
                                                                            output_origin=output_origin)
        y_cpu = y_npu.cpu()
        y_scale_cpu = y_scale_npu.cpu()
        y_origin_cpu = y_origin_npu.cpu()

        self.assertRtolEqual(y_golden, y_cpu)
        self.assertRtolEqual(y_scale_golden.type(torch.float32), y_scale_cpu.type(torch.float32))
        self.assertRtolEqual(y_origin_golden.type(torch.float32), y_origin_cpu.type(torch.float32))


if __name__ == "__main__":
    run_tests()

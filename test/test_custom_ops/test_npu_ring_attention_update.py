import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def _repeat_last_dim_8(x):
    return x.unsqueeze(-1).repeat_interleave(8, dim=-1).contiguous()


def _ring_attention_update_golden(
    prev_attn_out,
    prev_softmax_max,
    prev_softmax_sum,
    cur_attn_out,
    cur_softmax_max,
    cur_softmax_sum,
    input_layout,
):
    prev_attn = prev_attn_out.to(torch.float32).cpu()
    cur_attn = cur_attn_out.to(torch.float32).cpu()
    prev_max = prev_softmax_max[..., 0].to(torch.float32).cpu()
    prev_sum = prev_softmax_sum[..., 0].to(torch.float32).cpu()
    cur_max = cur_softmax_max[..., 0].to(torch.float32).cpu()
    cur_sum = cur_softmax_sum[..., 0].to(torch.float32).cpu()

    softmax_max = torch.maximum(prev_max, cur_max)
    prev_scale = torch.exp(prev_max - softmax_max)
    cur_scale = torch.exp(cur_max - softmax_max)
    softmax_sum = prev_sum * prev_scale + cur_sum * cur_scale

    if input_layout == "SBH":
        seq_len, batch_size, hidden_size = prev_attn.shape
        head_num = prev_max.shape[1]
        head_dim = hidden_size // head_num
        prev_attn = prev_attn.reshape(seq_len, batch_size, head_num, head_dim)
        cur_attn = cur_attn.reshape(seq_len, batch_size, head_num, head_dim)
        prev_factor = (prev_scale * prev_sum / softmax_sum).permute(2, 0, 1).unsqueeze(-1)
        cur_factor = (cur_scale * cur_sum / softmax_sum).permute(2, 0, 1).unsqueeze(-1)
        attn_out = prev_attn * prev_factor + cur_attn * cur_factor
        attn_out = attn_out.reshape(seq_len, batch_size, hidden_size)
    else:
        prev_factor = (prev_scale * prev_sum / softmax_sum).unsqueeze(-1)
        cur_factor = (cur_scale * cur_sum / softmax_sum).unsqueeze(-1)
        attn_out = prev_attn * prev_factor + cur_attn * cur_factor

    return (
        attn_out.to(prev_attn_out.dtype),
        _repeat_last_dim_8(softmax_max),
        _repeat_last_dim_8(softmax_sum),
    )


class TestNpuRingAttentionUpdate(TestCase):
    def _run_sbh_case(self, dtype):
        seq_len, batch_size, head_num, head_dim = 4, 2, 2, 16
        hidden_size = head_num * head_dim

        prev_attn_out = torch.randn(seq_len, batch_size, hidden_size, dtype=dtype, device="npu")
        cur_attn_out = torch.randn(seq_len, batch_size, hidden_size, dtype=dtype, device="npu")
        prev_softmax_max = _repeat_last_dim_8(torch.rand(batch_size, head_num, seq_len, dtype=torch.float32, device="npu") + 0.2)
        prev_softmax_sum = _repeat_last_dim_8(torch.rand(batch_size, head_num, seq_len, dtype=torch.float32, device="npu") + 0.5)
        cur_softmax_max = _repeat_last_dim_8(torch.rand(batch_size, head_num, seq_len, dtype=torch.float32, device="npu") + 0.3)
        cur_softmax_sum = _repeat_last_dim_8(torch.rand(batch_size, head_num, seq_len, dtype=torch.float32, device="npu") + 0.4)

        attn_out, softmax_max, softmax_sum = torch_npu.npu_ring_attention_update(
            prev_attn_out,
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
        )
        golden = _ring_attention_update_golden(
            prev_attn_out,
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            input_layout="SBH",
        )
        self.assertRtolEqual(attn_out.cpu(), golden[0], prec16=0.005)
        self.assertRtolEqual(softmax_max.cpu(), golden[1], prec16=0.005)
        self.assertRtolEqual(softmax_sum.cpu(), golden[2], prec16=0.005)

    def _run_tnd_case(self, dtype):
        total_tokens, head_num, head_dim = 5, 2, 64
        actual_seq_qlen = torch.tensor([2, total_tokens], dtype=torch.int64, device="npu")

        prev_attn_out = torch.randn(total_tokens, head_num, head_dim, dtype=dtype, device="npu")
        cur_attn_out = torch.randn(total_tokens, head_num, head_dim, dtype=dtype, device="npu")
        prev_softmax_max = _repeat_last_dim_8(torch.rand(total_tokens, head_num, dtype=torch.float32, device="npu") + 0.2)
        prev_softmax_sum = _repeat_last_dim_8(torch.rand(total_tokens, head_num, dtype=torch.float32, device="npu") + 0.5)
        cur_softmax_max = _repeat_last_dim_8(torch.rand(total_tokens, head_num, dtype=torch.float32, device="npu") + 0.3)
        cur_softmax_sum = _repeat_last_dim_8(torch.rand(total_tokens, head_num, dtype=torch.float32, device="npu") + 0.4)

        attn_out, softmax_max, softmax_sum = torch_npu.npu_ring_attention_update(
            prev_attn_out,
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            actual_seq_qlen=actual_seq_qlen,
            input_layout="TND",
            input_softmax_layout="TND",
        )
        golden = _ring_attention_update_golden(
            prev_attn_out,
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            input_layout="TND",
        )
        self.assertRtolEqual(attn_out.cpu(), golden[0], prec16=0.005)
        self.assertRtolEqual(softmax_max.cpu(), golden[1], prec16=0.005)
        self.assertRtolEqual(softmax_sum.cpu(), golden[2], prec16=0.005)

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_npu_ring_attention_update_sbh(self):
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            self._run_sbh_case(dtype)

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_npu_ring_attention_update_tnd(self):
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            self._run_tnd_case(dtype)


if __name__ == "__main__":
    run_tests()

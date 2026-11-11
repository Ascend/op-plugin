import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNativeMultiHeadAttention(TestCase):
    # ATen 算子 _native_multi_head_attention：自注意力路径。
    # 权重约定（对齐 native_multi_head_attention_cpu）：
    #   query/key/value: [B, S, D]
    #   qkv_weight: [3D, D], qkv_bias: [3D]
    #   proj_weight: [D, D], proj_bias: [D]
    # mask 为 bool，True == 屏蔽。mask_type: 0 == src_mask [S, S]；1 == padding mask [B, S]；2 == generic mask [B, N, S, S]。

    PREC = 1.e-3       # fp32 判定阈值
    PREC16 = 1.e-2     # fp16 判定阈值

    def _mha(self, query, embed_dim, num_head, qkv_weight, qkv_bias,
             proj_weight, proj_bias, mask, mask_type):
        return torch.ops.aten._native_multi_head_attention(
            query, query, query, embed_dim, num_head,
            qkv_weight, qkv_bias, proj_weight, proj_bias,
            mask, False, True, mask_type)[0]

    def _gen_inputs(self, batch, seq_len, embed_dim, num_head, dtype):
        inputs = {}
        for name, shape, low, high in [
            ("query", [batch, seq_len, embed_dim], -1.0, 1.0),
            ("qkv_weight", [3 * embed_dim, embed_dim], -0.1, 0.1),
            ("qkv_bias", [3 * embed_dim], -0.1, 0.1),
            ("proj_weight", [embed_dim, embed_dim], -0.1, 0.1),
            ("proj_bias", [embed_dim], -0.1, 0.1),
        ]:
            cpu, npu = create_common_tensor([dtype, -1, shape], low, high)
            inputs[name] = (cpu, npu)
        return inputs

    def _padding_mask(self, batch, seq_len, pad_len):
        mask = np.zeros((batch, seq_len), dtype=np.bool_)
        if pad_len > 0:
            mask[:, -pad_len:] = True
        return torch.from_numpy(mask)

    def _generic_mask(self, batch, num_head, seq_len, pad_len):
        mask = np.zeros((batch, num_head, seq_len, seq_len), dtype=np.bool_)
        if pad_len > 0:
            mask[:, :, :, -pad_len:] = True
        return torch.from_numpy(mask)

    def _src_mask(self, seq_len, pad_len):
        # src_mask（attn_mask）：[S, S]，True == 屏蔽。
        mask = np.zeros((seq_len, seq_len), dtype=np.bool_)
        if pad_len > 0:
            mask[:, -pad_len:] = True
        return torch.from_numpy(mask)

    def _run(self, inputs, embed_dim, num_head, dtype, mask_cpu=None, mask_npu=None, mask_type=None):
        """双端执行并断言：CPU 参考实现 vs NPU 融合实现。

        mask 按设备分开传入（mask_cpu / mask_npu），避免跨设备 tensor 直接调用。
        """
        cpu_out = self._mha(inputs["query"][0], embed_dim, num_head,
                            inputs["qkv_weight"][0], inputs["qkv_bias"][0],
                            inputs["proj_weight"][0], inputs["proj_bias"][0],
                            mask_cpu, mask_type)
        npu_out = self._mha(inputs["query"][1], embed_dim, num_head,
                            inputs["qkv_weight"][1], inputs["qkv_bias"][1],
                            inputs["proj_weight"][1], inputs["proj_bias"][1],
                            mask_npu, mask_type)
        self.assertRtolEqual(cpu_out, npu_out.cpu(), prec=self.PREC, prec16=self.PREC16)

    def _pair(self, mask):
        """同一 numpy mask 生成 (cpu, npu) 两份 tensor。"""
        return mask, mask.to("npu")

    def test_native_multi_head_attention_no_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)
        self._run(inputs, embed_dim, num_head, np.float32)

    def test_native_multi_head_attention_padding_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)
        mask_cpu, mask_npu = self._pair(self._padding_mask(batch, seq_len, pad_len=2))
        self._run(inputs, embed_dim, num_head, np.float32, mask_cpu, mask_npu, 1)

    def test_native_multi_head_attention_generic_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)
        mask_cpu, mask_npu = self._pair(self._generic_mask(batch, num_head, seq_len, pad_len=2))
        self._run(inputs, embed_dim, num_head, np.float32, mask_cpu, mask_npu, 2)

    def test_native_multi_head_attention_src_mask_float32(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)
        mask_cpu, mask_npu = self._pair(self._src_mask(seq_len, pad_len=2))
        self._run(inputs, embed_dim, num_head, np.float32, mask_cpu, mask_npu, 0)

    def test_native_multi_head_attention_no_mask_float16(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float16)
        self._run(inputs, embed_dim, num_head, np.float16)

    def test_native_multi_head_attention_padding_mask_float16(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float16)
        mask_cpu, mask_npu = self._pair(self._padding_mask(batch, seq_len, pad_len=2))
        self._run(inputs, embed_dim, num_head, np.float16, mask_cpu, mask_npu, 1)

    def test_native_multi_head_attention_src_mask_float16(self):
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float16)
        mask_cpu, mask_npu = self._pair(self._src_mask(seq_len, pad_len=2))
        self._run(inputs, embed_dim, num_head, np.float16, mask_cpu, mask_npu, 0)

    def test_native_multi_head_attention_float_additive_mask_fallback_float32(self):
        # 非布尔（加性浮点）mask 不得进入融合路径（融合仅接受 bool mask），
        # 应自动回退到组合实现：0 权重表示保留、-inf 表示屏蔽。
        batch, seq_len, embed_dim, num_head = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_head, np.float32)
        mask = np.zeros((seq_len, seq_len), dtype=np.float32)
        mask[:, -2:] = -np.inf
        mask_t = torch.from_numpy(mask)
        self._run(inputs, embed_dim, num_head, np.float32,
                  mask_cpu=mask_t, mask_npu=mask_t.to("npu"), mask_type=0)


if __name__ == "__main__":
    run_tests()

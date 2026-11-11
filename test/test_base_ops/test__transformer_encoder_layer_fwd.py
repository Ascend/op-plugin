import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTransformerEncoderLayerFwd(TestCase):
    # ATen 算子 _transformer_encoder_layer_fwd：单层 Transformer Encoder。
    # 权重约定（对齐 transformer_encoder_layer_forward）：
    #   src: [B, S, D]
    #   qkv_weight: [3D, D], qkv_bias: [3D]
    #   proj_weight: [D, D], proj_bias: [D]
    #   norm_weight_1/2: [D], norm_bias_1/2: [D]
    #   ffn_weight_1: [4D, D], ffn_bias_1: [4D]
    #   ffn_weight_2: [D, 4D], ffn_bias_2: [D]
    # mask 为 bool，True == 屏蔽；mask_type: 1 == padding mask [B, S]。

    PREC = 1.e-3       # fp32 判定阈值
    PREC16 = 1.e-2     # fp16 判定阈值

    def _fwd(self, src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
             use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2,
             ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type):
        return torch.ops.aten._transformer_encoder_layer_fwd(
            src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
            use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2,
            ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type)

    def _gen_inputs(self, batch, seq_len, embed_dim, num_heads, dtype):
        inner = 4 * embed_dim
        inputs = {}
        for name, shape, low, high in [
            ("src", [batch, seq_len, embed_dim], -1.0, 1.0),
            ("qkv_weight", [3 * embed_dim, embed_dim], -0.1, 0.1),
            ("qkv_bias", [3 * embed_dim], -0.1, 0.1),
            ("proj_weight", [embed_dim, embed_dim], -0.1, 0.1),
            ("proj_bias", [embed_dim], -0.1, 0.1),
            ("norm_weight_1", [embed_dim], 0.5, 1.5),
            ("norm_bias_1", [embed_dim], -0.1, 0.1),
            ("norm_weight_2", [embed_dim], 0.5, 1.5),
            ("norm_bias_2", [embed_dim], -0.1, 0.1),
            ("ffn_weight_1", [inner, embed_dim], -0.1, 0.1),
            ("ffn_bias_1", [inner], -0.1, 0.1),
            ("ffn_weight_2", [embed_dim, inner], -0.1, 0.1),
            ("ffn_bias_2", [embed_dim], -0.1, 0.1),
        ]:
            cpu, npu = create_common_tensor([dtype, -1, shape], low, high)
            inputs[name] = (cpu, npu)
        return inputs

    def _padding_mask(self, batch, seq_len, pad_len):
        mask = np.zeros((batch, seq_len), dtype=np.bool_)
        if pad_len > 0:
            mask[:, -pad_len:] = True
        return torch.from_numpy(mask)

    def _run(self, inputs, embed_dim, num_heads, use_gelu, norm_first,
             mask_cpu=None, mask_npu=None, mask_type=None):
        """双端执行并断言：CPU 参考实现 vs NPU 实现（内部 MHA 命中 F1 融合路径）。

        mask 按设备分开传入（mask_cpu / mask_npu），避免跨设备 tensor 直接调用。
        """
        def call(idx, mask):
            return self._fwd(inputs["src"][idx], embed_dim, num_heads,
                             inputs["qkv_weight"][idx], inputs["qkv_bias"][idx],
                             inputs["proj_weight"][idx], inputs["proj_bias"][idx],
                             use_gelu, norm_first, 1e-5,
                             inputs["norm_weight_1"][idx], inputs["norm_bias_1"][idx],
                             inputs["norm_weight_2"][idx], inputs["norm_bias_2"][idx],
                             inputs["ffn_weight_1"][idx], inputs["ffn_bias_1"][idx],
                             inputs["ffn_weight_2"][idx], inputs["ffn_bias_2"][idx],
                             mask, mask_type)

        cpu_out = call(0, mask_cpu)
        npu_out = call(1, mask_npu)
        self.assertRtolEqual(cpu_out, npu_out.cpu(), prec=self.PREC, prec16=self.PREC16)

    def test_transformer_encoder_layer_fwd_norm_first_float32(self):
        batch, seq_len, embed_dim, num_heads = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_heads, np.float32)
        self._run(inputs, embed_dim, num_heads, False, True)

    def test_transformer_encoder_layer_fwd_norm_last_float32(self):
        batch, seq_len, embed_dim, num_heads = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_heads, np.float32)
        self._run(inputs, embed_dim, num_heads, False, False)

    def test_transformer_encoder_layer_fwd_gelu_float32(self):
        batch, seq_len, embed_dim, num_heads = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_heads, np.float32)
        self._run(inputs, embed_dim, num_heads, True, True)

    def test_transformer_encoder_layer_fwd_padding_mask_float32(self):
        batch, seq_len, embed_dim, num_heads = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_heads, np.float32)
        mask = self._padding_mask(batch, seq_len, pad_len=2)
        self._run(inputs, embed_dim, num_heads, False, True,
                  mask_cpu=mask, mask_npu=mask.to("npu"), mask_type=1)

    def test_transformer_encoder_layer_fwd_norm_first_float16(self):
        batch, seq_len, embed_dim, num_heads = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_heads, np.float16)
        self._run(inputs, embed_dim, num_heads, False, True)

    def test_transformer_encoder_layer_fwd_padding_mask_float16(self):
        batch, seq_len, embed_dim, num_heads = 2, 8, 16, 4
        inputs = self._gen_inputs(batch, seq_len, embed_dim, num_heads, np.float16)
        mask = self._padding_mask(batch, seq_len, pad_len=2)
        self._run(inputs, embed_dim, num_heads, False, True,
                  mask_cpu=mask, mask_npu=mask.to("npu"), mask_type=1)


if __name__ == "__main__":
    run_tests()

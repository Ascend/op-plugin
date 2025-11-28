import torch
import torch_npu
import unittest
import numpy as np
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

def genrate_input(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens):
    k = key.to(torch.float32)
    q = query.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()

    beta = torch.ones(T, n_heads_v).to(torch.float32) if beta is None else beta.to(torch.float32)
    o = torch.empty_like(v).to(torch.float32)
    if scale is None:
        scale = k.shape[-1]**-0.5
    q = q * scale

    seq_start = 0
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]]

        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + actual_seq_lengths[i]):
                
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_k)]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_k)]
                v_i = v[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                beta_i = beta[slot_id][head_id]
                S = S * alpha_i

                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S_ = y[:, None] * k_i[None, :]
                S = S + S_
                initial_state[ssm_state_indices[slot_id]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(torch.bfloat16), initial_state.to(torch.bfloat16)

class TestRecurrentGatedDeltaRule(TestCase):
    @unittest.skip('Skip test temporarily: CANN version-related operators not supported yet')
    @SupportedDevices(["Ascend910B"])
    def test_recurrent_gated_delta_rule_1(self, device="npu"):
        (b, mtp, nk, nv, dk, dv) = (64, 2, 4, 8, 128, 128)

        actual_seq_lengths = (torch.ones(b) * mtp).npu().to(torch.int32)
        T = int(torch.sum(actual_seq_lengths))
        state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
        query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
        key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
        value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
        g = torch.rand((T, nv), dtype=torch.float32).npu()
        beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()
        ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        key = torch.nn.functional.normalize(key, p=2, dim=-1)
        scale = 0.5
        num_accepted_tokens = torch.randint(1, mtp + 1, (b,)).npu().to(torch.int32)

        out_golden, state_golden = genrate_input(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens)
        out_golden = out_golden.to(torch.float32).cpu()
        state_golden = state_golden.to(torch.float32).cpu()

        state_copy = state.clone()
        out = torch_npu.npu_recurrent_gated_delta_rule(query, key, value, state_copy, beta=beta, scale=scale, actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices, g=g, num_accepted_tokens=num_accepted_tokens)
        out = out.to(torch.float32).cpu()

        self.assertRtolEqual(out_golden, out, 0.001)

        state_copy = state.clone()
        out, state_out = torch_npu.npu_recurrent_gated_delta_rule_functional(query, key, value, state_copy, beta=beta, scale=scale, actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices, g=g, num_accepted_tokens=num_accepted_tokens)
        out = out.to(torch.float32).cpu()
        state_out = state_out.to(torch.float32).cpu()
        
        self.assertRtolEqual(out_golden, out, 0.001)
        self.assertRtolEqual(state_golden, state_out, 0.001)

if __name__ == "__main__":
    run_tests()

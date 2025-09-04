import os
import re

import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, create_common_tensor


class TestNPUAdvanceStepFlashattn(TestCase):

    # pylint:disable = huawei-too-many-arguments
    def advance_step_flashattn_golden(self, num_seqs, num_queries, block_size, input_tokens, sampled_token_ids, 
                                     input_positions, seq_lens, slot_mapping, block_tables):
        num_core = 40
        cmd = 'npu-smi info -t common -i 0'
        test_list = os.popen(cmd).readlines()
        for line in test_list:
            if re.search('Aicore Count', line):
                num_core = int(line[-3:]) * 2

        for block_idx in range(num_core):
            n_pad = num_seqs - num_queries
            if n_pad > 0 and block_idx == 0:
                offset = num_queries
                i = 0
                while i < n_pad:
                    input_tokens[offset + i] = 0
                    input_positions[offset + i] = 0
                    slot_mapping[offset + i] = -1
                    i += num_core
            num_query_blocks = num_queries // 1
            if block_idx >= num_query_blocks:
                break
            cur_query_id = block_idx * 1 + 0
            if cur_query_id >= num_queries:
                break
            
            input_tokens[cur_query_id] = sampled_token_ids[cur_query_id]
            seq_len = seq_lens[cur_query_id]
            next_seq_len = seq_len + 1
            next_input_pos = next_seq_len - 1
            seq_lens[cur_query_id] = next_seq_len
            input_positions[cur_query_id] = next_input_pos
            
            block_index = next_input_pos // block_size
            block_offset = next_input_pos % block_size
            cur_block = block_tables.flatten()[block_index]
            slot_num = (cur_block + block_tables.stride(0) * cur_query_id) * block_size + block_offset
            
            slot_mapping[cur_query_id] = slot_num

    def advance_step_spec_flashattn_golden(self, num_seqs, num_queries, block_size, input_tokens, sampled_token_ids, 
                                     input_positions, seq_lens, slot_mapping, block_tables, spec_token, accepted_num):
        token_each_reqs = 1 + len(spec_token[0])
        input_positions += torch.repeat_interleave(accepted_num, token_each_reqs) + 1
        seq_lens.copy_((input_positions + 1).to(seq_lens.dtype))
        index = torch.argmin(
            torch.cat([
                sampled_token_ids,
                torch.full((num_seqs, 1), -1, device=sampled_token_ids.device)
            ], dim=1),
            dim=1
        ) - 1
        last_tokens = sampled_token_ids[torch.arange(num_seqs), index]
        if token_each_reqs == 1:
            input_tokens[:num_seqs] = last_tokens.to(dtype=input_tokens.dtype)
        else:
            input_tokens_2d = input_tokens.view(-1, token_each_reqs)
            input_tokens_2d[:num_seqs, 0] = last_tokens
            input_tokens_2d[:num_seqs, 1:] = spec_token
        req_indices = torch.repeat_interleave(
            torch.arange(num_seqs),
            token_each_reqs,
            dim=0
        )
        max_num_blocks_per_req = block_tables.shape[1]
        block_tables_indices = (
            req_indices * max_num_blocks_per_req +
            input_positions // block_size
        )
        block_numbers = block_tables.flatten()[block_tables_indices]
        block_offset = input_positions % block_size
        slot_mapping.copy_(block_numbers * block_size + block_offset)

    @unittest.skip("Skipping test_npu_advance_step_flashattn for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_advance_step_flashattn(self):
        iter_num = 7
        for i in range(iter_num):
            num_seqs = 2 * i + 2
            num_queries = i + 1
            block_size = i + 1
            shape_format_num_seqs = [np.int64, 2, [num_seqs, ]]
            shape_format_num_queries = [np.int64, 2, [num_queries, 1]]

            input_tokens_cpu, input_tokens = create_common_tensor(shape_format_num_seqs, 5, 50)
            sampled_token_ids_cpu, sampled_token_ids = create_common_tensor(shape_format_num_queries, 5, 50)
            input_positions_cpu, input_positions = create_common_tensor(shape_format_num_seqs, 5, 50)
            seq_lens_cpu, seq_lens = create_common_tensor(shape_format_num_seqs, 5, 50)
            slot_mapping_cpu, slot_mapping = create_common_tensor(shape_format_num_seqs, 5, 50)
            
            shape_format_block_tables = [np.int64, 2, [num_seqs, torch.max(seq_lens_cpu) // block_size + 1]]
            block_tables_cpu, block_tables = create_common_tensor(shape_format_block_tables, 5, 50)
            
            self.advance_step_flashattn_golden(num_seqs, num_queries, block_size, input_tokens_cpu,
                                               sampled_token_ids_cpu, input_positions_cpu, seq_lens_cpu,
                                               slot_mapping_cpu, block_tables_cpu)
            torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions,
                                                 seq_lens, slot_mapping, block_tables, num_seqs,
                                                 num_queries, block_size)
            self.assertRtolEqual(input_tokens_cpu, input_tokens.cpu())
            self.assertRtolEqual(input_positions_cpu, input_positions.cpu())
            self.assertRtolEqual(seq_lens_cpu, seq_lens.cpu())
            self.assertRtolEqual(slot_mapping_cpu, slot_mapping.cpu())

    @unittest.skip("Skip until CANN is updated to 8.3.RC1 to support aclnnAdvanceStepV2")
    @SupportedDevices(['Ascend910B'])
    def test_npu_advance_step_spec_flashattn(self):
        iter_num = 7
        for i in range(iter_num):
            num_seqs = 2 * i + 2
            block_size = 8
            spec_num = 3
            shape_format_num_seqs = [np.int64, 2, [num_seqs * (spec_num + 1), ]]
            shape_format_spec_token = [np.int64, 2, [num_seqs, spec_num]]
            shape_format_accepted_num = [np.int64, 2, [num_seqs, ]]
            shape_format_sampled_token = [np.int64, 2, [num_seqs, spec_num + 1]]

            input_tokens_cpu, input_tokens = create_common_tensor(shape_format_num_seqs, 5, 50)
            sampled_token_ids_cpu, sampled_token_ids = create_common_tensor(shape_format_sampled_token, 5, 50)
            input_positions_cpu, input_positions = create_common_tensor(shape_format_num_seqs, 5, 50)
            seq_lens_cpu, seq_lens = create_common_tensor(shape_format_num_seqs, 5, 50)
            slot_mapping_cpu, slot_mapping = create_common_tensor(shape_format_num_seqs, 5, 50)
            spec_token_cpu, spec_token = create_common_tensor(shape_format_spec_token, 5, 50)
            accepted_num_cpu, accepted_num = create_common_tensor(shape_format_accepted_num, 5, 50)
            
            shape_format_block_tables = [np.int64, 2, [num_seqs, 10000]]
            block_tables_cpu, block_tables = create_common_tensor(shape_format_block_tables, 5, 50)
            
            self.advance_step_spec_flashattn_golden(num_seqs, num_seqs, block_size, input_tokens_cpu,
                                               sampled_token_ids_cpu, input_positions_cpu, seq_lens_cpu,
                                               slot_mapping_cpu, block_tables_cpu, spec_token_cpu, accepted_num_cpu)
            torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions,
                                                 seq_lens, slot_mapping, block_tables, num_seqs,
                                                 num_seqs, block_size, spec_token=spec_token, accepted_num=accepted_num)
            self.assertRtolEqual(input_tokens_cpu, input_tokens.cpu())
            self.assertRtolEqual(input_positions_cpu, input_positions.cpu())
            self.assertRtolEqual(seq_lens_cpu, seq_lens.cpu())
            self.assertRtolEqual(slot_mapping_cpu, slot_mapping.cpu())

if __name__ == "__main__":
    run_tests()

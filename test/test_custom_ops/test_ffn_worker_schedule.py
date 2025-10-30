import unittest
import torch
import torch_npu
import os
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

window_size = 209715200
ffn_window_tensor = torch.zeros([window_size], dtype=torch.int8).npu()

attn_workers = 2
micro_batch_number = 3
batch_size = 6
top_k = 8
hidden_size = 7168
expert_num = 288
attn_to_ffn_token_size = (7168 + 4 + 511) // 512 * 512
ffn_to_attn_token_size = 7168 * 2
ffn_window = ffn_window_tensor.data_ptr()

context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode = 0, session_num = attn_workers,
    micro_batch_num = micro_batch_number, micro_batch_size = batch_size, selected_expert_num = top_k + 1,
    expert_num = expert_num, attn_to_ffn_token_size = attn_to_ffn_token_size, ffn_to_attn_token_size = ffn_to_attn_token_size, 
    ffn_window = ffn_window, ffn_window_size = window_size)
    
schedule_context = context_holder.get_schedule_context_tensor()

def _set_all_flags():
    num_int8 = attn_workers * micro_batch_number * (8 + batch_size * top_k * 4)
    per_session_num = micro_batch_number * (8 + batch_size * top_k * 4)
    int32_view = ffn_window_tensor[:num_int8].view(torch.int32)
    int32_view[:] = 1


class TestFfnWorkerScheduler(TestCase):
    @unittest.skip("skip case until cann supported")
    @SupportedDevices(['Ascend910B'])
    def test_ffn_worker_scheduler_(self):
        _set_all_flags()
        schedule_context1 = schedule_context.clone()
        torch_npu.ffn_worker_scheduler_(schedule_context, sync_group_size = 2)
        self.assertNotEqual(schedule_context1, schedule_context)

    @unittest.skip("skip case until cann supported")
    @SupportedDevices(['Ascend910B'])
    def test_ffn_worker_scheduler(self):
        _set_all_flags()
        schedule_context1 = schedule_context.clone()
        schedule_context2 = torch_npu.ffn_worker_scheduler(schedule_context, sync_group_size = 2)
        self.assertEqual(schedule_context1, schedule_context)
        self.assertNotEqual(schedule_context2, schedule_context)


if __name__ == '__main__':
    run_tests()

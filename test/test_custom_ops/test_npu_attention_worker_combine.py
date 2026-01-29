import unittest
import sys
import random
import os
import ctypes
from multiprocessing import Pool, cpu_count
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

A = 4
M = 1
BS = 54
K_plus_1 = 9
H = 7168
expert_num = 2  # 从 0 ~ 1024
out_num = A # 需要小于等于A
tokenDtype = 2
need_schedule = 1
if need_schedule != 0:
    need_schedule = 1
Y = A * BS * K_plus_1
HS = H * 2

F = (1 + 1 + BS * K_plus_1) # 表示个数，都是int32类型
cur_micro_batch_id = 0

# global参数
uniq_expert_id_cnt = 0

# ---------- 结构体定义（必须与生成时完全一致） ----------


class CommonArea(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("session_num", ctypes.c_uint32),
        ("micro_batch_num", ctypes.c_uint32),
        ("micro_batch_size", ctypes.c_uint32),
        ("selected_expert_num", ctypes.c_uint32),
        ("expert_num", ctypes.c_uint32),
        ("attn_to_ffn_token_size", ctypes.c_uint32),
        ("ffn_to_attn_token_size", ctypes.c_uint32),
        ("schedule_mode", ctypes.c_int32),
        ("reserve0", ctypes.c_int8 * 96)
    ]


class ControlArea(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("run_flag", ctypes.c_int32),
        ("reserve1", ctypes.c_int8 * 4),
        ("schedule_wait_buf", ctypes.c_uint64),
        ("ffn_wait_buf", ctypes.c_uint64),
        ("attention_wait_buf", ctypes.c_uint64),
        ("reserve2", ctypes.c_int8 * 96)
    ]


class AttentionArea(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("token_info_buf", ctypes.c_uint64),
        ("token_info_buf_size", ctypes.c_uint64),
        ("token_data_buf", ctypes.c_uint64),
        ("token_data_buf_size", ctypes.c_uint64),
        ("micro_batch_id", ctypes.c_uint32),
        ("reserve5", ctypes.c_int8 * 92)
    ]


class FfnArea(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("token_info_buf", ctypes.c_uint64),
        ("token_info_buf_size", ctypes.c_uint64),
        ("token_data_buf", ctypes.c_uint64),
        ("token_data_buf_size", ctypes.c_uint64),
        ("polling_index", ctypes.c_uint64),
        ("reserve3", ctypes.c_int8 * 88),
        ("layer_ids_buf", ctypes.c_uint64),
        ("layer_ids_buf_size", ctypes.c_uint64),
        ("session_ids_buf", ctypes.c_uint64),
        ("session_ids_buf_size", ctypes.c_uint64),
        ("micro_batch_ids_buf", ctypes.c_uint64),
        ("micro_batch_ids_buf_size", ctypes.c_uint64),
        ("expert_ids_buf", ctypes.c_uint64),
        ("expert_ids_buf_size", ctypes.c_uint64),
        ("out_num", ctypes.c_uint32),
        ("reserve4", ctypes.c_int8 * 60)
    ]


class DataDesc(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("flag", ctypes.c_int32),
        ("layer_id", ctypes.c_int32),
        ("expert_ids", ctypes.c_int32 * (BS * K_plus_1))
    ]


class TokenData(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        # 改为float16数组（每个元素占2字节）
        ("hidden_states", ctypes.c_uint16 * H),  # H是个数
        ("padding", ctypes.c_uint8 * (HS - H * 2))  # 保持总大小为HS Bytes
    ]


class ScheduleContext(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("common", CommonArea),
        ("control", ControlArea),
        ("attention", AttentionArea),
        ("ffn", FfnArea),
        ("reserve6", ctypes.c_int8 * 384)
    ]


def golden_attention_worker_combie(token_data, micro_batch_id, expert_scales, layer_id):
    micro_batch, batch_seq, k_add_1, token_length = token_data.shape
    k = k_add_1 - 1
    y = torch.zeros((batch_seq, token_length), dtype=token_data.dtype)

    expert_scales_unsqzd = expert_scales.unsqueeze(-1)

    token_data_fp32 = token_data.float()
    mul_result = token_data_fp32[micro_batch_id, :, :k, :] * expert_scales_unsqzd
    y = torch.sum(mul_result, dim=-2)
    y += token_data_fp32[micro_batch_id, :, k, :]
    y = y.half()
    next_layer_id = layer_id + 1
    return y, next_layer_id


class TestAttentionWorkerCombine(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_npu_attention_worker_combine_001(self):
        expert_scales = torch.rand(BS, K_plus_1 - 1, dtype=torch.float32)
        layer_id = torch.randint(1, 20, (1,), dtype=torch.int32)
        expert_scales_npu = copy.deepcopy(expert_scales).npu()
        layer_id_npu = copy.deepcopy(layer_id).npu()

        ctx = ScheduleContext()
        # 填充CommonArea
        ctx.common.session_num = A
        ctx.common.micro_batch_num = M
        ctx.common.micro_batch_size = BS
        ctx.common.selected_expert_num = K_plus_1
        ctx.common.expert_num = expert_num
        ctx.common.attn_to_ffn_token_size = HS
        ctx.common.ffn_to_attn_token_size = 512
        ctx.common.schedule_mode = 2
        ctypes.memset(ctx.common.reserve0, 0, 96)

        # 填充ControlArea
        ctx.control.run_flag = 1
        ctypes.memset(ctx.control.reserve1, 0, 4)
        ctx.control.schedule_wait_buf = 0x1000
        ctx.control.ffn_wait_buf = 0x2000
        ctx.control.attention_wait_buf = 0x3000
        ctypes.memset(ctx.control.reserve2, 0, 96)

        # 填充FfnArea
        ctx.ffn.token_info_buf = 0x4000
        ctx.ffn.token_info_buf_size = 1024
        ctx.ffn.token_data_buf = 0x5000
        ctx.ffn.token_data_buf_size = 2048
        ctx.ffn.polling_index = 1
        ctypes.memset(ctx.ffn.reserve3, 0, 88)
        ctx.ffn.layer_id_buf = 0x6000
        ctx.ffn.layer_id_buf_size = 1024
        ctx.ffn.session_ids_buf = 0x7000
        ctx.ffn.session_ids_buf_size = 1024
        ctx.ffn.expert_ids_buf = 0x8000
        ctx.ffn.expert_ids_buf_size = 1024
        ctx.ffn.out_num = 1
        ctypes.memset(ctx.ffn.reserve4, 0, 60)

        # 生成FfnArea数据
        ffn_data_parts = []

        # 生成各部分数据
        # 1. token_info数据
        token_info_tensors = torch.ones((M, BS, K_plus_1), dtype=torch.int32)
        token_info_ptr = token_info_tensors.data_ptr()
        flagArrayType = ctypes.c_int32 * (M * BS * K_plus_1)
        flagArray = ctypes.cast(token_info_ptr, ctypes.POINTER(flagArrayType)).contents
        token_info_bytes = token_info_tensors.numpy().tobytes()

        # 2. token_data数据
        token_data = torch.rand(M, BS, K_plus_1, H, dtype=torch.half)
        token_data_ptr = token_data.data_ptr()
        ArrayType = ctypes.c_int16 * (H * M * BS * K_plus_1)
        array = ctypes.cast(token_data_ptr, ctypes.POINTER(ArrayType)).contents
        token_data_bytes = token_data.numpy().tobytes()

        ctx.attention.token_info_buf = token_info_tensors.npu().data_ptr()
        ctx.attention.token_info_buf_size = len(token_info_bytes)
        ctx.attention.token_data_buf = token_data.npu().data_ptr()
        ctx.attention.token_data_buf_size = len(token_data_bytes)
    
        ctx.attention.micro_batch_id = 0

        struct_size = ctypes.sizeof(ctx)
        full_buffer = bytearray(struct_size)
        struct_bytes = ctypes.string_at(ctypes.addressof(ctx), struct_size)
        full_buffer[0:struct_size] = struct_bytes

        schedule_context_tensor = torch.frombuffer(full_buffer, dtype=torch.uint8).view(torch.int8)
        schedule_context_npu = schedule_context_tensor.clone().npu()

        y_cpu, next_layer_id_cpu = golden_attention_worker_combie(token_data, ctx.attention.micro_batch_id, expert_scales, layer_id)

        y_npu, next_layer_id_npu = \
            torch_npu.npu_attention_worker_combine(schedule_context_npu, expert_scales_npu, layer_id_npu, H, token_dtype=torch.half, need_schedule=0)
        torch_npu.npu.synchronize()
        self.assertRtolEqual(y_cpu, y_npu)


if __name__ == "__main__":
    run_tests()

import unittest
import sys
import random
import os
import ctypes
from multiprocessing import Pool, cpu_count
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
invaildProb = 0 # 0.75 #0.993
if need_schedule != 0:
    need_schedule = 1

Y = A * BS * K_plus_1
HS = H * 2
# 带quant时存储的是int8类型且后面带一个fp32的scale，整体要对齐到512 Bytes
if tokenDtype == 2:
    HS = (H + 4 + 512 - 1) // 512 * 512

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

if tokenDtype == 2:
    class TokenDataQuant(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            # 改为c_int8数组（每个元素占1字节）
            ("hidden_states", ctypes.c_int8 * H),  # H是个数
            ("quant_scale", ctypes.c_float),
            ("padding", ctypes.c_uint8 * (HS - H - 4))  # 保持总大小为HS
        ]
else:
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


def read_schedule_context(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()

    # 解析主结构体
    ctx = ScheduleContext.from_buffer_copy(file_content[:ctypes.sizeof(ScheduleContext)])
    parsed_data = {}

    # 解析FFN区域的数据块
    def parse_data_block(buf_field, size_field, data_type):
        offset = getattr(ctx.ffn, buf_field)
        size = getattr(ctx.ffn, size_field)
        if offset == 0 or size == 0:
            return None
        data_bytes = file_content[offset: offset + size]
        element_size = ctypes.sizeof(data_type)
        num_elements = size // element_size
        array_type = data_type * num_elements
        return array_type.from_buffer_copy(data_bytes)

    # token_info_buf
    parsed_data['token_info_buf'] = parse_data_block('token_info_buf', 'token_info_buf_size', DataDesc)
    if tokenDtype == 2:
        # TokenData数组
        parsed_data['token_data_buf'] = parse_data_block('token_data_buf', 'token_data_buf_size', TokenDataQuant)
    else:
        # TokenData数组
        parsed_data['token_data_buf'] = parse_data_block('token_data_buf', 'token_data_buf_size', TokenData)
    # Session IDs (int32数组)
    parsed_data['session_ids_buf'] = parse_data_block('session_ids_buf', 'session_ids_buf_size', ctypes.c_int32)
    # Micro Batch IDs (int64数组)
    parsed_data['micro_batch_ids_buf'] = parse_data_block('micro_batch_ids_buf', 'micro_batch_ids_buf_size',
                                                          ctypes.c_int32)
    # Expert IDs (int32数组)
    parsed_data['expert_ids_buf'] = parse_data_block('expert_ids_buf', 'expert_ids_buf_size', ctypes.c_int32)

    return ctx, parsed_data


def convert_ctypes_to_torch(ctypes_array, expected_dtype):
    # 类型映射表（ctypes到torch）
    type_map = {
        ctypes.c_int32: torch.int32,
        ctypes.c_float: torch.float32,
        ctypes.c_uint8: torch.uint8,
        ctypes.c_int64: torch.int64,
    }

    # 验证输入类型
    if not isinstance(ctypes_array, ctypes.Array):
        raise TypeError("输入必须是ctypes数组")

    # 获取原始内存指针和元素数量
    ptr = ctypes.addressof(ctypes_array)
    num_elements = len(ctypes_array)

    # 构建张量（零拷贝）
    dtype = type_map.get(ctypes_array._type_, None)
    if dtype is None:
        raise ValueError(f"不支持的ctypes类型: {ctypes_array._type_}")

    # 通过内存地址直接构建张量
    tensor = torch.frombuffer(
        (ctypes.c_byte * num_elements * ctypes.sizeof(ctypes_array._type_)).from_address(ptr),
        dtype=dtype
    )

    return tensor.clone()  # 创建独立内存副本（确保数据安全）


# 生成随机数列表 (不填充无效值)
def create_random_c_array2(array_length, min_val, max_val, c_type=ctypes.c_int32, extreme_val=None):
    """
    生成一个C语言风格的数组，内容为指定范围的随机整数,
    参数:
        array_length: 数组长度
        min_val: 随机数最小值（包含）
        max_val: 随机数最大值（包含）
        c_type: C数据类型，默认为ctypes.c_int32
        max_val: 极大值 (默认自动选择c_type的最大值)
    返回:
        C风格的数组（指针可直接传递给C函数）
    """
    random_values = [random.randint(min_val, max_val) for _ in range(array_length)]

    # 创建C数组并填充随机值
    c_array = (c_type * array_length)(*random_values)
    return c_array


# 生成随机数列表 (填充无效值)
def create_random_c_array(array_length, min_val, max_val, c_type=ctypes.c_int32, prob=0.0, extreme_val=None):
    """
    生成一个C语言风格的数组，内容为指定范围的随机整数,
    参数:
        array_length: 数组长度
        min_val: 随机数最小值（包含）
        max_val: 随机数最大值（包含）
        c_type: C数据类型，默认为ctypes.c_int32
        prob: 元素被替换为极大值的概率 (默认0.0)
        max_val: 极大值 (默认自动选择c_type的最大值)
    返回:
        C风格的数组（指针可直接传递给C函数）
    """
    # 生成随机数列表
    if extreme_val is None:
        extreme_val = 214748364
    random_values = [random.randint(min_val, max_val) for _ in range(array_length)]

    if prob > 0.0:
        a = 0
        for i in range(array_length):
            if random.random() < prob and a < 1512:
                random_values[i] = extreme_val
                a = a + 1

    # 创建C数组并填充随机值
    c_array = (c_type * array_length)(*random_values)
    return c_array


def generate_token_data_element(_):
    """生成单个TokenData结构体元素（支持float16）""" 
    if tokenDtype == 2:
        token = TokenDataQuant()
        # 1. 生成随机 int8 数据
        int8_values = np.random.randint(-10, 10, size=H).astype(np.int8)
        # 2. 存入结构体
        ctypes.memmove(token.hidden_states, int8_values.ctypes.data, ctypes.sizeof(token.hidden_states))
        token.quant_scale = np.random.uniform(0.1, 2.0)
        ctypes.memset(ctypes.addressof(token.padding), 0, HS - H - 4)
    else:
        # 使用ctypes.memmove 有問題
        token = TokenData()
        # 生成随机的float16 hidden_states
        float16_values = np.random.uniform(-1.0, 1.0, size=H).astype(np.float16)
        ctypes.memmove(token.hidden_states, float16_values.ctypes.data, ctypes.sizeof(token.hidden_states))
        ctypes.memset(ctypes.addressof(token.padding), 0, HS - H * 2)
    return token


def generate_token_info_element(AM_idx):
    """生成单个TokenInfo结构体元素"""
    tokenInfoDes = DataDesc()
    if AM_idx % M == cur_micro_batch_id:
        tokenInfoDes.flag = 1
        cur_expert_ids = create_random_c_array((BS * K_plus_1), 0, 1, prob=invaildProb) # expert_num - 1
    else:
        tokenInfoDes.flag = 0
        cur_expert_ids = create_random_c_array((BS * K_plus_1), 0, 1, prob=invaildProb) # expert_num - 1
    tokenInfoDes.expert_ids = cur_expert_ids
    return tokenInfoDes


def generate_token_info():
    """生成完整的toke_info数据（并行优化版）"""
    total_elements = A * M

    # 使用更智能的chunksize（避免过多小任务）
    chunksize = max(1000, total_elements // (cpu_count() * 4))
    with Pool(cpu_count()) as pool:
        results = list(pool.imap(
            generate_token_info_element,
            range(total_elements),
            chunksize=chunksize
        ))
    #  对于生成的没有无效值的进行填充，一个整块bs*k 为有效，一个整块为无效，剩余按新概率填充无效值，整体保持无效值概率不变
    return (DataDesc * total_elements)(*results)


def generate_token_data():
    """生成完整的token_data数据（并行优化版）"""
    total_elements = A * M * BS * K_plus_1

    # 使用更智能的chunksize（避免过多小任务）
    chunksize = max(1000, total_elements // (cpu_count() * 4))

    with Pool(cpu_count()) as pool:
        results = list(pool.imap(
            generate_token_data_element,
            range(total_elements),
            chunksize=chunksize
        ))
    if tokenDtype == 2:
        arr_type = TokenDataQuant * total_elements
    else:
        arr_type = TokenData * total_elements
    return arr_type(*results)

tensor_buffers = {}
ffn_data_parts = []
def generate_input_func():
    ctx = ScheduleContext()
    # 填充CommonArea
    ctx.common.session_num = A
    ctx.common.micro_batch_num = M
    ctx.common.micro_batch_size = BS
    ctx.common.selected_expert_num = K_plus_1
    ctx.common.expert_num = expert_num
    ctx.common.attn_to_ffn_token_size = HS
    ctx.common.ffn_to_attn_token_size = 512
    ctx.common.schedule_mode = 1
    ctypes.memset(ctx.common.reserve0, 0, 96)

    # 填充ControlArea
    ctx.control.run_flag = 1
    ctypes.memset(ctx.control.reserve1, 0, 4)
    ctx.control.schedule_wait_buf = 0x1000
    ctx.control.ffn_wait_buf = 0x2000
    ctx.control.attention_wait_buf = 0x3000
    ctypes.memset(ctx.control.reserve2, 0, 96)

    # 填充AttentionArea
    ctx.attention.token_info_buf = 0x4000
    ctx.attention.token_info_buf_size = 1024
    ctx.attention.token_data_buf = 0x5000
    ctx.attention.token_data_buf_size = 2048
    ctx.attention.micro_batch_id = 1
    ctypes.memset(ctx.attention.reserve5, 0, 92)

    # 生成各部分数据
    # 1. token_info数据
    if need_schedule == 1:
        token_info = generate_token_info()
        token_info_bytes = bytes(token_info)
        int32_array = np.frombuffer(token_info_bytes, dtype=np.int32).copy()
        np.savetxt('tokeninfo_org.txt', int32_array, fmt='%d', delimiter=',')
        ffn_data_parts.append(("token_info", token_info_bytes))

    # 2. token_data数据
    token_data = generate_token_data()
    token_data_bytes = bytes(token_data)
    ffn_data_parts.append(("token_data", token_data_bytes))

    # 4. session_ids数据
    random_session_ids = [i for i in range(A)]
    session_ids = (ctypes.c_int32 * A)(*random_session_ids)
    ffn_data_parts.append(("session_ids", session_ids))

    # 5. micro_batch_ids数据
    random_micro_batch_ids = [cur_micro_batch_id for _ in range(M)]
    micro_batch_ids = (ctypes.c_int32 * M)(*random_micro_batch_ids)
    ffn_data_parts.append(("micro_batch_ids", micro_batch_ids))

    if need_schedule == 0:
        expert_ids = create_random_c_array((out_num * BS * K_plus_1), 0, 1, prob=invaildProb) # expert_num - 1, prob=0.9)
        ffn_data_parts.append(("expert_ids", expert_ids))

    # 计算正确偏移量
    current_offset = ctypes.sizeof(ctx)

    # 设置FfnArea字段
    for name, data in ffn_data_parts:
        data_bytes = bytes(data) if isinstance(data, ctypes.Array) else data
        size = len(data_bytes)

        if name == "token_info":
            ctx.ffn.token_info_buf = current_offset
            ctx.ffn.token_info_buf_size = size
        elif name == "token_data":
            ctx.ffn.token_data_buf = current_offset
            ctx.ffn.token_data_buf_size = size
        elif name == "session_ids":
            ctx.ffn.session_ids_buf = current_offset
            ctx.ffn.session_ids_buf_size = size
        elif name == "micro_batch_ids":
            ctx.ffn.micro_batch_ids_buf = current_offset
            ctx.ffn.micro_batch_ids_buf_size = size
        elif name == "expert_ids":
            ctx.ffn.expert_ids_buf = current_offset
            ctx.ffn.expert_ids_buf_size = size

        current_offset += size

    if need_schedule == 0:
        ctx.ffn.token_info_buf = 0
        ctx.ffn.token_info_buf_size = 0
    else:
        ctx.ffn.expert_ids_buf = 0
        ctx.ffn.expert_ids_buf_size = 0

    # 填充剩余字段
    ctx.ffn.polling_index = cur_micro_batch_id
    ctx.ffn.out_num = out_num

    # 写入文件
    with open("schedule_context.bin", "wb") as f:
        f.write(bytes(ctx))
        for _, data in ffn_data_parts:
            f.write(data if isinstance(data, bytes) else bytes(data))

    tensor_buffers.clear()
    for name, data in ffn_data_parts:
        data_bytes = bytes(data) if isinstance(data, ctypes.Array) else data

        if name in ["token_info", "session_ids", "micro_batch_ids", "expert_ids"]:
            np_array = np.frombuffer(data_bytes, dtype=np.int32).copy()
            tensor = torch.from_numpy(np_array).int()
        else:
            if tokenDtype == 2:
                np_array = np.frombuffer(data_bytes, dtype=np.int8).copy()
                tensor = torch.from_numpy(np_array).to(torch.int8)
            else:
                np_array = np.frombuffer(data_bytes, dtype=np.float16).copy()
                tensor = torch.from_numpy(np_array).to(torch.float16)
        
        tensor = tensor.npu()
        tensor_buffers[name] = tensor


def calc_expect_func():
    ctx, ffn_area = read_schedule_context("schedule_context.bin")

    if tokenDtype == 2:
        # 转换 dynamic_scales 确保从原始数据到PyTorch Tensor全程保持float32
        dynamic_scales_numpy = np.array(
            [token.quant_scale for token in ffn_area["token_data_buf"]],
            dtype=np.float32  # 显式指定NumPy类型
        )
        dynamic_scales = torch.from_numpy(dynamic_scales_numpy)  # 自动继承np.float32 -> torch.float32
        dynamic_scales = dynamic_scales.view(A, M, BS, K_plus_1, 1)

    if tokenDtype == 2:
        # 转换token_data（复杂结构体处理）
        uint8_data = np.array(
            [token.hidden_states for token in ffn_area["token_data_buf"]],
            dtype=np.int8  # 显式指定NumPy类型
        )

        # 原始代码（生成只读的 NumPy 数组）
        byte_stream = uint8_data.tobytes()
        int8_data = np.frombuffer(byte_stream, dtype=np.int8)

        # 修复：添加 .copy() 创建可写副本
        int8_data_writable = int8_data.copy()  # 关键步骤
        token_data = torch.from_numpy(int8_data_writable)  # 不再警告

        # 验证类型
        token_data = token_data.view(A, M, BS, K_plus_1, H)
    else:
        # 转换token_data（复杂结构体处理）
        uint16_data = np.array(
            [token.hidden_states for token in ffn_area["token_data_buf"]],
            dtype=np.uint16  # 显式指定NumPy类型
        )

        # 原始代码（生成只读的 NumPy 数组）
        byte_stream = uint16_data.tobytes()
        float16_data = np.frombuffer(byte_stream, dtype=np.float16)
        # 修复：添加 .copy() 创建可写副本
        float16_data_writable = float16_data.copy()  # 关键步骤
        token_data = torch.from_numpy(float16_data_writable)  # 不再警告
        # 验证类型
        token_data = token_data.view(A, M, BS, K_plus_1, H)
    
    if need_schedule == 0:
        # 转换专家ID（直接转PyTorch）
        expert_ids = convert_ctypes_to_torch(
            ffn_area["expert_ids_buf"],
            torch.int32
        )
    else:
        AM_idx = 0
        expert_ids_array = np.array([], dtype=np.int32)
        for token_info in ffn_area["token_info_buf"]:
            if AM_idx % M == cur_micro_batch_id:
                tmp_expert_ids = np.array(token_info.expert_ids, dtype=np.int32)
                expert_ids_array = np.concatenate((expert_ids_array, tmp_expert_ids))
            AM_idx += 1

        # 原始代码（生成只读的 NumPy 数组）
        byte_stream = expert_ids_array.tobytes()
        int32_data = np.frombuffer(byte_stream, dtype=np.int32)

        # 修复：添加 .copy() 创建可写副本
        int32_data_writable = int32_data.copy()  # 关键步骤
        expert_ids = torch.from_numpy(int32_data_writable)  # 不再警告

        # 验证类型
        expert_ids_shape = expert_ids.view(A, BS, K_plus_1)

    session_ids_buf = convert_ctypes_to_torch(
        ffn_area["session_ids_buf"],
        torch.int32
    )

    # 转换微批次ID
    micro_batch_ids_buf = convert_ctypes_to_torch(
        ffn_area["micro_batch_ids_buf"],
        torch.int32
    )

    # 后续可直接使用PyTorch操作
    masked_expert_ids = expert_ids.clone()  # 合法操作
    # 统计一下 最大值的个数
    masked_count = (masked_expert_ids >= 1e6).sum().item()
    zero_up_count = (masked_expert_ids >= 0).sum().item()
    # 展平专家ID并生成索引
    flat_expert_ids = masked_expert_ids.view(-1)
    # 根据专家ID排序
    sorted_indices = torch.argsort(flat_expert_ids, stable=True)
    sorted_expert_ids = flat_expert_ids[sorted_indices]

    # 生成Gather索引
    valid_count = len(sorted_indices) - masked_count
    actual_token_num = valid_count
    gather_idx = sorted_indices.to(torch.int32)
    # 计算输出维度
    # 初始化输出张量
    dtype_map = {0: torch.float16, 1: torch.bfloat16, 2: torch.int8, 3: torch.int8}
    y = torch.zeros(Y, H, dtype=dtype_map[0], device=token_data.device)
    # 计算数据收集索引
    a_indices = gather_idx // (BS * K_plus_1)
    remainder = gather_idx % (BS * K_plus_1)
    bs_indices = remainder // K_plus_1
    k_indices = remainder % K_plus_1
    # 使用高级索引进行数据收集
    session_indices = session_ids_buf[a_indices]
    micro_batch_indices = micro_batch_ids_buf[cur_micro_batch_id]
    # 构造多维索引
    token_data_indices = (session_indices, micro_batch_indices, bs_indices, k_indices)
    # 收集数据
    y = token_data[token_data_indices]
    if tokenDtype == 2:
        dynamic_scale = dynamic_scales[token_data_indices]

    # 生成group_list
    unique_experts, counts = torch.unique_consecutive(
        sorted_expert_ids, return_counts=True
    )
    group_list = torch.zeros(expert_num, 2, dtype=torch.int64, device=token_data.device)
    mask = unique_experts <= 1e6
    filtered_experts = unique_experts[mask]
    filtered_counts = counts[mask]
    group_list[:, 0] = filtered_experts
    group_list[:, 1] = filtered_counts

    global uniq_expert_id_cnt
    uniq_expert_id_cnt = len(filtered_experts)

    # 生成其他输出
    session_ids = session_ids_buf[a_indices]
    micro_batch_ids = micro_batch_ids_buf[cur_micro_batch_id]
    token_ids = bs_indices  # 简化处理
    expert_offsets = k_indices

    if tokenDtype == 2:
        return y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale.squeeze(), actual_token_num
    else:
        return y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, None, actual_token_num


class TestFfnWorkerBatching(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_worker_batching_001(self):
        generate_input_func()

        y_cpu, group_list_cpu, session_ids_cpu, micro_batch_ids_cpu, token_ids_cpu, expert_offsets_cpu, dynamic_scale_cpu, actual_token_num_cpu = calc_expect_func()

        schedule_context_bin = "./schedule_context.bin"
        ctx, ffn_area = read_schedule_context(schedule_context_bin)

        for name, data in ffn_data_parts:
            data_bytes = bytes(data) if isinstance(data, ctypes.Array) else data

            if name == "token_info":
                ctx.ffn.token_info_buf = tensor_buffers[name].data_ptr()
                ctx.ffn.token_info_buf_size = len(data_bytes)
            elif name == "token_data":
                ctx.ffn.token_data_buf = tensor_buffers[name].data_ptr()
                ctx.ffn.token_data_buf_size = len(data_bytes)
            elif name == "session_ids":
                ctx.ffn.session_ids_buf = tensor_buffers[name].data_ptr()
                ctx.ffn.session_ids_buf_size = len(data_bytes)
            elif name == "micro_batch_ids":
                ctx.ffn.micro_batch_ids_buf = tensor_buffers[name].data_ptr()
                ctx.ffn.micro_batch_ids_buf_size = len(data_bytes)
            elif name == "expert_ids":
                ctx.ffn.expert_ids_buf = tensor_buffers[name].data_ptr()
                ctx.ffn.expert_ids_buf_size = len(data_bytes)
        
        ctx_bytes = bytes(ctx)
        ctx_numpy = np.frombuffer(ctx_bytes, dtype=np.int8).copy()
        ctx_tensor = torch.from_numpy(ctx_numpy).to(torch.int8)
        schedule_context_npu = ctx_tensor.npu()

        max_out_shape = [A, BS, K_plus_1, H]
        y_npu, group_list_npu, session_ids_npu, micro_batch_ids_npu, token_ids_npu, expert_offsets_npu, dynamic_scale_npu, actual_token_num_npu = \
            torch_npu.npu_ffn_worker_batching(schedule_context_npu, expert_num, max_out_shape, token_dtype=tokenDtype, need_schedule=need_schedule, layer_num=0)
        torch_npu.npu.synchronize()
        self.assertRtolEqual(y_cpu, y_npu)


if __name__ == "__main__":
    run_tests()

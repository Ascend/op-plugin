import argparse
import itertools
import random
import dataclasses
from dataclasses import dataclass
import copy
import unittest
import numpy as np
import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# 定义 NumPy dtype 到 PyTorch dtype 的映射字典
NP_TO_TORCH_DTYPE = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.bool_: torch.bool,
}


@dataclass
class additionAttr:
    is_need_logits: bool = False
    top_k_guess: int = 32
    eps: float = 1e-8


def mySoftmaxAndSort(x, dim=-1):
    if dim < 0:
        dim = x.dim() + dim
    
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    shifted = x - max_vals
    exp_vals = torch.exp(shifted)
    softmax_output = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
    sorted_probs, sorted_indices = torch.sort(softmax_output, dim=dim, descending=True)

    return sorted_probs, sorted_indices
    

def onlySoftmax(x, dim=-1):
    if dim < 0:
        dim = x.dim() + dim
    
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    shifted = x - max_vals
    exp_vals = torch.exp(shifted)
    softmax_output = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)

    return softmax_output


def top_k_top_p_sample(data, run_attr: additionAttr):
    # 获取输入数组和参数
    logits_np = data[0]
    top_ks = data[1]
    top_ps = data[2]
    q_np = data[3]

    ALL_P_MAX = 1.0

    # Get runtime Attr
    is_need_logits = run_attr.is_need_logits
    top_k_guess = run_attr.top_k_guess
    eps = run_attr.eps

    # 验证输入数据类型
    if not isinstance(logits_np, np.ndarray) or not isinstance(top_ks, np.ndarray) or not isinstance(top_ps, np.ndarray):
        raise ValueError("输入数据必须是NumPy数组")

    # 处理 ml_dtypes.bfloat16 类型的辅助函数
    def convert_ml_bfloat16_to_torch(np_array):
        """将 ml_dtypes.bfloat16 转换为 PyTorch bfloat16"""
        return torch.tensor(np_array.astype(np.float32)).to(torch.bfloat16)

    # 处理 logits_np (bfloat16)
    if hasattr(logits_np.dtype, 'name') and 'bfloat16' in logits_np.dtype.name:
        logits = convert_ml_bfloat16_to_torch(logits_np)
        print(f"Converted logits from {logits_np.dtype} to {logits.dtype}")
    else:
        dtype_logits_tsr = NP_TO_TORCH_DTYPE.get(logits_np.dtype, torch.float32)
        logits = torch.from_numpy(logits_np).type(dtype_logits_tsr)

    # 处理 top_ps (bfloat16)
    if hasattr(top_ps.dtype, 'name') and 'bfloat16' in top_ps.dtype.name:
        topP = convert_ml_bfloat16_to_torch(top_ps)
        print(f"Converted topP from {top_ps.dtype} to {topP.dtype}")
    else:
        dtype_top_ps_tsr = NP_TO_TORCH_DTYPE.get(top_ps.dtype, torch.float32)
        topP = torch.from_numpy(top_ps).type(dtype_top_ps_tsr)

    # 处理其他参数 (保持原样)
    topK = torch.as_tensor(top_ks)

    # 处理 q：若为 None，则设为 None，后续跳过 q 采样
    if q_np is None:
        q = None    
    else:
        dtype_q_tsr = NP_TO_TORCH_DTYPE.get(q_np.dtype, torch.float32)
        q = torch.from_numpy(q_np).type(dtype_q_tsr)  # q_np 是 float32

    batch_size, vocab_size = logits.shape

    print(f'top_k_top_p_sample golden: logits_np.type={logits_np.dtype}, logits.tensor.type={logits.dtype}')
    print(f'top_ps.type={top_ps.dtype}, topP.tensor.type={topP.dtype}')
    print(f'is_need_logits={is_need_logits}')
    print(f'top_k_guess={top_k_guess}')
    print(f'logits_np={logits_np},\n topK={topK}, topP={topP}')

    # 初始化结果张量
    rs_index = torch.zeros(batch_size, dtype=torch.long)

    # 根据是否需要 logits 初始化 rs_value
    if is_need_logits == 1:
        rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
    else:
        rs_value = torch.empty(0)

    for i in range(batch_size):
        original_logits = logits[i].float()
        k_val = topK[i].item()  # 获取标量值
        p = topP[i].item()
        # 判断是否使用 q 采样
        use_q = q is not None and q[i].numel() > 0
        
        # 判断是否使用guessK逻辑：当没有明确topK但有topP时
        temp = min(1024, vocab_size)
        if k_val <= min(1024, vocab_size) and k_val > 0:
            sorted_logits, sorted_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)
            k_val = min(k_val, vocab_size)
            topk_logits = sorted_logits[:k_val]
            topk_indices = sorted_indices[:k_val]

            topk_probs = onlySoftmax(topk_logits, dim=-1)

            if p < ALL_P_MAX and p > 0:
                sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True)
                probs_sum = sorted_probs.cumsum(dim=-1)

                top_p_mask = (probs_sum - sorted_probs) > p
                selected_probs_indices = sorted_probs_indices[~top_p_mask]
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = topk_logits[selected_probs_indices]
            else:
                selected_indices = topk_indices
                selected_logits = topk_logits

            selected_probs = onlySoftmax(selected_logits, dim=-1)

            if use_q:
                q_i = q[i, :len(selected_indices)]
                q_sample = selected_probs / (q_i.abs() + eps)
                probs_index = q_sample.argmax(dim=0).view(-1)
            else:
                probs_index = selected_probs.argmax(dim=0).view(-1)
            
            golden_index = selected_indices[probs_index].squeeze(0)
            rs_index[i] = golden_index

            if is_need_logits == 1:
                rs_value[i, selected_indices] = original_logits[selected_indices]
        
        elif p < ALL_P_MAX and p > 0:
            sorted_logits, sorted_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)

            sorted_probs = onlySoftmax(sorted_logits, dim=-1)
            probs_sum = sorted_probs.cumsum(dim=-1)

            top_p_mask = (probs_sum - sorted_probs) > p
            false_count = torch.sum(~top_p_mask)
            selected_indices = sorted_indices[~top_p_mask]
            selected_logits = sorted_logits[~top_p_mask]
            selected_probs = onlySoftmax(selected_logits, dim=-1)

            if use_q:
                q_i = q[i, :false_count]
                q_sample = selected_probs / (q_i.abs() + eps)
                probs_index = q_sample.argmax(dim=0).view(-1)
            else:
                probs_index = selected_probs.argmax(dim=0).view(-1)

            golden_index = selected_indices[probs_index].squeeze(0)
            rs_index[i] = golden_index

            if is_need_logits == 1:
                rs_value[i, selected_indices] = original_logits[selected_indices]
        
        else:
            selected_probs = onlySoftmax(original_logits, dim=-1)

            if use_q:
                q_i = q[i]
                q_sample = selected_probs / (q_i.abs() + eps)
                probs_index = q_sample.argmax(dim=0).view(-1)
            else:
                probs_index = selected_probs.argmax(dim=0).view(-1)
        
            rs_index[i] = probs_index

            if is_need_logits:
                selected_indices = torch.arange(vocab_size)
                rs_value[i, selected_indices] = original_logits[selected_indices]

    print(f"\nrs_index={rs_index}")
    print(f"\nrs_value={rs_value}")

    return rs_index, rs_value


class TestTopKTopPSample(TestCase):
    def cpu_exec(self, data, run_attr: additionAttr):
        # golden脚本抄这里
        logits_select_idx, logits_top_kp_select = top_k_top_p_sample(data, run_attr)
        # torch tensor to npu
        logits_select_idx_golden_npu = logits_select_idx.npu()
        logits_top_kp_select_golden_npu = logits_top_kp_select.npu()
        return logits_select_idx_golden_npu, logits_top_kp_select_golden_npu


    def npu_exec(self, data_npu, run_attr: additionAttr):
        logits_npu = data_npu[0]
        top_k_npu = data_npu[1]
        top_p_npu = data_npu[2]
        if data_npu[3] is not None:
            q_npu = data_npu[3]
        else:
            q_npu = None

        logits_select_idx, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits_npu, top_k_npu, top_p_npu, q_npu, run_attr.eps, run_attr.is_need_logits, run_attr.top_k_guess)
        return logits_select_idx, logits_top_kp_select


    def _custom_test(self, bs, voc_size, data_info, run_attr: additionAttr):
        logits = np.random.uniform(0, 1, size=(bs, voc_size)).astype(np.float16)
        # generate top_k
        if data_info[0] > 0:
            # enable topk
            top_k = np.random.randint(low=1, high=min(voc_size, 1024), size=(bs,)).astype(np.int32)
        else:
            top_k = np.random.randint(low=1025, high=max(voc_size, 2048), size=(bs,)).astype(np.int32)

        # generate top_p
        if data_info[1] > 0:
            # enable topp
            top_p = np.random.uniform(0, 1, size=(bs, )).astype(np.float16)
        else:
            top_p = np.ones(bs).astype(np.float16)

        # generate q
        if data_info[2] > 0:
            q = np.random.uniform(0, 1, size=(bs, voc_size)).astype(np.float32)
        else:
            q = None

        logits_npu = torch.from_numpy(logits).npu()
        top_k_npu = torch.from_numpy(top_k).npu()
        top_p_npu = torch.from_numpy(top_p).npu()
        if q is not None:
            q_npu = torch.from_numpy(q).npu()
        else:
            q_npu = None

        logits_select_idx_golden_npu, logits_top_kp_select_golden_npu = \
            self.cpu_exec([logits, top_k, top_p, q], run_attr)

        logits_select_idx_npu, logits_top_kp_select_npu = self.npu_exec([logits_npu, top_k_npu, top_p_npu, q_npu], run_attr)

        return logits_select_idx_npu, logits_top_kp_select_npu, logits_select_idx_golden_npu, logits_top_kp_select_golden_npu

    @unittest.skip("skip test_top_k_top_p_sample_major for now")
    @SupportedDevices(['Ascend910B'])
    def test_top_k_top_p_sample_major(self, device="npu"):
        bs_rng_list = [(1, 128)]
        voc_size = 2 ^ 14
        topK_flags = [1]
        topP_flags = [1]
        q_flags = [1]
        need_logits_flags = [0, 1]

        case_no = 0
        for bs_rng in bs_rng_list:
            bs = random.randint(bs_rng[0], bs_rng[1])
            for use_topK, use_topP, use_q, is_need_logits in itertools.product(topK_flags, topP_flags, q_flags, need_logits_flags):

                kpq_set = [use_topK, use_topP, use_q]
                run_attr = additionAttr(is_need_logits, 32, 1e-8)

                logits_select_idx_npu, logits_top_kp_select_npu, logits_select_idx_golden_npu, logits_top_kp_select_golden_npu = \
                self._custom_test(bs, voc_size, kpq_set, run_attr)

                self.assertRtolEqual(logits_select_idx_npu, logits_select_idx_golden_npu)
                if is_need_logits == 1:
                    self.assertRtolEqual(logits_top_kp_select_npu, logits_top_kp_select_golden_npu)
                
                case_no += 1
                print(f"{case_no} cases passed.")

if __name__ == "__main__":
    run_tests()
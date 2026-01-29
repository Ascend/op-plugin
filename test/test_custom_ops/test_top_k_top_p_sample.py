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

USE_FAST_PROBS = 1
FLT_NEG_INF = float('-inf')


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
    eps: float = 1e-8
    is_need_logits: bool = False
    top_k_guess: int = 32
    ks_max: int = 1024
    input_is_logits: bool = True
    post_sample: str = "qSample"


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


def top_k_top_p_sample(data, run_attr: additionAttr, generator):
    # 获取输入数组和参数
    logits_np = data[0]
    top_ks = data[1]
    top_ps = data[2]
    q_np = data[3]
    min_p_np = data[4]

    ALL_P_MAX = 1.0

    # Get runtime Attr
    eps = run_attr.eps
    is_need_logits = run_attr.is_need_logits
    top_k_guess = run_attr.top_k_guess
    ks_max = run_attr.ks_max
    input_is_logits = run_attr.input_is_logits
    post_sample = run_attr.post_sample

    k_max_aligned = (ks_max * 4 + 32 - 1) // 32 * 32 // 4
    k_max = k_max_aligned if k_max_aligned < 1024 else 1024

    # 处理 ml_dtypes.bfloat16 类型的辅助函数
    def convert_ml_bfloat16_to_torch(np_array):
        """将 ml_dtypes.bfloat16 转换为 PyTorch bfloat16"""
        return torch.tensor(np_array.astype(np.float32)).to(torch.bfloat16)

    # 处理 logits_np (bfloat16)
    if hasattr(logits_np.dtype, 'name') and 'bfloat16' in logits_np.dtype.name:
        logits = convert_ml_bfloat16_to_torch(logits_np)
    else:
        dtype_logits_tsr = NP_TO_TORCH_DTYPE.get(logits_np.dtype, torch.float32)
        logits = logits_np

    # 处理 top_ps (bfloat16)
    if hasattr(top_ps.dtype, 'name') and 'bfloat16' in top_ps.dtype.name:
        topP = convert_ml_bfloat16_to_torch(top_ps)
    else:
        dtype_top_ps_tsr = NP_TO_TORCH_DTYPE.get(top_ps.dtype, torch.float32)
        topP = top_ps

    # 处理其他参数 (保持原样)
    topK = torch.as_tensor(top_ks)

    # 处理 q：若为 None，则设为 None，后续跳过 q 采样
    if q_np is None:
        q = None
    else:
        dtype_q_tsr = NP_TO_TORCH_DTYPE.get(q_np.dtype, torch.float32)
        q = torch.from_numpy(q_np).type(dtype_q_tsr)  # q_np 是 float32

    # 处理 min_ps：若为 None，则设为 None，后续跳过 minp 采样
    if min_p_np is None:
        min_ps = None
    elif hasattr(min_p_np.dtype, 'name') and 'bfloat16' in min_p_np.dtype.name:
        min_ps = convert_ml_bfloat16_to_torch(min_p_np)
    else:
        dtype_min_p_np_tsr = NP_TO_TORCH_DTYPE.get(min_p_np.dtype, torch.float32)
        min_ps = min_p_np

    batch_size, vocab_size = logits.shape

    # 初始化结果张量
    rs_index = torch.zeros(batch_size, dtype=torch.long)
    logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.long)
    logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32)


    # 根据是否需要 logits 初始化 rs_value
    if is_need_logits:
        if input_is_logits:    # pred_top_p_mode == "softmax":
            # 输入logits未归一化，先归一化再做采样，则无效概率初始化为全-INF，表示所有位置都未通过筛选
            rs_value = torch.ones((batch_size, vocab_size), dtype=torch.float32) * FLT_NEG_INF
        else:
            # 输入logits已经归一化，就不需要再进行归一化以免梯度消失，则无效概率初始化为全0，表示所有位置都未通过筛选
            rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
    else:
        rs_value = torch.empty(0)

    # compute golden 
    for i in range(batch_size):
        # 保存原始logits，用于后续填充rs_value
        original_logits = logits[i].float()
  
        # TopKSample switch
        k_val = topK[i].item()  # 获取标量值
        top_ks_max = min(k_max, vocab_size)
        use_top_k = (k_val >=1 and k_val<=top_ks_max)

        # topPSample switch
        p = topP[i].item()
        use_top_p = p<ALL_P_MAX
        
        # sort input logits firstly （降序）
        topk_logits, topk_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)

        # topK
        if use_top_k:
            # 对 sorted_logits 取前 k 个
            k_val = min(k_val, vocab_size)  # 确保不超过词汇表大小
            topk_logits = topk_logits[:k_val]
            topk_indices = topk_indices[:k_val]

        if input_is_logits:
            # 先归一化再做采样，计算排序后logits的softmax，仅用于topP
            topk_probs = onlySoftmax(topk_logits, dim=-1)
        else:
            # immediately cumsum on sorted logit probs
            topk_probs = topk_logits

        # 根据 p 和 ALL_P_MAX 的关系决定是否使用 Top-P
        if use_top_p:
            # 判断是否使用guessK逻辑：当没有明确topK但有topP时
            use_top_k_guess = not use_top_k
            # 对概率排序以便计算累积概率
            sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True) #TODO: 为什么还需要进行排序
            if p>0:
                # common top-p sample using cumsum prob filter
                probs_sum = sorted_probs.cumsum(dim=-1)
                top_p_mask = (probs_sum - sorted_probs) >= p
            else:
                # reserve only one token with max prob if p<=0
                top_p_mask = [True] * sorted_probs.numel()
                top_p_mask[0] = False

            top_p_mask = torch.tensor(top_p_mask)
            top_p_sel = ~top_p_mask

            # 获取通过筛选的token在排序后概率中的索引
            selected_probs_indices = sorted_probs_indices[top_p_sel]    # local indices    

            # 映射回原始索引
            if USE_FAST_PROBS:
                # 统一使用截断的top_p采样结果
                selected_indices = topk_indices[selected_probs_indices]  
                selected_logits = sorted_probs[top_p_sel]   # Logits
            else:
                selected_indices = topk_indices[selected_probs_indices]
                selected_logits = topk_logits[selected_probs_indices] # 待定

            # 获取非掩码部分的数量（即保留的token数量）
            false_count = (top_p_sel>0).sum().item()
        else:
            # 不使用 Top-P，继承top-P的输入
            selected_indices = topk_indices  # 所有通过TopK筛选的token
            selected_logits = topk_probs
            false_count = topk_probs.numel()
            top_p_sel = [True] * false_count
            top_p_sel = torch.tensor(top_p_sel)
        
        if p <= 0 and input_is_logits: # kernel侧p小于零时取最大值，并且概率为1
            selected_logits[0] = 1

        # Saliency filtering using min_p，按核函数实现保存中间结果
        if min_ps != None:
            min_p = min_ps[i].item()
        else:
            min_p = -1 # 跳过minp采样

        if not use_top_k and not use_top_p and min_p < 1: # 没做过top_k、top_p时，kernel中未进行排序
            selected_indices = torch.arange(len(original_logits))
            if input_is_logits:
                selected_logits = onlySoftmax(original_logits, dim = -1)
            else:
                selected_logits = original_logits

        if min_p<=0:
            # keep all logits inherit from previous stage
            min_p_sel = [True] * false_count    # ones mask matching inherited input
        elif min_p<1:
            # ensure min_p_thd ALWAYS computed with the max logit of current batch
            min_p_thd = torch.max(selected_logits) * min_p
            sel_prob_mask = selected_logits >= min_p_thd
            min_p_sel = [a and b for a,b in zip(top_p_sel, sel_prob_mask)]
        else:
            # reserve only 1 max token for current batch
            min_p_sel = [False] * false_count
            min_p_sel[0] = True

        min_p_sel = torch.tensor(min_p_sel)
        # if not use_top_k and not use_top_p: # 当不使能topK、topP时，对logits归一化结果直接进行minP采样
        
        selected_logits = selected_logits[min_p_sel]
        selected_indices = selected_indices[min_p_sel]
        false_count = selected_logits.numel()

        # Metric consistency
        if USE_FAST_PROBS:
            # 更快的计算，直接使用采样后的结果，仅确保它们都已归一化，但不确保当前batch的采样结果概率之和为1
            selected_probs = selected_logits
        else:
            if input_is_logits:
                # 计算筛选后logits的softmax (0910 vllm golden)
                selected_probs = onlySoftmax(selected_logits, dim=-1)
            else:
                # 已经归一化，则使用直接值以免梯度消失 (sglang)
                selected_probs = selected_logits

        # Post sample 

        if post_sample == "qSample": # qsample
            q_i = q[i, :false_count]
            q_sample = selected_probs / (q_i.abs() + eps)
            probs_index = q_sample.argmax(dim=0).view(-1)
        elif post_sample == "None": # 直接在选择概率中取最大值
            probs_index = selected_probs.argmax(dim=0).view(-1)
        elif post_sample == "multiNomial":
            logits_sort_masked[i, :len(selected_logits)] = selected_probs
            logits_idx[i, :len(selected_indices)] = selected_indices

        if post_sample != "multiNomial":
            golden_index = selected_indices[probs_index].squeeze(0)
            rs_index[i] = golden_index

        if is_need_logits:
            # 设置所有通过筛选的token位置的值为原始logits值
            rs_value[i, selected_indices] = original_logits[selected_indices]
            #rs_value[rs_value==0] = FLT_NEG_INF

    if post_sample == "multiNomial":
        probs_index = torch.multinomial(logits_sort_masked.npu(), num_samples=1, replacement=True, generator=generator)
        probs_index = probs_index.cpu()
        for j in range(batch_size):
            rs_index[j] = logits_idx[j][probs_index[j]]        
        
    return rs_index, rs_value


class TestTopKTopPSample(TestCase):
    def cpu_exec(self, data, run_attr: additionAttr, generator):
        # golden脚本抄这里
        logits_select_idx, logits_top_kp_select = top_k_top_p_sample(data, run_attr, generator)
        # torch tensor to npu
        logits_select_idx_golden_npu = logits_select_idx.npu()
        logits_top_kp_select_golden_npu = logits_top_kp_select.npu()
        return logits_select_idx_golden_npu, logits_top_kp_select_golden_npu


    def npu_exec(self, data_npu, run_attr: additionAttr, generator_npu):
        logits_npu = data_npu[0]
        top_k_npu = data_npu[1]
        top_p_npu = data_npu[2]
        if data_npu[3] is not None:
            q_npu = data_npu[3]
        else:
            q_npu = None
        if data_npu[4] is not None:
            min_ps_npu = data_npu[4]
        else:
            min_ps_npu = None

        logits_select_idx, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits_npu, top_k_npu, top_p_npu, q=q_npu, min_ps=min_ps_npu, eps=run_attr.eps, is_need_logits=run_attr.is_need_logits, top_k_guess=run_attr.top_k_guess, ks_max=run_attr.ks_max, input_is_logits=run_attr.input_is_logits, post_sample=run_attr.post_sample, generator=generator_npu)
        return logits_select_idx, logits_top_kp_select


    def _custom_test(self, bs, voc_size, data_info, run_attr: additionAttr, dtype):
        torch.manual_seed(1)
        np.random.seed(1)

        input_is_logits = run_attr.input_is_logits
        if dtype != torch.bfloat16:
            if input_is_logits == 1:
                logits = np.random.uniform(0, 10, size=(bs, voc_size)).astype(dtype)
            else:
                logits = np.random.uniform(0, 1, size=(bs, voc_size)).astype(dtype)
            logits = torch.from_numpy(logits)
        else:
            if input_is_logits == 1:
                logits = torch.rand(size=(bs, voc_size), dtype=dtype) * 10
            else:
                logits = torch.rand(size=(bs, voc_size), dtype=dtype)
        # generate top_k
        if data_info[0] > 0:
            # enable topk
            top_k = np.random.randint(low=1, high=min(voc_size, 1024), size=(bs,)).astype(np.int32)
        else:
            top_k = np.random.randint(low=1025, high=max(voc_size, 2048), size=(bs,)).astype(np.int32)

        # generate top_p
        if data_info[1] > 0:
            # enable topp
            if dtype != torch.bfloat16:
                top_p = np.random.uniform(0, 1, size=(bs, )).astype(dtype)
                top_p = torch.from_numpy(top_p)
            else:
                top_p = torch.rand(size=(bs, ), dtype=dtype)
        else:
            if dtype != torch.bfloat16:
                top_p = np.ones(bs).astype(dtype)
                top_p = torch.from_numpy(top_p)
            else:
                top_p = torch.ones(size=(bs, ), dtype=dtype)


        # generate q
        post_sample = run_attr.post_sample
        if post_sample == "qSample":
            q = np.random.uniform(0, 1, size=(bs, voc_size)).astype(np.float32)
        else:
            q = None

        # generate min_ps
        if data_info[2] > 0:
            if dtype != torch.bfloat16:
                min_ps = np.random.uniform(0, 1, size=(bs, )).astype(dtype)
                min_ps = torch.from_numpy(min_ps)
            else:
                min_ps = torch.rand(size=(bs, ), dtype=dtype)
        else:
            min_ps = None
        
        # generator
        if post_sample == "multiNomial":
            generator = torch.Generator(device="npu")
            generator_npu = torch.Generator(device="npu")
            generator.manual_seed(1)
            generator_npu.manual_seed(1)
            
        else:
            generator = None
            generator_npu = None

        logits_npu = logits.npu()
        top_k_npu = torch.from_numpy(top_k).npu()
        top_p_npu = top_p.npu()
        if q is not None:
            q_npu = torch.from_numpy(q).npu()
        else:
            q_npu = None
        if min_ps is not None:
            min_ps_npu = min_ps.npu()
        else:
            min_ps_npu = None
            
        logits_select_idx_golden_npu, logits_top_kp_select_golden_npu = \
            self.cpu_exec([logits, top_k, top_p, q, min_ps], run_attr, generator)
            
            
        logits_select_idx_npu, logits_top_kp_select_npu = self.npu_exec([logits_npu, top_k_npu, top_p_npu, q_npu, min_ps_npu], run_attr, generator_npu)

        return logits_select_idx_npu, logits_top_kp_select_npu, logits_select_idx_golden_npu, logits_top_kp_select_golden_npu


    @unittest.skip("skip test_top_k_top_p_sample_v2_major for now")
    @SupportedDevices(['Ascend910B'])
    def test_top_k_top_p_sample_major(self, device="npu"):
        bs_rng_list = [(1, 128)]
        voc_size = 2 ** 10
        topK_flags = [0, 1] # topK and topP can not be 0 at the same time 
        topP_flags = [0, 1]
        minPs_flags = [0, 1]
        need_logits_flags = [0, 1]
        input_is_logits_flags = [0, 1]
        post_sample_flags = ["qSample", "multiNomial", "None"]
        dtype_list = [np.float16, torch.bfloat16, np.float32]

        case_no = 0
        for bs_rng in bs_rng_list:
            bs = random.randint(bs_rng[0], bs_rng[1])
            for use_topK, use_topP, use_minPs, is_need_logits, input_is_logits, post_sample, dtype in itertools.product(topK_flags, topP_flags, minPs_flags, need_logits_flags, input_is_logits_flags, post_sample_flags, dtype_list):

                kpq_set = [use_topK, use_topP, use_minPs]
                run_attr = additionAttr(1e-8, is_need_logits, 32, 1024, input_is_logits, post_sample)

                logits_select_idx_npu, logits_top_kp_select_npu, logits_select_idx_golden_npu, logits_top_kp_select_golden_npu = \
                self._custom_test(bs, voc_size, kpq_set, run_attr, dtype)

                self.assertRtolEqual(logits_select_idx_npu, logits_select_idx_golden_npu)
                if is_need_logits == 1:
                    self.assertTrue(torch.allclose(logits_top_kp_select_npu, logits_top_kp_select_golden_npu))
                
                case_no += 1
                print(f"{case_no} cases passed.")

if __name__ == "__main__":
    run_tests()
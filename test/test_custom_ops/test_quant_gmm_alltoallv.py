import os
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

class TestQuantGmmAlltoAllv(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist
    
    @classmethod
    def _test_npu_quant_gmm_alltoallv(cls, rank, dtype, c2p, init_pg, input_list1, input_list2, expertTokenNum):
        gmmX, gmmWeight, gmm_x_scale, gmm_weight_scale, mmX, mmWeight, mm_x_scale, mm_weight_scale, gmm_y_dtype, mm_y_dtype = input_list1
        epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance = input_list2
        e = e_epWorldSize // epWorldSize
        pg = init_pg(rank, epWorldSize)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ >= '2.0':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)
        # gmm_x_shape = (mc2_recv_counts[rank].sum(), gmmX.size(1))
        # gmmX = torch.rand(gmm_x_shape).to(dtype)
        goldenOut = TestQuantGmmAlltoAllv._construct_golden_output(rank, dtype, input_list1, input_list2, expertTokenNum)
        send_counts = mc2_recv_counts[rank].reshape(epWorldSize, e).transpose().flatten()
        recv_counts = mc2_send_counts[rank]
        gmmX = gmmX.npu()
        gmmWeight = gmmWeight.npu()
        gmm_x_scale = gmm_x_scale.npu()
        gmm_weight_scale = gmm_weight_scale.npu()
        if mmX is not None:
            mmX = mmX.npu()
        if mmWeight is not None:
            mmWeight = mmWeight.npu()
        if mm_x_scale is not None:
            mm_x_scale = mm_x_scale.npu()
        if mm_weight_scale is not None:
            mm_weight_scale = mm_weight_scale.npu()
        gmmYOut, mmYOut = torch_npu.npu_quant_gmm_alltoallv(
                                                    gmm_x=gmmX,
                                                    gmm_weight=gmmWeight,
                                                    gmm_x_scale = gmm_x_scale,
                                                    gmm_weight_scale = gmm_weight_scale,
                                                    hcom = hcom_name,
                                                    ep_world_size = epWorldSize,
                                                    send_counts=send_counts,
                                                    recv_counts=recv_counts,
                                                    gmm_y_dtype=gmm_y_dtype,
                                                    send_counts_tensor=None,
                                                    recv_counts_tensor=None,
                                                    mm_x=mm_x,
                                                    mm_weight=mm_weight,
                                                    mm_x_scale = mm_x_scale,
                                                    mm_weight_scale = mm_weight_scale,
                                                    comm_quant_scale = None,
                                                    gmm_x_quant_mode = 1,
                                                    gmm_weight_quant_mode = 1,
                                                    mm_x_quant_mode = 1,
                                                    mm_weight_quant_mode = 1,
                                                    comm_quant_mode = 0,
                                                    group_size=0,
                                                    gmm_x_dtype = torch_npu.hifloat8,
                                                    gmm_weight_dtype = torch_npu.hifloat8,
                                                    gmm_x_scale_dtype = torch.float32,
                                                    gmm_weight_scale_dtype = torch.float32,
                                                    mm_x_dtype = torch_npu.hifloat8,
                                                    mm_weight_dtype = torch_npu.hifloat8,
                                                    mm_x_scale_dtype = torch.float32,
                                                    mm_weight_scale_dtype = torch.float32,
                                                    comm_quant_dtype = None,
                                                    mm_y_dtype = mm_y_dtype,)
        if mmYOut is not None:
            mmYOut = mmYOut.cpu()
        gmmYGolden, mmYGolden = goldenOut
        c2p.put((rank, gmmYOut.cpu(), mmYOut))
        if (mmYGolden is not None) or (mmYOut is not None):
            for golden_i, out_i in zip(mmYGolden, mmYOut):
                assert torch.allclose(golden_i, out_i, rtol=0.005, atol=0.005)
        pg.barrier()


    @classmethod
    def _construct_golden_output(cls, rank, dtype, input_list1, input_list2, expertTokenNum):
        gmmX, gmmWeight, gmm_x_scale, gmm_weight_scale, mmX, mmWeight, mm_x_scale, mm_weight_scale, gmm_y_dtype, mm_y_dtype = input_list1
        epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance = input_list2
        e = e_epWorldSize // epWorldSize
        hccl_send_counts = torch.tensor(np.sum(mc2_recv_counts[rank].reshape(-1, e), axis=1).reshape(epWorldSize)).npu().to(torch.int64).to(torch.device('cpu')).numpy()
        hccl_recv_counts = torch.tensor(np.sum(mc2_send_counts[rank].reshape(-1, e), axis=1).reshape(epWorldSize)).npu().to(torch.int64).to(torch.device('cpu')).numpy()
        gmmX = gmmX.npu()
        gmmWeight = gmmWeight.npu()
        num_tokens_per_local_expert = torch.tensor(np.sum(mc2_recv_counts[rank].reshape(-1, e), axis=0).reshape(e)).npu().to(torch.int64)
        alltoAllvGolden = torch_npu.npu_grouped_matmul(
            x=[gmmX],
            weight=[gmmWeight],
            group_list=num_tokens_per_local_expert,
            group_list_type=1,
            split_item=3
        )
        unpermuteGolden = TestQuantGmmAlltoAllv.unpermute_npu(alltoAllvGolden[0], e, epWorldSize, expertTokenNum, rank)
        gmmYGolden = torch.empty((bsk, gmmWeight.size(2)), dtype=dtype).npu()
        dist.all_to_all_single(gmmYGolden, unpermuteGolden, hccl_recv_counts, hccl_send_counts)
        mmGolden = None
        if (mmX is not None) and (mmWeight is not None):
            mmGolden = torch.matmul(mmX.npu(), mmWeight.npu())
        if mmGolden is not None:
            mmGolden = mmGolden.cpu()
        return gmmYGolden[0].cpu(), mmGolden

    @classmethod
    def unpermute_npu(cls, tokens, exp_per_card, epWorldSize, expertTokenNum, rank):
        recv = torch.zeros(epWorldSize, exp_per_card).to(torch.int64)
        for i in range(epWorldSize):
            tmp1 = expertTokenNum[i][rank * exp_per_card : (rank + 1) * exp_per_card]
            recv[i:] = torch.tensor(tmp1)
        tmp1 = recv.t()
        sum_list = torch.cumsum(tmp1, dim=1)
        indices_list = []
        for i in range(exp_per_card):
            tmp = []
            for j in range(epWorldSize):
                if j == 0:
                    tmp.append(list(range(0, sum_list[i][j])))
                else:
                    tmp.append(list(range(sum_list[i][j - 1], sum_list[i][j])))
            indices_list.append(tmp)
        selected = []
        for i in range(epWorldSize):
            for j in range(exp_per_card):
                indices = torch.tensor(indices_list[j][i], dtype=torch.long)
                selected_rows = tokens.index_select(dim=0, index=indices.npu())
                selected.append(selected_rows)
        return torch.cat(selected, dim=0).to(tokens.dtype)
    
    def _test_multiprocess(self, f, init_pg, input_list1, input_list2, dtype, expertTokenNum):
        ctx = mp.get_context("spawn")
        bsk, gmmWeight, mmX, mmWeight, is_trans_gmm_weight, is_trans_mm_weight = input_list1
        epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance = input_list2
        c2p = ctx.Queue(epWorldSize)
        ps = []
        for i in range(epWorldSize):
            p = ctx.Process(
                target=f,
                args=(i, dtype, c2p, init_pg, input_list1, input_list2, expertTokenNum)
            )
            p.start()
            ps.append(p)
        for _ in range(epWorldSize):
            c2p.get()
        for p in ps:
            p.join()
    
    def generate_matrix(self, e, ep_world_size, bsk, balance=True, name="alltoallv_gmm", max_iter=10000):
        if name is not None:
            import hashlib
            hash_bytes = hashlib.sha256(name.encode()).digest()
            seed = int.from_bytes(hash_bytes[:4], 'big')
            np.random.seed(seed)
        row_size = ep_world_size
        col_size = e * ep_world_size
        matrix = []
        if balance:
            avg = bsk // col_size
            tail_num = bsk % col_size
            matrix = np.full((row_size, col_size), avg)
            matrix[:, -1] += tail_num
        else:
            part_col_size = ep_world_size
            part_sum = bsk // e
            tail_sum = bsk % e
            matrix = np.hstack([np.random.multinomial(part_sum - part_col_size, [1 / part_col_size] * part_col_size, size=row_size) + 1 for _ in range(e)])
            matrix[:, -1] += tail_sum
        return matrix
    
    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_npu_alltoallv_gmm(self):
        dtype = torch.float16
        is_balance = True
        e = 2
        BS = 128
        K = 2
        H1 = 256
        N1 = 256
        H2 = 256
        N2 = 128
        gmm_y_dtype = torch.float16
        mm_y_dtype = torch.float16
        bsk = BS * K
        epWorldSize = 2
        e_epWorldSize = e * epWorldSize
        expertTokenNum = torch.tensor(self.generate_matrix(e, epWorldSize, bsk, balance=is_balance))
        mc2_send_counts = self.generate_matrix(e, epWorldSize, bsk, balance=is_balance)
        mc2_recv_counts = np.hstack(np.split(mc2_send_counts.reshape(-1, e), epWorldSize, axis=0))
        gmm_x_shape = (bsk, H1)
        gmmX = torch.rand(gmm_x_shape).to(dtype)
        gmm_weight_shape = (e, H1, N1)
        gmmWeight = torch.rand(gmm_weight_shape).to(dtype)
        gmm_x_scale = torch.tensor([1.0], dtype=torch.float32)
        gmm_weight_scale = torch.tensor([1.0], dtype=torch.float32)
        mm_x_shape = (BS, H2)
        mm_weight_shape = (H2, N2)
        mmX = torch.rand(mm_x_shape)
        mmWeight = torch.rand(mm_weight_shape)
        if (H2 > 0) and (N2 > 0):
            mmX = mmX.to(dtype)
            mmWeight = mmWeight.to(dtype)
        else:
            mmX = None
            mmWeight = None
        mm_x_scale = torch.tensor([1.0], dtype=torch.float32) if mmX is not None else None
        mm_weight_scale = torch.tensor([1.0], dtype=torch.float32) if mmWeight is not None else None
        balance = is_balance
        self._test_multiprocess(
            TestQuantGmmAlltoAllv._test_npu_quant_gmm_alltoallv,
            TestQuantGmmAlltoAllv._init_dist_hccl,
            [gmmX, gmmWeight, gmm_x_scale, gmm_weight_scale, mmX, mmWeight, mm_x_scale, mm_weight_scale, gmm_y_dtype, mm_y_dtype],
            [epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance],
            dtype, expertTokenNum)


class TestQuantGmmAlltoAllvException(TestCase):
    """PTA-layer exception interception tests for npu_quant_gmm_alltoallv.

    These tests validate TORCH_CHECK guards fire before EXEC_NPU_CMD,
    so they only need a single NPU card (no HCCL group required).
    """

    def _make_valid_params(self):
        """Construct a set of valid GMM-only default parameters."""
        e, H, N = 4, 256, 128
        bsk = 256
        ep_world_size = 2
        return dict(
            gmm_x=torch.randn(bsk, H, dtype=torch.float16).npu(),
            gmm_weight=torch.randn(e, H, N, dtype=torch.float16).npu(),
            gmm_x_scale=torch.tensor([1.0], dtype=torch.float32).npu(),
            gmm_weight_scale=torch.tensor([1.0], dtype=torch.float32).npu(),
            hcom="dummy_group",
            ep_world_size=ep_world_size,
            send_counts=[bsk // (e * ep_world_size)] * (e * ep_world_size),
            recv_counts=[bsk // (e * ep_world_size)] * (e * ep_world_size),
            gmm_y_dtype=torch.float16,
        )

    def _make_valid_params_with_mm(self):
        """Construct a set of valid parameters including MM branch."""
        params = self._make_valid_params()
        H2, N2 = 256, 128
        BS = 128
        params.update(
            mm_x=torch.randn(BS, H2, dtype=torch.float16).npu(),
            mm_weight=torch.randn(H2, N2, dtype=torch.float16).npu(),
            mm_x_scale=torch.tensor([1.0], dtype=torch.float32).npu(),
            mm_weight_scale=torch.tensor([1.0], dtype=torch.float32).npu(),
            mm_x_quant_mode=1,
            mm_weight_quant_mode=1,
            mm_y_dtype=torch.float16,
        )
        return params

    # ================================================================
    # A. GMM quant mode validation
    # ================================================================

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_x_quant_mode_invalid(self):
        params = self._make_valid_params()
        params["gmm_x_quant_mode"] = 0
        self.assertRaisesRegex(
            RuntimeError, "gmm_x_quant_mode only support 1",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_x_quant_mode_invalid_2(self):
        params = self._make_valid_params()
        params["gmm_x_quant_mode"] = 2
        self.assertRaisesRegex(
            RuntimeError, "gmm_x_quant_mode only support 1",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_x_quant_mode_default(self):
        """gmm_x_quant_mode=None should default to PERTENSOR, no PTA error."""
        params = self._make_valid_params()
        # Do not pass gmm_x_quant_mode, let it default to None
        try:
            torch_npu.npu_quant_gmm_alltoallv(**params)
        except RuntimeError as e:
            self.assertNotIn("gmm_x_quant_mode only support 1", str(e))

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_weight_quant_mode_invalid(self):
        params = self._make_valid_params()
        params["gmm_weight_quant_mode"] = 0
        self.assertRaisesRegex(
            RuntimeError, "gmm_weight_quant_mode only support 1",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_weight_quant_mode_default(self):
        """gmm_weight_quant_mode=None should default to PERTENSOR, no PTA error."""
        params = self._make_valid_params()
        try:
            torch_npu.npu_quant_gmm_alltoallv(**params)
        except RuntimeError as e:
            self.assertNotIn("gmm_weight_quant_mode only support 1", str(e))

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_comm_quant_mode_invalid(self):
        params = self._make_valid_params()
        params["comm_quant_mode"] = 1
        self.assertRaisesRegex(
            RuntimeError, "comm_quant_mode only support 0",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_comm_quant_mode_default(self):
        """comm_quant_mode=None should default to 0, no PTA error."""
        params = self._make_valid_params()
        try:
            torch_npu.npu_quant_gmm_alltoallv(**params)
        except RuntimeError as e:
            self.assertNotIn("comm_quant_mode only support 0", str(e))

    # ================================================================
    # B. GMM dimension validation
    # ================================================================

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_x_wrong_dim_3d(self):
        params = self._make_valid_params()
        params["gmm_x"] = torch.randn(4, 64, 256, dtype=torch.float16).npu()
        self.assertRaisesRegex(
            RuntimeError, "dimension of gmm_x should be 2D",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_x_wrong_dim_1d(self):
        params = self._make_valid_params()
        params["gmm_x"] = torch.randn(256, dtype=torch.float16).npu()
        self.assertRaisesRegex(
            RuntimeError, "dimension of gmm_x should be 2D",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_weight_wrong_dim_2d(self):
        params = self._make_valid_params()
        params["gmm_weight"] = torch.randn(256, 128, dtype=torch.float16).npu()
        self.assertRaisesRegex(
            RuntimeError, "dimension of gmm_weight should be 3D",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_gmm_weight_wrong_dim_4d(self):
        params = self._make_valid_params()
        params["gmm_weight"] = torch.randn(4, 2, 256, 128, dtype=torch.float16).npu()
        self.assertRaisesRegex(
            RuntimeError, "dimension of gmm_weight should be 3D",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    # ================================================================
    # C. MM quant mode validation (bug fix verification)
    # ================================================================

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_x_quant_mode_invalid(self):
        params = self._make_valid_params_with_mm()
        params["mm_x_quant_mode"] = 0
        self.assertRaisesRegex(
            RuntimeError, "mm_x_quant_mode only support 1",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_x_quant_mode_default(self):
        """mm_x_quant_mode=None should default to PERTENSOR after bug fix."""
        params = self._make_valid_params_with_mm()
        del params["mm_x_quant_mode"]
        try:
            torch_npu.npu_quant_gmm_alltoallv(**params)
        except RuntimeError as e:
            self.assertNotIn("mm_x_quant_mode only support 1", str(e))

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_weight_quant_mode_invalid(self):
        params = self._make_valid_params_with_mm()
        params["mm_weight_quant_mode"] = 0
        self.assertRaisesRegex(
            RuntimeError, "mm_weight_quant_mode only support 1",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_weight_quant_mode_default(self):
        """mm_weight_quant_mode=None should default to PERTENSOR after bug fix."""
        params = self._make_valid_params_with_mm()
        del params["mm_weight_quant_mode"]
        try:
            torch_npu.npu_quant_gmm_alltoallv(**params)
        except RuntimeError as e:
            self.assertNotIn("mm_weight_quant_mode only support 1", str(e))

    # ================================================================
    # D. MM scale consistency validation
    # ================================================================

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_scale_both_missing(self):
        params = self._make_valid_params_with_mm()
        del params["mm_x_scale"]
        del params["mm_weight_scale"]
        self.assertRaisesRegex(
            RuntimeError, "mm_x_scale and mm_weight_scale are required",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_x_scale_missing(self):
        params = self._make_valid_params_with_mm()
        del params["mm_x_scale"]
        self.assertRaisesRegex(
            RuntimeError, "mm_x_scale and mm_weight_scale are required",
            torch_npu.npu_quant_gmm_alltoallv, **params)

    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_mm_weight_scale_missing(self):
        params = self._make_valid_params_with_mm()
        del params["mm_weight_scale"]
        self.assertRaisesRegex(
            RuntimeError, "mm_x_scale and mm_weight_scale are required",
            torch_npu.npu_quant_gmm_alltoallv, **params)


if __name__ == "__main__":
    run_tests()
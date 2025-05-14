# Owner(s): ["module: meta tensors"]

import contextlib
import copy
import itertools
import unittest
import weakref
from unittest.mock import patch
import numpy as np

import torch
import torch._dynamo
import torch._functorch.config
import torch._prims as prims

import torch_npu
import torch_npu.testing
import torch.testing._internal.optests as optests
from torch import distributed as dist
from torch._dynamo.testing import rand_strided
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    FakeTensorConverter,
    DynamicOutputShapeException,
    UnsupportedOperatorException,
)
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests, OpDTypes
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_utils import (
    TestCase, TEST_WITH_TORCHDYNAMO, run_tests, skipIfCrossRef, skipIfRocm, skipIfTorchDynamo, parametrize,
    instantiate_parametrized_tests)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

RUN_NPU = torch.npu.is_available()


def _get_test_torch_version():
    torch_npu_version = torch_npu.__version__
    version_list = torch_npu_version.split('.')
    if len(version_list) > 2:
        return f'v{version_list[0]}.{version_list[1]}'
    else:
        raise RuntimeError("Invalid torch_npu version.")


class TestMoeDistributeDispatch(TestCase):
    def test_moe_distribute_dispatchA2(self):
        with FakeTensorMode():
            quant_mode = 2
            ep_world_size = 16
            tp_world_size = 0
            world_size = ep_world_size
            bs = 8
            h = 7168
            k = 8
            sharedExpertRankNum = 0
            moeExpertNum = 16
            global_bs = bs * ep_world_size
            expert_num_per_rank = 1
            total_expert_num = world_size * expert_num_per_rank

            local_moe_expert_num = moeExpertNum // ep_world_size
            a = global_bs * min(local_moe_expert_num, k)

            ep_recv_cnt_num = ep_world_size * local_moe_expert_num + global_bs * 2 * k * (ep_world_size // 8)

            x = torch.randn(bs, h).to(torch.bfloat16)
            expert_ids = torch.randn(bs, k).to(torch.int32)
            scales = torch.randn(total_expert_num, h).to(torch.float32)
            expert_scales = torch.randn(bs, k).to(torch.float32)
            result = torch_npu.npu_moe_distribute_dispatch(x, expert_ids, "group_ep", ep_world_size, 0, moeExpertNum, scales=scales, x_active_mask=None, expert_scales=expert_scales, group_tp="", tp_world_size=0,
                                                           tp_rank_id=0, expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=sharedExpertRankNum, quant_mode=quant_mode, global_bs=global_bs, expert_token_nums_type=1)
            self.assertEqual(result[0].shape[0], a)
            self.assertEqual(result[0].shape[1], h)
            self.assertEqual(result[0].dtype, torch.int8)

            self.assertEqual(result[1].shape[0], a)
            self.assertEqual(result[1].dtype, torch.float32)

            self.assertEqual(result[2].shape[0], bs * k)
            self.assertEqual(result[2].dtype, torch.int32)

            self.assertEqual(result[3].shape[0], local_moe_expert_num)
            self.assertEqual(result[3].dtype, torch.int64)

            self.assertEqual(result[4].shape[0], ep_recv_cnt_num)
            self.assertEqual(result[4].dtype, torch.int32)

            self.assertEqual(result[5].shape[0], tp_world_size)
            self.assertEqual(result[5].dtype, torch.int32)

            self.assertEqual(result[6].shape[0], a)
            self.assertEqual(result[6].dtype, torch.float32)


if __name__ == "__main__":
    run_tests()

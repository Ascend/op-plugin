# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.
#
# WARNING WARNING: LOOKING TO EDIT THIS FILE?  Think carefully about
# why you need to add a toggle to the global behavior of code
# generation.  The parameters here should really only be used
# for "temporary" situations, where we need to temporarily change
# the codegen in some cases because we cannot conveniently update
# all call sites, and are slated to be eliminated once all call
# sites are eliminated.  If you don't have a plan for how to get there,
# DON'T add a new entry here.


class Locals(threading.local):
    use_const_ref_for_mutable_tensors: Optional[bool] = None
    use_ilistref_for_tensor_lists: Optional[bool] = None


_locals = Locals()


def use_const_ref_for_mutable_tensors() -> bool:
    if _locals.use_const_ref_for_mutable_tensors is None:
        raise ValueError("need to initialize local.use_const_ref_for_mutable_tensors with "
                        "local.parametrize")
    return _locals.use_const_ref_for_mutable_tensors


def use_ilistref_for_tensor_lists() -> bool:
    if _locals.use_ilistref_for_tensor_lists is None:
        raise ValueError("need to initialize local.use_ilistref_for_tensor_lists with "
                        "local.parametrize")
    return _locals.use_ilistref_for_tensor_lists


@contextmanager
def parametrize(
    *, param_use_const_ref_for_mutable_tensors: bool, param_use_ilistref_for_tensor_lists: bool
) -> Iterator[None]:
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_use_ilistref_for_tensor_lists = _locals.use_ilistref_for_tensor_lists
    try:
        _locals.use_const_ref_for_mutable_tensors = param_use_const_ref_for_mutable_tensors
        _locals.use_ilistref_for_tensor_lists = param_use_ilistref_for_tensor_lists
        yield
    finally:
        _locals.use_const_ref_for_mutable_tensors = (
            old_use_const_ref_for_mutable_tensors
        )
        _locals.use_ilistref_for_tensor_lists = old_use_ilistref_for_tensor_lists

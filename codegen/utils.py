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

import re
import os
from typing import Tuple, List, Dict, Iterable, Iterator, Generic, Callable, Sequence, TypeVar, NoReturn, Optional
from enum import Enum
import contextlib
import textwrap
import sys
if sys.version_info >= (3,8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper  # type: ignore[misc]
YamlDumper = Dumper


# for getting mypy to do exhaustiveness checking
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))


T = TypeVar("T")
S = TypeVar("S")


# Map over function that returns sequences and cat them all together
def concatMap(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        yield from func(x)


# A custom loader for YAML that errors on duplicate keys.
class YamlLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            if key in mapping:
                raise ValueError(f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}")
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping

# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
#
# This is an OPEN enum (we may add more cases to it in the future), so be sure
# to explicitly specify with Union[Literal[Target.XXX]] what targets are valid
# for your use.
Target = Enum('Target', (
    # top level namespace (not including at)
    'DEFINITION',
    'DECLARATION',
    # TORCH_LIBRARY(...) { ... }
    'REGISTRATION',
    # namespace { ... }
    'ANONYMOUS_DEFINITION',
    'ANONYMOUS_DEFINITION_UNSUPPORT',
    # namespace cpu { ... }
    'NAMESPACED_DEFINITION',
    'NAMESPACED_DECLARATION',
))

# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r'(^|\W){}($|\W)'


def split_name_params(schema: str) -> Tuple[str, List[str]]:
    m = re.match(r'(\w+)(\.\w+)?\((.*)\)', schema)
    if m is None:
        raise RuntimeError(f'Unsupported function schema: {schema}')
    name, _, params = m.groups()
    return name, params.split(', ')


T = TypeVar('T')
S = TypeVar('S')


# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them

# Map over function that may return None; omit Nones from output sequence
def map_maybe(func: Callable[[T], Optional[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r


# Map over function that returns sequences and cat them all together
def concat_map(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        for r in func(x):
            yield r


# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg_fn: Callable[[], str]) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        msg = msg_fn()
        msg = textwrap.indent(msg, '  ')
        msg = f'{e.args[0]}\n{msg}' if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise


class NamespaceHelper:
    """A helper for constructing the namespace open and close strings for a nested set of namespaces.

    e.g. for namespace_str torch::lazy,

    prologue:
    namespace torch {
    namespace lazy {

    epilogue:
    } // namespace lazy
    } // namespace torch
    """

    def __init__(self, namespace_str: str, entity_name: str = "", max_level: int = 2):
        # cpp_namespace can be a colon joined string such as torch::lazy
        cpp_namespaces = namespace_str.split("::")
        if len(cpp_namespaces) > max_level:
            raise ValueError(f"Codegen doesn't support more than {max_level} level(s)"
                             "of custom namespace. Got {namespace_str}.")
        self.cpp_namespace_ = namespace_str
        self.prologue_ = "\n".join([f"namespace {n} {{" for n in cpp_namespaces])
        self.epilogue_ = "\n".join(
            [f"}} // namespace {n}" for n in reversed(cpp_namespaces)]
        )
        self.namespaces_ = cpp_namespaces
        self.entity_name_ = entity_name

    @staticmethod
    def from_namespaced_entity(
        namespaced_entity: str, max_level: int = 2
    ) -> "NamespaceHelper":
        """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
        names = namespaced_entity.split("::")
        entity_name = names[-1]
        namespace_str = "::".join(names[:-1])
        return NamespaceHelper(
            namespace_str=namespace_str, entity_name=entity_name, max_level=max_level
        )

    @property
    def prologue(self) -> str:
        return self.prologue_

    @property
    def epilogue(self) -> str:
        return self.epilogue_

    @property
    def entity_name(self) -> str:
        return self.entity_name_

    # Only allow certain level of namespaces
    def get_cpp_namespace(self, default: str = "") -> str:
        """
        Return the namespace string from joining all the namespaces by "::" (hence no leading "::").
        Return default if namespace string is empty.
        """
        return self.cpp_namespace_ if self.cpp_namespace_ else default


class OrderedSet(Generic[T]):
    storage: Dict[T, Literal[None]]

    def __init__(self, iterable: Optional[Iterable[T]] = None):
        if iterable is None:
            self.storage = {}
        else:
            self.storage = {k: None for k in iterable}

    def __contains__(self, item: T) -> bool:
        return item in self.storage

    def __iter__(self) -> Iterator[T]:
        return iter(self.storage.keys())

    def update(self, items: "OrderedSet[T]") -> None:
        self.storage.update(items.storage)

    def add(self, item: T) -> None:
        self.storage[item] = None

    def copy(self) -> "OrderedSet[T]":
        ret: OrderedSet[T] = OrderedSet()
        ret.storage = self.storage.copy()
        return ret

    @staticmethod
    def union(*args: "OrderedSet[T]") -> "OrderedSet[T]":
        ret = args[0].copy()
        for s in args[1:]:
            ret.update(s)
        return ret

    def __or__(self, other: "OrderedSet[T]") -> "OrderedSet[T]":
        return OrderedSet.union(self, other)

    def __ior__(self, other: "OrderedSet[T]") -> "OrderedSet[T]":
        self.update(other)
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self.storage == other.storage
        else:
            return set(self.storage.keys()) == other


class PathManager:

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the process owner.
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current process.")

    @classmethod
    def check_directory_path_readable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def remove_path_safety(cls, path: str):
        if os.path.islink(path):
            raise RuntimeError(f"Invalid path is a soft chain: {path}")
        if os.path.exists(path):
            os.remove(path)


def get_version(ver, all_version):
    if ver is None:
        return []

    if isinstance(ver, list):
        start_ver = all_version.index(ver[0])
        end_ver = None if ver[1] == 'newest' else all_version.index(ver[1])
        real_ver = all_version[start_ver:end_ver + 1] if end_ver is not None else all_version[start_ver:]
    elif ver == 'all_version':
        real_ver = all_version
    else:
        real_ver = ver.split(', ') if ', ' in ver else ver.split(',')

    return real_ver

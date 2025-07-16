import importlib
import inspect
import json
import os
import re
import unittest
from typing import Callable
from itertools import chain
from pathlib import Path
import pkgutil

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._utils_internal import get_file_path_2
import torch_npu


def set_failure_list(api_str, value, signature, failure_list):
    failure_list.append(f"# {api_str}:")
    failure_list.append(f"  - function signature is different: ")
    failure_list.append(f"    - the base signature is {value}.")
    failure_list.append(f"    - now it is {signature}.")


def is_not_compatibility(base_str, new_str):
    base_io_params = base_str.split("->")
    new_io_params = new_str.split("->")
    base_input_params = base_io_params[0].strip()
    new_input_params = new_io_params[0].strip()
    base_out_params = "" if len(base_io_params) == 1 else base_io_params[1].strip()
    new_out_params = "" if len(new_io_params) == 1 else new_io_params[1].strip()

    # output params
    if base_out_params != new_out_params:
        return True

    base_params = base_input_params[1:-1].split(",")
    new_params = new_input_params[1:-1].split(",")
    base_diff_params = set(base_params) - set(new_params)

    # case: delete/different default value/different parameter name/different parameter dtype
    if base_diff_params:
        return True

    new_diff_params = set(new_params) - set(base_params)
    for elem in new_diff_params:
        # case: add params
        if "=" not in elem:
            return True

    # case: position parameters
    base_arr = [elem for elem in base_params if "=" not in elem]
    new_arr = [elem for elem in new_params if "=" not in elem]
    i = 0
    while i < len(base_arr):
        if base_arr[i] != new_arr[i]:
            return True
        i += 1

    return False


def get_func_from_yaml(yaml_path):
    content = []
    with open(yaml_path, 'r') as f:
        for line in f.readlines():
            if " func:" in line:
                string = line.split("func:")[1].strip()
                content.append(string)
    return content


def parse_func_str(func_str):
    if "(" in func_str:
        func_name = func_str.split("(")[0]
        signature = func_str.split(func_name)[1]
    else:
        func_name = func_str
        signature = ""
    return func_name, signature


def func_from_yaml(content, base_schema, failure_list):
    torch_npu_path = torch_npu.__path__[0]
    yaml1_path = os.path.join(torch_npu_path, "csrc/aten/npu_native_functions_by_codegen.yaml")
    yaml2_path = os.path.join(torch_npu_path, "csrc/aten/npu_native_functions.yaml")
    yaml1_content = get_func_from_yaml(yaml1_path)
    yaml2_content = get_func_from_yaml(yaml2_path)

    op_funcs = set(yaml1_content) - set(yaml2_content)

    for func in op_funcs:
        func_name, signature = parse_func_str(func)
        if "func: " + func_name in base_schema:
            value = base_schema["func: " + func_name]["signature"]
            if is_not_compatibility(value, signature):
                set_failure_list("func: " + func_name, value, signature, failure_list)
        content["func: " + func_name] = {"signature": signature}


def _get_test_torch_version():
    torch_npu_version = torch_npu.__version__
    version_list = torch_npu_version.split('.')
    if len(version_list) > 2:
        return f'v{version_list[0]}.{version_list[1]}'
    else:
        raise RuntimeError("Invalid torch_npu version.")


def api_signature(obj, api_str, content, base_schema, failure_list):
    signature = inspect.signature(obj)
    signature = str(signature)
    if api_str in base_schema.keys():
        value = base_schema[api_str]["signature"]
        if is_not_compatibility(value, signature):
            set_failure_list(api_str, value, signature, failure_list)
    content[api_str] = {"signature": signature}


def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages.
    """
    for dir_path, _d, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name,) + rel_pt.parts)
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
            (str(pkg_dir_path),), prefix=f'{pkg_pref}.',
        )
        )


def _find_all_importables(pkg):
    """Find all importables in the project.

    Return them in order.
    """
    return sorted(
        set(
            chain.from_iterable(
                _discover_path_importables(Path(p), pkg.__name__)
                for p in pkg.__path__
            ),
        ),
    )


class TestOpApiCompatibility(TestCase):

    @staticmethod
    def _is_mod_public(modname):
        split_strs = modname.split('.')
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    @staticmethod
    def _deleted_apis(base_funcs, now_funcs, failure_list):
        deleted_apis = set(base_funcs) - set(now_funcs)
        for func in deleted_apis:
            failure_list.append(f"# {func}:")
            failure_list.append(f"  - {func} has been deleted.")

    @staticmethod
    def _newly_apis(base_funcs, now_funcs, failure_list, content):
        newly_apis = set(now_funcs) - set(base_funcs)
        for func in newly_apis:
            failure_list.append(f"# {func}:")
            failure_list.append(f"  - {func} is new. Please add it to the torch_npu_OpApi_schema_all.json")
            signature = content[func]["signature"]
            failure_list.append(f"  - it's signature is {signature}.")

    @unittest.skipIf(_get_test_torch_version() in ["v1.11", "v2.0", "v2.2"],
                     "Skipping test for these torch versions.")
    def test_op_api_compatibility(self):
        failure_list = []

        with open(get_file_path_2(os.path.dirname(os.path.dirname(__file__)),
                                  'allowlist_for_publicAPI.json')) as json_file:
            allow_dict = json.load(json_file)

        # load torch_npu_OpApi_schema_all.json
        base_schema = {}
        version_tag = _get_test_torch_version()
        with open(get_file_path_2(os.path.dirname(__file__), "torch_npu_OpApi_schema_all.json")) as fp:
            base_schema0 = json.load(fp)
            for key, value in base_schema0.items():
                if key.startswith("op_api:"):
                    if not ("all_version" in value['version'] or version_tag in value['version']):
                        if 'newest' not in value['version']:
                            continue
                        else:
                            idx = value['version'].index("newest") - 1
                            if version_tag < value['version'][idx]:
                                continue
                    key = key.replace("op_api: ", "").strip()
                    func_name, signature = parse_func_str(key)
                    if func_name not in base_schema:
                        base_schema[func_name] = dict()
                        base_schema[func_name]["signature"] = signature

        content = {}

        def test_module(modname):
            try:
                if "__main__" in modname or \
                        modname in ["torch_npu.dynamo.torchair.core._backend",
                                    "torch_npu.dynamo.torchair.core._torchair"]:
                    return
                mod = importlib.import_module(modname)
            except Exception:
                # It is ok to ignore here as we have a test above that ensures
                # this should never happen
                return

            if not self._is_mod_public(modname):
                return

            def check_one_element(elem, modname, mod, *, is_public):
                obj = getattr(mod, elem)
                if not (isinstance(obj, (Callable, torch.dtype)) or inspect.isclass(obj)):
                    return

                elem_module = getattr(obj, '__module__', None)

                if not elem_module == "torch._ops.npu":
                    return

                # check if the api is public
                if (modname in allow_dict and elem in allow_dict[modname]) or elem in torch_npu.__all__:
                    api_str = f"{modname}.{elem}"
                    api_signature(obj, api_str, content, base_schema, failure_list)

            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, modname, mod, is_public=elem in public_api)
            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith('_'):
                        check_one_element(elem, modname, mod, is_public=True)

        for modname in _find_all_importables(torch_npu):
            test_module(modname)

        test_module('torch_npu')

        base_funcs = base_schema.keys()
        now_funcs = content.keys()
        self._deleted_apis(base_funcs, now_funcs, failure_list)
        self._newly_apis(base_funcs, now_funcs, failure_list, content)

        msg = "All the APIs below do not meet the compatibility guidelines. "
        msg += "If the change timeline has been reached, you can modify the torch_npu_OpApi_schema_all.json to make it OK."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))
        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)

    @unittest.skipIf(_get_test_torch_version() in ["v1.11", "v2.0", "v2.2"],
                     "Skipping test for these torch versions.")
    def test_op_func_compatibility(self):
        failure_list = []
        all_base_schema = {}
        # load torch_npu_OpApi_schema_all.json
        with open(get_file_path_2(os.path.dirname(__file__), "torch_npu_OpApi_schema_all.json")) as fp:
            all_base_schema0 = json.load(fp)
            for key, value in all_base_schema0.items():
                if not key.startswith("op_api:"):
                    all_base_schema[key] = value

        version_tag = _get_test_torch_version()
        base_schema = {}
        for key, value in all_base_schema.items():
            if not("all_version" in value['version'] or version_tag in value['version']):
                if 'newest' not in value['version']:
                    continue
                else:
                    idx = value['version'].index("newest") - 1
                    if version_tag < value['version'][idx]:
                        continue
            func_name, signature = parse_func_str(key)
            if func_name not in base_schema:
                base_schema[func_name] = dict()
                base_schema[func_name]["signature"] = signature

        content = {}

        # functions from npu_native_functions_by_codegen.yaml
        func_from_yaml(content, base_schema, failure_list)

        base_funcs = base_schema.keys()
        now_funcs = content.keys()
        self._deleted_apis(base_funcs, now_funcs, failure_list)
        self._newly_apis(base_funcs, now_funcs, failure_list, content)

        msg = "All the OpAPIs below do not meet the compatibility guidelines. "
        msg += "If the change timeline has been reached, you can modify the torch_npu_OpApi_schema_all.json to make it OK."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))
        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    run_tests()

# Copyright (c) 2024 Huawei Technologies Co., Ltd
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

import os
import argparse
import yaml

from codegen.gen import FileManager, parse_native_yaml_struct
from codegen.struct.struct_codegen import parse_struct_yaml, gen_op_api
from codegen.utils import PathManager


def main() -> None:

    parser = argparse.ArgumentParser(description='Generate struct aclnn files')
    parser.add_argument(
        '-n',
        '--native_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '--struct_yaml',
        help='path to struct yaml file containing aclnn operators struct definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    options = parser.parse_args()

    fm = FileManager(
            install_dir=options.output_dir, template_dir="codegen/struct/templates", dry_run=False
        )

    native_yaml_path = os.path.realpath(options.native_yaml)
    PathManager.check_directory_path_readable(native_yaml_path)
    with open(native_yaml_path, "r") as f:
        es = yaml.safe_load(f)

    native_functions = parse_native_yaml_struct(es)

    struct_info = parse_struct_yaml(options.struct_yaml, native_functions)
    gen_op_api(fm, struct_info)

if __name__ == '__main__':
    main()

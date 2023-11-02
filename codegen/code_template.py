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
import stat
from typing import Match, Optional, Sequence, Mapping

# match $identifier or ${identifier} and replace with value in env
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting to that depth and putting each element
# of the list on its own line
# if the identifier is on a line starting with non-whitespace and a list
# then it is comma separated ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.


class CodeTemplate:
    # Python 2.7.5 has a bug where the leading (^[^\n\S]*)? does not work,
    # workaround via appending another [^\n\S]? inside

    substitution_str = r'(^[^\n\S]*[^\n\S]?)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})'

    # older versions of Python have a bug where \w* does not work,
    # so we need to replace with the non-shortened version [a-zA-Z0-9_]*

    substitution_str = substitution_str.replace(r'\w', r'[a-zA-Z0-9_]')

    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def check_dir_path(file_path) -> 'CodeTemplate':
        """
        check file path readable.
        """
        file_path = os.path.realpath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The path does not exist: {file_path}")
        if os.stat(file_path).st_uid != os.getuid():
            check_msg = input("The path does not belong to you, do you want to continue? [y/n]")
            if check_msg.lower() != 'y':
                raise RuntimeError("The user choose not to contiue")
        if os.path.islink(file_path):
            raise RuntimeError(f"Invalid path is a soft chain: {file_path}")

    @staticmethod
    def from_file(filename: str) -> 'CodeTemplate':
        CodeTemplate.check_dir_path(filename)
        with open(filename, 'r') as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        self.pattern = pattern
        self.filename = filename

    def substitute(self, env: Optional[Mapping[str, object]] = None, **kwargs: object) -> str:
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            if env is None:
                raise ValueError("env is not set.")
            return kwargs.get(v) if v in kwargs else env.get(v)

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            return "".join([indent + line + "\n" for e in v for line in str(e).splitlines()]).rstrip()

        def replace(match: Match[str]) -> str:
            indent = match.group(1)
            key = match.group(2)
            comma_before = ''
            comma_after = ''
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ', '
                    key = key[1:]
                if key[-1] == ',':
                    comma_after = ', '
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ', '.join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)
        return self.substitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate("""\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """)
    print(c.substitute(args=["hi", 8], bar=["what", 7],
                       a=3, b=4, stuff=["things...", "others"], empty=[]))

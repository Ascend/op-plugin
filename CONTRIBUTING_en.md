# OpPlugin Contribution Guide

Thank you for considering contributing to OpPlugin. We welcome contributions of all kinds, including bug fixes, feature enhancements, and documentation improvements. Whether you are an experienced developer or contributing to an open-source project for the first time, your help is valuable.

## Project Overview

OpPlugin is the operator plugin for Ascend Extension for PyTorch (`torch_npu`). It enables developers using the PyTorch framework to conveniently access the NPU operator library. This project provides the operator adaptation files required to execute `torch_npu`.

### Project Architecture

```sh
op-plugin
├── ci/                            # CI build scripts
│── docs/                          # Project documentation
├── op_plugin/                     # Core operator plugin implementation
│   ├── ops/                       # Operator implementations
│   │   ├── aclops/                # ACL operator kernel implementations
│   │   ├── atb/                   # ATB operator implementations
│   │   └── opapi/                 # OpAPI operator wrappers
│   ├── python/                    # Python APIs
│   ├── config/                    # Operator configuration
│   ├── utils/                     # Utility modules
│   └── third_party/               # Third-party dependencies
├── examples/                      # Code samples
│   ├── aclnn_extension/           # ACLNN extension examples
│   ├── cpp_extension/             # C++ extension examples
│   └── framwork_cpp_extension/    # Framework C++ extension examples
├── torchnpugen/                   # Code generation tool
│   ├── struct/                    # Data structure generation
│   ├── api/                       # API generation
│   └── templates/                 # Code templates
└── test/                          # Test cases
```

### Core Modules

| Module| Description|
|------|------|
| `op_plugin/ops/aclops` | ACL operator kernel implementations based on `AscendCL`|
| `op_plugin/ops/atb` | ATB operator implementations for the Transformer Acceleration Library|
| `op_plugin/ops/opapi` | OpAPI wrappers that adapt PyTorch operator APIs|
| `op_plugin/config` | Operator configuration where `derivatives.yaml` defines operator differentiation rules|
| `torchnpugen` | Code generation tool that automatically generates operator adaptation code|
| `test/test_base_ops` | Operator test suite for functional validation of operators|

## Ways to Contribute

We look forward to your contributions. Every contribution helps make OpPlugin better:

- **Report issues**: Submit bugs or feature requests to identify and resolve code system issues.
- **Contribute code**: Submit bug fixes or new features to participate directly in core project development.
- **Improve documentation**: Enhance existing documentation or add missing content to maximize usability.
- **Review code**: Evaluate open pull requests to improve overall codebase quality.
- **Spread the word**: Share the project on technical blogs or social media platforms and ⭐ the repository.

## Contribution Scenarios

We welcome contributions of all kinds and look forward to your participation.

### 1. Feature Requests and Suggestions

If you have ideas for new features or performance improvements, we encourage you to submit an issue and discuss them with the community.

**Issue type**: Feature request

**Please include the following information**:

- **Background**: Describe the problem the feature addresses and the value it brings to users.
- **Feature description**: Provide a detailed description of the proposed feature.
- **Design proposal**: Describe the technical approach, key module design, and relationships with upstream and downstream components.
- **Expected benefits**: Describe the expected functionality, performance improvements, and accuracy targets.

### 2. Bug Reports and Fixes

If you discover a bug or documentation issue, we welcome your feedback and proposed fixes.

**Bug Report Template**

- **Environment information**: `OpPlugin` version, `torch_npu` version, `Python` version, `CANN` version, and any other relevant environment information.
- **Issue description**: Describe the issue clearly. Add appropriate labels if applicable.
- **Steps to reproduce**: Provide detailed steps to reproduce the issue whenever possible.
- **Expected behavior**: Describe the expected behavior.
- **Additional notes for reviewers**: Include any special considerations or additional information, if applicable.

**Fix Process**

1. Locate the corresponding bug report in the issue tracker.
2. Comment `/assign` to claim the issue.
3. Create a development branch and implement the fix.
4. Submit a pull request.

### 3. Community Support

If you know how to solve issues raised by other contributors, we encourage you to share your solutions in the corresponding issues.

## Contribution Process

### Contributor License Agreement

Before submitting code to the OpPlugin community for the first time, you must sign the Contributor License Agreement (CLA).

For individual contributors, see the [ICLA documentation](https://clasign.osinfra.cn/sign/6971f3727a5429cd9010ed41).

### Development and Testing

1. **Fork the repository**: Click **Fork** in the upper-right corner of the repository page on GitCode to create a copy under your account.

2. **Clone the repository**:

    ```bash
    git clone https://gitcode.com/<your-username>/op-plugin.git
    cd op-plugin
    ```

3. **Create a development branch**:

    ```bash
    git checkout -b {new_branch_name} origin/master
    ```

4. **Develop your code**: Follow the **[Code Specifications](#code-specifications)**.

5. **Test your code**: Run tests to ensure that the code functions as expected.

6. **Run CI checks**: Ensure your changes pass compilation, static analysis, and unit tests (UTs).

7. **Submit a pull request**: Submit a pull request and wait for code review.

8. **Community review**: Changes involving patches, header macros, public APIs, and similar components require community review before merging.

### Code Review Requirements

The following types of changes require community review:

- **Patch replacements**: Replace native PyTorch APIs with patches.
- **Header macro updates**: Add or modify macro definitions.
- **Public API changes**: Add, modify, or remove public APIs.
- **Core component changes**: Modify core modules such as memory management and device management.

**Requirements for New Computational APIs**

When introducing a new computational API, contributors must provide accuracy test results demonstrating that the API meets the required accuracy standards. The test report must include:

- Comparisons with the PyTorch CPU and GPU backends.
- Error statistics, including maximum error, mean error, and root mean square error (RMSE).
- Validation in representative scenarios, such as neural network models.

### Acceptance Criteria

#### API Function Requirements

- **Full input parameter coverage**: Test all parameter types, including required parameters, optional parameters, and default-value scenarios.
- **Test methods**: Use techniques such as equivalence partitioning and boundary value analysis to cover both valid and invalid input scenarios.
- **Exception handling**: Verify that error messages are accurate, informative, and user-friendly.

#### API Accuracy Requirements

- **Number of test cases**: Generate 100 to 200 test cases for each supported `dtype`.
- **Operator coverage**: Generate two test cases each for forward and backward operators.
- **Accuracy threshold**: Ensure that computation errors remain within acceptable limits and that results are consistent with the PyTorch CPU and GPU backends.

### Development Deliverables

Operator adaptation development must include the following deliverables.

| Deliverable| Description|
|--------|------|
| **YAML configuration**| Declare the operator version, schema, and adaptation method in `op_plugin_functions.yaml`.|
| **Forward and backward binding configuration**| Configure forward and backward operator bindings in `derivatives.yaml` (required only for operators that require forward/backward bindings).|
| **Operator adaptation code**| Implement the operator in `op_plugin/ops/opapi/` or `op_plugin/ops/aclops/`.|
| **UT cases**| Add functional and accuracy tests under `test/test_base_ops/`.|
| **API documentation**| Update the public operator documentation, either through automatic generation or manual additions.|

## Code Specifications

To make OpPlugin Hub easy to develop, review, and maintain, notice the following specifications:

### Coding Guidelines

- **Python**: Follow the [PEP 8 style guide](https://pep8.org/).
- **C++**: Follow the [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html).

Use tools such as [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), and [pylint](https://pylint.org/) to check code formatting and quality.

### UT Guidelines

- **Python**: Use [pytest](http://pytest.org/en/latest/).
- **C++**: Use [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md).

Ensure that test case names clearly reflect the design intent of each test.

### Refactoring Guidelines

We encourage contributors to refactor code to eliminate code smells. All code must comply with the project coding and testing standards.

## Static Code Checks (`pre-commit`)

This project uses [pre-commit](https://pre-commit.com/) to automatically perform static analysis and style fixes before code is committed, ensuring consistent code quality.

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Register Git hooks in the repository root directory
pre-commit install
```

### Usage

```bash
# Commit staged changes to trigger automated pre-commit runs
git commit -m "your message"

# Check only modified files currently in the working area
pre-commit run

# Manually run a full check on all files in the repository
pre-commit run --all-files

# Manually run a specific hook by specifying a hook ID from .pre-commit-config.yaml
pre-commit run <hook-id>

# Automatically fix code style by running only hooks that support auto-fix
pre-commit run ruff-check
pre-commit run ruff-format
pre-commit run clang-format
```

> Certain hooks (such as `ruff-format` and `clang-format`) modify files directly. After they complete, execute `git add` again before committing.

### Integrated Static Analysis Tools

| Tool| Version| Application Scope| Description|
|------|------|----------|----------|
| [trailing-whitespace](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files| Removes trailing whitespace|
| [end-of-file-fixer](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files| Ensures files end with a newline character|
| [check-yaml](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | YAML files| Validates `YAML` syntax|
| [check-added-large-files](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files| Prevents oversized files from being committed|
| [check-merge-conflict](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files| Detects unresolved merge conflict markers|
| [detect-private-key](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files| Detects private keys and other sensitive information leakage|
| [check-json](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | JSON files| Validates `JSON` syntax|
| [ruff-check](https://gitcode.com/gh_mirrors/ru/ruff-pre-commit) | v0.14.14 | Python files| Performs Python linting and automatically fixes supported issues|
| [ruff-format](https://gitcode.com/gh_mirrors/ru/ruff-pre-commit) | v0.14.14 | Python files| Formats Python code as a replacement for `Black`|
| [codespell](https://gitcode.com/gh_mirrors/co/codespell) | v2.4.1 | Non-code files| Detects spelling mistakes, excluding source files such as `.py`, `.cpp`, and `.h`|
| [pylint](https://gitcode.com/gh_mirrors/pyl/pylint) | v4.0.5 | Python files| Performs in-depth Python code quality analysis|
| [bandit](https://gitcode.com/gh_mirrors/ba/bandit) | v1.9.4 | Python files| Performs static security analysis for Python vulnerability scanning|
| [typos](https://gitcode.com/gh_mirrors/ty/typos) | v1.32.0 | All files| Performs fast spell checking implemented in `Rust`|
| [clang-format](https://gitcode.com/pre-commit-clang/mirrors-clang-format) | v18.1.8 | C/C++ files| Formats C/C++ code according to the `.clang-format` configuration|

## Practical Guide

### Environment Setup and Build

**Build**:

```bash
# Install dependencies and execute compilation
bash ci/build.sh

# Build manually using CMake as an alternative approach
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Pull Request Requirements

**PR merge checklist** (for details, see the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] Code compilation succeeds.
- [ ] Static analysis checks (such as `CppLint` and `CppCheck`) pass.
- [ ] UT cases pass.
- [ ] Code style matches standards such as `PEP 8` and `Google C++ Style Guide`.
- [ ] Commit messages follow the `Conventional Commits` specification.
- [ ] The PR title correctly utilizes type tags such as `feat`, `fix`, `refactor`, `docs`, and `test`.
- [ ] Code comments are complete and error logs are recorded correctly.
- [ ] Code implementation executes validations for return values and null pointers.
- [ ] Newly added computational APIs provide accuracy test results.

### Functional Verification

**Test case locations**:

- `test/test_base_ops/`: basic operator functional tests
- `test/core_tests/`: core functionality tests

**Running Tests**

```bash
# Install the test dependencies
pip3 install -r test/requirements.txt

# Run a single test file
python test/test_base_ops/test_add.py

# Run a test by using run_test.py
python test/run_test.py -i test_add
```

### Dealing with CI failures

Common Continuous Integration (CI) check failures include the following categories. Rectify them as prompted.

- **Build failure**: Inspect the build failure logs and rebuild the target after resolving the issue.
- **Static analysis failure**: Follow the prompts to identify and resolve issues in the code, such as style violations and potential bugs.
- **UT failure**: Follow the prompts to identify the failing test cases and investigate the cause.

## Submitting a Pull Request

1. **Push your code to the remote repository**.

    ```bash
    git add .
    git status
    git commit -m "Your commit title"
    git commit -s --amend  # Add detailed description
    git push origin {new_branch_name}
    ```

2. **Create a pull request**.

Create a pull request on GitCode and complete the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md) details:

- Source Branch
- Modification Plan
- Documentation Changes
- API Changes
- Functional Verification
- CheckList

Verify that all required information is complete and accurate before submitting the pull request, then wait for code review.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and inclusive environment for all community participants:

- **Respect differences**: Value diverse perspectives and experiences while embracing cultural diversity.
- **Be open-minded**: Accept constructive criticism to drive continuous learning and progress.
- **Focus on contributions**: Prioritize actions that best benefit the community and advance the project.
- **Show empathy**: Express empathy toward other community members and support each other.

### Communication Channels

We provide multiple communication channels to help you participate in the community:

- **[Issues](https://gitcode.com/Ascend/op-plugin/issues)**: Report bugs or submit new feature recommendations.
- **[Pull Requests](https://gitcode.com/Ascend/op-plugin/pulls)**: Participate in code reviews and engineering design discussions.

### Questions

We welcome every developer to participate actively in community discussions and look forward to growing together with you:

- **Unresolved issues**: Comment on open issues to present your solutions.
- **Long-standing issues**: Perform a pre-check before resolving them to avoid duplicate work.
- **Self-reported issues**: Share your solution to help the community learn and benefit from it.

If you have any questions, feel free to join the discussion. We look forward to your contributions.

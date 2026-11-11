# OpPlugin Contribution Guide

Thank you for considering contributing to OpPlugin! We welcome contributions of all forms, including bug fixes, feature enhancements, documentation improvements, and so on. Whether you are an experienced developer or a first-time open source participant, your help is invaluable.

## Project Introduction

OpPlugin is the operator plugin of TorchNPU. It provides developers who use the PyTorch framework with convenient access to the NPU operator library. This project aims to provide the operator adaptation files that TorchNPU requires for running.

### Project Architecture

```sh
op-plugin
├── ci/                            # CI build scripts
│── docs/                          # Project documents
├── op_plugin/                     # Core operator plugin implementation
│   ├── ops/                       # Operator implementations
│   │   ├── aclops/                # ACL operator kernel implementations
│   │   ├── atb/                   # ATB operator implementations
│   │   └── opapi/                 # OpAPI operator wrappers
│   ├── python/                    # Python interfaces
│   ├── config/                    # Operator configurations
│   ├── utils/                     # Utility classes
│   └── third_party/               # Third-party dependencies
├── examples/                      # Sample code
│   ├── aclnn_extension/           # ACLNN extension samples
│   ├── cpp_extension/             # C++ extension samples
│   └── framwork_cpp_extension/    # Framework C++ extension samples
├── torchnpugen/                   # Code generation tool
│   ├── struct/                    # Data structure generation
│   ├── api/                       # API generation
│   └── templates/                 # Code templates
└── test/                          # Test cases
```

### Core Modules

| Module | Description |
|------|------|
| `op_plugin/ops/aclops` | ACL operator kernel implementations: underlying operator implementations based on AscendCL |
| `op_plugin/ops/atb` | ATB operator implementations: wrappers of operators in the Transformer acceleration library |
| `op_plugin/ops/opapi` | OpAPI operator wrappers: adaptation layer of PyTorch operator interfaces |
| `op_plugin/config` | Operator configurations: derivatives.yaml defines operator differentiation rules |
| `torchnpugen` | Code generation tool: automatically generates operator adaptation code |
| `test/test_base_ops` | Operator tests: function verification covering each operator |

## How to Contribute

We look forward to your participation! Every contribution drives the progress of OpPlugin:

- **Report issues**: Report bugs or submit feature suggestions to help us identify and resolve problems
- **Contribute code**: Submit code fixes or new features to participate in project development directly
- **Improve documentation**: Improve documents or supplement missing content to make the project more readable
- **Review code**: Review pull requests to help improve code quality
- **Spread the word**: Share the project on blogs or social media, and give the repository a star ⭐

## Contribution Scenarios

This project welcomes contributions of all forms. We look forward to your participation!

### 1. Requirements and Feature Suggestions

If you have suggestions for new features or ideas for performance optimization, submit an issue to discuss them with the community.

**Issue type**: requirement or feature suggestion

**Content to include**:

- **Feature background**: the problem that the feature solves and the value that it brings to users
- **Feature description**: describe the suggested feature in detail
- **Design scheme**: the technical approach, design of key modules, and relationships between upstream and downstream components
- **Expected benefits**: feature objectives, performance metrics, and accuracy

### 2. Bug Reports and Fixes

If you find a bug or a document problem, we sincerely welcome your feedback and fix suggestions.

**Bug report format**:

- **Environment information**: OpPlugin version, TorchNPU version, Python version, CANN version, and so on
- **Problem description**: add labels so that the issue is highlighted on the issue dashboard
- **Reproduction steps**: describe how to reproduce the problem in as much detail as possible
- **Expected behavior**: describe the behavior that you expect
- **Special notes for reviewers**: any special circumstances, if applicable

**Fix process**:

1. Find the corresponding bug description in the issue
2. Comment `/assign` to claim the task
3. Create a branch for the fix
4. Submit a pull request

### 3. Community Support

If you can resolve issues raised by others, share your solutions in the issues.

## Contribution Process

### Contributor License Agreement

Sign the CLA before you submit code to the OpPlugin community for the first time.

For individual contributors, refer to the [ICLA online document](https://www.mindspore.cn/icla) for details.

### Development and Testing

1. **Fork the repository**: On the GitCode platform, click **Fork** in the upper right corner of the repository to clone it to your personal account

2. **Clone the repository to your local machine**:

    ```bash
    git clone https://gitcode.com/<your-username>/op-plugin.git
    cd op-plugin
    ```

3. **Create a development branch**:

    ```bash
    git checkout -b {new_branch_name} origin/master
    ```

4. **Develop code**: Follow the **[coding standards](#coding-standards)**

5. **Test the code**: Run tests to ensure that the code functions properly

6. **CI gate checks**: Run CI checks to ensure that the code passes compilation, static checks, and UT tests

7. **Submit a pull request**: Submit the PR and wait for code review

8. **Community review**: If the change involves patches, header file macros, API interfaces, and so on, submit it for community review

### Code Merge Review Requirements

The following types of changes require community review:

- **Patch replacement**: patch replacement of native PyTorch interfaces
- **Header file macro updates**: adding or modifying macro definitions
- **API interface changes**: adding, modifying, or deleting public APIs
- **Core component changes**: modifications to core modules such as memory management and device management

**Constraints on new compute-related API interfaces**:

For new compute-related API interfaces, submitters must provide additional accuracy test results to ensure that the computation accuracy of the new interfaces meets requirements. Accuracy tests must include the following items:

- Comparison results with the PyTorch CPU/GPU backends
- Error range statistics (maximum error, average error, root mean square error, and so on)
- Typical scenario verification (such as actual performance in neural network models)

### Acceptance Criteria

#### API Function Criteria

- **Full input parameter coverage**: all input parameter types must be covered, including required parameters, optional parameters, and default value scenarios
- **Test methods**: use test methods such as equivalence class partitioning and boundary value analysis to cover normal and abnormal parameter passing scenarios
- **Exception handling**: verify that error messages in exception scenarios are accurate and user-friendly

#### API Accuracy Criteria

- **Number of test cases**: generalize 100 to 200 test cases for each dtype
- **Operator coverage**: generate test cases for forward operators and backward operators according to 2 APIs respectively
- **Accuracy threshold**: the computation result error must be within the allowed range to ensure consistency with CPU/GPU results

### Development Deliverables

Operator adaptation development must deliver the following items:

| Deliverable | Description |
|--------|------|
| **YAML configuration** | Declare the operator version, schema, and adaptation method in `op_plugin_functions.yaml` |
| **Forward-backward binding configuration** | Configure the forward-backward binding relationship of operators in `derivatives.yaml` (only for operators that require forward-backward binding) |
| **Operator adaptation code** | Implement operator interfaces in the `op_plugin/ops/opapi/` or `op_plugin/ops/aclops/` directory |
| **Unit test cases** | Write function tests and accuracy tests in the `test/test_base_ops/` directory |
| **Interface document** | Update the external operator interface document (automatically generated or manually supplemented) |

## Coding Standards

Follow these styles to make OpPlugin easy to develop, review, and maintain.

### Coding Guidelines

- **Python**: We recommend the [PEP 8 coding style](https://pep8.org/)
- **C++**: We recommend the [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html)

You can use the following tools to check code format: [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), and [pylint](https://pylint.org/)

### Unit Test Guidelines

- **Python**: We recommend [pytest](http://pytest.org/en/latest/)
- **C++**: We recommend the [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)

The design intent of a test case should be reflected in its descriptive name.

### Refactoring Guidelines

We encourage developers to refactor code to eliminate code smells. All code must comply with the coding style and test style requirements.

## Code Static Checks (pre-commit)

This project uses [pre-commit](https://pre-commit.com/) to automatically run static checks and style fixes before code submission, ensuring consistent code quality.

### Installation

```bash
# Install the pre-commit tool
pip install pre-commit

# Register Git hooks in the repository root directory
pre-commit install
```

### Usage

```bash
# Check all staged files (triggered automatically during commit)
git commit -m "your message"

# Check only the files modified in the current workspace
pre-commit run

# Manually run a full check on all files
pre-commit run --all-files

# Manually run a specified check tool. The hook IDs in the following commands can be found in .pre-commit-config.yaml.
pre-commit run <hook-id>

# Automatically fix code style (run only hooks with the auto-fix capability)
pre-commit run ruff-check
pre-commit run ruff-format
pre-commit run clang-format
```

> Some hooks (such as `ruff-format` and `clang-format`) directly modify files. After fixes, run `git add` again and then commit.

### Integrated Static Check Tools

| Tool | Version | Scope | Description |
|------|------|----------|----------|
| [trailing-whitespace](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files | Deletes extra whitespace at the end of lines |
| [end-of-file-fixer](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files | Ensures that files end with a newline |
| [check-yaml](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | YAML files | Validates YAML syntax |
| [check-added-large-files](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files | Blocks submission of oversized files |
| [check-merge-conflict](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files | Detects unresolved merge conflict markers |
| [detect-private-key](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | All files | Detects leakage of sensitive information such as private keys |
| [check-json](https://gitcode.com/pre-commit/pre-commit-hooks) | v4.6.0 | JSON files | Validates JSON syntax |
| [ruff-check](https://gitcode.com/gh_mirrors/ru/ruff-pre-commit) | v0.14.14 | Python files | Runs Python lint checks and automatically fixes fixable problems |
| [ruff-format](https://gitcode.com/gh_mirrors/ru/ruff-pre-commit) | v0.14.14 | Python files | Formats Python code (replaces black) |
| [codespell](https://gitcode.com/gh_mirrors/co/codespell) | v2.4.1 | Non-code files | Checks spelling errors (skips source code files such as .py/.cpp/.h) |
| [pylint](https://gitcode.com/gh_mirrors/pyl/pylint) | v4.0.5 | Python files | Performs in-depth Python code quality checks |
| [bandit](https://gitcode.com/gh_mirrors/ba/bandit) | v1.9.4 | Python files | Performs static scanning for Python security vulnerabilities |
| [typos](https://gitcode.com/gh_mirrors/ty/typos) | v1.32.0 | All files | Detects spelling errors quickly (implemented in Rust) |
| [clang-format](https://gitcode.com/pre-commit-clang/mirrors-clang-format) | v18.1.8 | C/C++ files | Formats C/C++ code by reading the `.clang-format` configuration |

## Practical Guide

### Environment Setup and Compilation

**Build**:

```bash
# Install dependencies and compile
bash ci/build.sh

# Alternatively, use CMake to compile manually
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### PR Merge Requirements

**Merge checklist** (For detailed requirements, refer to the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] The code compiles successfully
- [ ] Static checks pass (CppLint, CppCheck, and so on)
- [ ] UT test cases pass
- [ ] The code style complies with standards (PEP 8, Google C++ Style)
- [ ] Commit messages comply with standards (Conventional Commits)
- [ ] The PR title correctly uses type labels (feat, fix, refactor, docs, test, and so on)
- [ ] Code comments are complete and error logs are correctly recorded
- [ ] The code implementation performs checks such as return value and null pointer checks
- [ ] Accuracy test results are provided for new compute-related APIs

### Function Verification Guide

**Test case locations**:

- `test/test_base_ops/` - Basic operator function tests
- `test/core_tests/` - Core function tests

**Run tests**:

```bash
# Install test dependencies
pip3 install -r test/requirements.txt

# Run a single test file
python test/test_base_ops/test_add.py

# Alternatively, use run_test.py
python test/run_test.py -i test_add
```

### CI Gate Exception Handling

CI gate exceptions mainly include the following types. Resolve them based on the related prompts:

- **Compilation exception**: Check the cause of the compilation failure, resolve the problem, and compile again
- **Static check exception**: Locate and resolve problems in the code based on the prompts (such as code style and potential bugs)
- **UT test failure**: Locate the failed test cases based on the prompts and check the causes

## Submitting a Pull Request

1. **Push code to the remote repository**:

    ```bash
    git add .
    git status
    git commit -m "Your commit title"
    git commit -s --amend  # Add a detailed description
    git push origin {new_branch_name}
    ```

2. **Create a pull request**

Create a pull request on GitCode and complete the form based on the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md):

- Merge source
- Modification scheme
- Document changes
- Interface changes
- Function verification
- CheckList

After confirming that the information is complete and accurate, submit the pull request and wait for code review.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, secure, and inclusive environment for all participants:

- **Respect differences**: respect different views and experiences, and embrace diverse cultures
- **Open mind**: accept constructive criticism, and keep learning and improving
- **Focus on contributions**: focus on what benefits the community most and drive project development
- **Empathy**: show empathy to other community members and help each other

### Communication Channels

We provide multiple communication channels for community interaction:

- **[Issues](https://gitcode.com/Ascend/op-plugin/issues)**: for reporting bugs and submitting feature suggestions
- **[Pull Requests](https://gitcode.com/Ascend/op-plugin/pulls)**: for code review and discussion

### Issue Consultation

We warmly welcome every developer to participate in community discussions. We look forward to growing together with you:

- **Found an unresolved issue**: comment in the issue to share your solution
- **Found a long-unhandled issue**: perform a pre-check before resolving it to avoid duplicated work
- **Resolved your own reported issue**: share your solution so that the community can learn and improve together

If you have any questions, join the community discussions at any time. We look forward to your wonderful contributions!

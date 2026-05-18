import argparse
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple
"""
Generate ATB public C++ wrappers.

Input:
  - op_plugin/config/atb_ops.yaml
  - op_plugin/ops/atb/*.cpp

Output:
  - op_plugin/include/AtbOpsInterface.h
  - op_plugin/ops/atb/AtbOpsInterface.cpp

YAML decides which wrappers to expose. Signatures come from the
corresponding ::atb implementation definitions.
"""


PUBLIC_NAMESPACE = "at_npu::native::atb"


@dataclass(frozen=True)
class AtbOpEntry:
    cpp_name: str
    cpp_short_name: str
    impl_name: str
    impl_signature: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATB C++ public interfaces")
    parser.add_argument(
        "--config",
        default="op_plugin/config/atb_ops.yaml",
        help="Path to op_plugin/config/atb_ops.yaml.",
    )
    parser.add_argument("--header", default="op_plugin/include/AtbOpsInterface.h", help="Output header path.")
    parser.add_argument("--source", default="op_plugin/ops/atb/AtbOpsInterface.cpp", help="Output source path.")
    parser.add_argument("--atb-src-dir", default="op_plugin/ops/atb", help="Directory containing ATB op cpp files.")
    return parser.parse_args()


def normalize_cpp_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_matching_paren(text: str, open_paren: int) -> int:
    depth = 1
    idx = open_paren + 1
    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        idx += 1
    return idx - 1 if depth == 0 else -1


def split_top_level(text: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    paren_depth = 0
    angle_depth = 0
    square_depth = 0
    brace_depth = 0
    for char in text:
        if char == delimiter and all(depth == 0 for depth in (paren_depth, angle_depth, square_depth, brace_depth)):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def split_signature(signature: str) -> Tuple[str, str, List[str]]:
    open_paren = signature.find("(")
    close_paren = signature.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
        raise RuntimeError(f"malformed implementation signature: {signature}")
    prefix = normalize_cpp_whitespace(signature[:open_paren])
    match = re.match(r"(?P<return_type>.+?)\s+(?P<function_name>[A-Za-z_][A-Za-z0-9_]*)$", prefix)
    if match is None:
        raise RuntimeError(f"malformed implementation signature: {signature}")
    parameter_text = signature[open_paren + 1:close_paren].strip()
    parameters = split_top_level(parameter_text) if parameter_text else []
    return match.group("return_type"), match.group("function_name"), parameters


def canonicalize_cpp_signature(signature: str) -> str:
    return_type, function_name, parameters = split_signature(signature)
    prefix = normalize_cpp_whitespace(f"{return_type} {function_name}")
    prefix = re.sub(r"([&>*])([A-Za-z_][A-Za-z0-9_]*)$", r"\1 \2", prefix)
    if not parameters:
        return f"{prefix}()"
    normalized_parameters = []
    for item in parameters:
        item = normalize_cpp_whitespace(item)
        item = re.sub(r"\(\s+", "(", item)
        item = re.sub(r"\s+\)", ")", item)
        item = re.sub(r"\s+,", ",", item)
        item = re.sub(r",\s*", ", ", item)
        item = re.sub(r"\s*&\s*(?=[,>)])", "&", item)
        item = re.sub(r"&\s+([A-Za-z_])", r"&\1", item)
        normalized_parameters.append(item)
    return f"{prefix}({', '.join(normalized_parameters)})"


def parse_parameter_declaration(parameter_text: str) -> Tuple[str, str, str]:
    parts = split_top_level(parameter_text, delimiter="=")
    declaration_text = parts[0].strip()
    default_text = "=".join(parts[1:]).strip() if len(parts) > 1 else ""
    if not declaration_text:
        raise RuntimeError(f"malformed parameter declaration: {parameter_text}")

    identifier_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for match in reversed(list(identifier_pattern.finditer(declaration_text))):
        prefix = declaration_text[:match.start()]
        suffix = declaration_text[match.end():].strip()
        if suffix and re.fullmatch(r"(?:\[[^\]]*\]\s*)+", suffix) is None:
            continue
        if not prefix or (not prefix[-1].isspace() and prefix[-1] not in "&*"):
            continue
        type_text = prefix.strip()
        if not type_text:
            continue
        return normalize_cpp_whitespace(type_text), match.group(0), default_text

    raise RuntimeError(f"parameter declaration has no name: {parameter_text}")


def load_config(config_path: Path) -> List[AtbOpEntry]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    functions = data.get("functions") if isinstance(data, dict) else None
    if not isinstance(functions, list):
        raise RuntimeError(f"config error: `functions` must exist and be a list in {config_path}")
    if not functions:
        raise RuntimeError(f"config error: no ATB public API entries found in {config_path}")

    entries: List[AtbOpEntry] = []
    for index, item in enumerate(functions, 1):
        if not isinstance(item, dict):
            raise RuntimeError(f"config error: entry #{index} must be a mapping in {config_path}")
        cpp_name = item.get("cpp_name")
        impl_name = item.get("impl_name")
        if not cpp_name or not impl_name:
            raise RuntimeError(f"config error: entry #{index} must contain both cpp_name and impl_name in {config_path}")
        prefix = f"{PUBLIC_NAMESPACE}::"
        if not cpp_name.startswith(prefix):
            raise RuntimeError(f"config error: cpp_name must start with {prefix} {cpp_name}")
        cpp_short_name = cpp_name[len(prefix):].strip()
        if not cpp_short_name or "::" in cpp_short_name:
            raise RuntimeError(f"config error: cpp_name must end with a function name under {PUBLIC_NAMESPACE}: {cpp_name}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cpp_short_name):
            raise RuntimeError(f"config error: cpp_name has invalid function name: {cpp_name}")
        entries.append(AtbOpEntry(cpp_name=cpp_name, cpp_short_name=cpp_short_name, impl_name=impl_name, impl_signature=""))
    return entries


def extract_impl_signature_index(cpp_texts: Dict[Path, str], impl_names: List[str]) -> Dict[str, str]:
    index: Dict[str, List[str]] = {}
    impl_short_names = sorted({impl_name.removeprefix("atb::") for impl_name in impl_names})
    patterns = {
        impl_short_name: re.compile(rf"^\s*.*\b{re.escape(impl_short_name)}\s*\(", re.M)
        for impl_short_name in impl_short_names
    }

    for text in cpp_texts.values():
        for impl_short_name, pattern in patterns.items():
            for match in pattern.finditer(text):
                open_paren = text.find("(", match.start())
                if open_paren == -1:
                    continue
                end_paren = find_matching_paren(text, open_paren)
                if end_paren == -1:
                    continue
                tail_idx = end_paren + 1
                while tail_idx < len(text) and text[tail_idx].isspace():
                    tail_idx += 1
                if tail_idx >= len(text) or text[tail_idx] != "{":
                    continue
                signature = normalize_cpp_whitespace(text[match.start():end_paren + 1])
                index.setdefault(f"atb::{impl_short_name}", []).append(signature)

    signatures: Dict[str, str] = {}
    for impl_name in impl_names:
        matches = index.get(impl_name, [])
        unique = sorted(set(matches))
        if not unique:
            raise RuntimeError(f"implementation error: no implementation found for impl_name {impl_name}")
        if len(unique) != 1:
            raise RuntimeError(f"implementation error: multiple implementations found for impl_name {impl_name}: {unique}")
        signatures[impl_name] = canonicalize_cpp_signature(unique[0])
    return signatures


def resolve_entries(entries: List[AtbOpEntry], impl_signatures: Dict[str, str]) -> List[AtbOpEntry]:
    resolved_entries: List[AtbOpEntry] = []
    seen_cpp_names = set()
    for entry in entries:
        if entry.cpp_name in seen_cpp_names:
            raise RuntimeError(f"config error: duplicate cpp_name in config: {entry.cpp_name}")
        seen_cpp_names.add(entry.cpp_name)
        impl_signature = impl_signatures.get(entry.impl_name)
        for parameter in split_signature(impl_signature)[2]:
            try:
                parse_parameter_declaration(parameter)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"implementation parameter must be named for public wrapper generation: "
                    f"impl_name={entry.impl_name}, signature='{impl_signature}', parameter='{parameter}'"
                ) from exc
        resolved_entries.append(
            AtbOpEntry(
                cpp_name=entry.cpp_name,
                cpp_short_name=entry.cpp_short_name,
                impl_name=entry.impl_name,
                impl_signature=impl_signature,
            )
        )
    return resolved_entries


def render_outputs(resolved_entries: List[AtbOpEntry]) -> Tuple[str, str]:
    template_dir = Path(__file__).resolve().parent / "templates"
    header_template = Template((template_dir / "AtbOpsInterface.h").read_text(encoding="utf-8"))
    source_template = Template((template_dir / "AtbOpsInterface.cpp").read_text(encoding="utf-8"))
    declarations: List[str] = []
    forward_declarations: List[str] = []
    definitions: List[str] = []

    for entry in resolved_entries:
        _, impl_short_name, _ = split_signature(entry.impl_signature)
        declaration_signature, count = re.subn(
            rf"\b{re.escape(impl_short_name)}\b", entry.cpp_short_name, entry.impl_signature, count=1
        )
        if count != 1:
            raise RuntimeError(
                f"failed to rename function {impl_short_name} to {entry.cpp_short_name} in signature: {entry.impl_signature}"
            )
        call_args = []
        for parameter in split_signature(declaration_signature)[2]:
            _, name, _ = parse_parameter_declaration(parameter)
            call_args.append(name)
        return_type, _, _ = split_signature(entry.impl_signature)
        call_statement = f"::{entry.impl_name}({', '.join(call_args)});"
        if return_type != "void":
            call_statement = f"return {call_statement}"
        declarations.append(f"TORCH_NPU_API {declaration_signature};")
        forward_declarations.append(f"{entry.impl_signature};")
        definitions.append(f"{declaration_signature}\n{{\n    {call_statement}\n}}")

    header_content = header_template.substitute(
        generated_comment="@generated by torchnpugen/gen_atb_ops.py",
        declarations="\n".join(declarations).strip(),
    )
    source_content = source_template.substitute(
        generated_comment="@generated by torchnpugen/gen_atb_ops.py",
        forward_declarations="\n".join(forward_declarations).strip(),
        definitions="\n\n".join(definitions).strip(),
    )
    return header_content, source_content


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    header_path = Path(args.header).resolve()
    source_path = Path(args.source).resolve()
    atb_src_dir = Path(args.atb_src_dir).resolve()
    # Load the public wrapper exposure list from yaml.
    entries = load_config(config_path)
    # Collect ATB implementation sources, excluding the generated wrapper file itself.
    cpp_texts = {
        path: path.read_text(encoding="utf-8")
        for path in sorted(atb_src_dir.glob("*.cpp"))
        if path.name != source_path.name
    }
    # Resolve each yaml entry to a unique ::atb implementation signature.
    impl_signatures = extract_impl_signature_index(cpp_texts, [entry.impl_name for entry in entries])
    resolved_entries = resolve_entries(entries, impl_signatures)
    # Render wrapper declarations/definitions from the resolved signatures.
    header_content, source_content = render_outputs(resolved_entries)
    # Only rewrite outputs when content actually changes.
    if header_content != (header_path.read_text(encoding="utf-8") if header_path.exists() else None):
        header_path.write_text(header_content, encoding="utf-8")
    if source_content != (source_path.read_text(encoding="utf-8") if source_path.exists() else None):
        source_path.write_text(source_content, encoding="utf-8")


if __name__ == "__main__":
    main()

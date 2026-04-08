"""
Layer modifier for NeuralArch-Bench.

Parses natural-language instructions and applies them to PyTorch model
source code (as a string). Operates on the predictable code style used by
arch_library.py:
    __init__:  self.<name> = nn.<LayerExpr>(...)
    forward:   x = self.<name>(x)

Public API:
    apply_layer_modification(code, instruction) -> (str, bool)
    Returns (modified_code, True) on success, (original_code, False) on failure.
"""

import re


def apply_layer_modification(code: str, instruction: str) -> tuple[str, bool]:
    """
    Apply a natural-language layer modification to model source code.

    Supported operations:
      - "add <LayerExpr> after <target_name>"
        e.g. "add BatchNorm1d(256) after fc1"
             "add Dropout(0.3) after conv1"
             "add ReLU() after fc2"
      - "remove <target_name>"
        e.g. "remove drop1"
      - "replace <target_name> with <NewLayerExpr>"
        e.g. "replace fc1 with Linear(784, 512)"
    """
    norm = instruction.strip()

    # Try each operation in order
    for op in (_try_add, _try_remove, _try_replace):
        result, success = op(code, norm)
        if success:
            return result, True

    return code, False


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------

def _try_add(code: str, instruction: str) -> tuple[str, bool]:
    """add <LayerExpr> after <target_name>"""
    m = re.match(
        r"add\s+([\w\d_.]+(?:\([^)]*\))?)\s+after\s+(\w+)",
        instruction,
        re.IGNORECASE,
    )
    if not m:
        return code, False

    layer_expr = m.group(1)
    target_name = m.group(2)

    # Build the new layer's attribute name
    new_name = _auto_name(target_name, layer_expr)

    # --- inject into __init__ ---
    # Match:  self.<target_name> = nn.<anything>(...)
    init_pattern = re.compile(
        r"^( *)(self\." + re.escape(target_name) + r"\s*=.+)$",
        re.MULTILINE,
    )
    m_init = init_pattern.search(code)
    if not m_init:
        return code, False

    indent = m_init.group(1)
    new_init_line = f"{indent}self.{new_name} = nn.{layer_expr}"
    code = (
        code[: m_init.end()]
        + "\n"
        + new_init_line
        + code[m_init.end() :]
    )

    # --- inject into forward ---
    fwd_pattern = re.compile(
        r"^( *)(x\s*=\s*self\." + re.escape(target_name) + r"\s*\([^)]*\).*)$",
        re.MULTILINE,
    )
    m_fwd = fwd_pattern.search(code)
    if not m_fwd:
        # Forward line not found; roll back __init__ change by returning failure
        # (Already modified code but the model would be broken, better to undo)
        return _undo_add_init(code, new_name, new_init_line), False

    fwd_indent = m_fwd.group(1)
    new_fwd_line = f"{fwd_indent}x = self.{new_name}(x)"
    code = (
        code[: m_fwd.end()]
        + "\n"
        + new_fwd_line
        + code[m_fwd.end() :]
    )

    return code, True


def _try_remove(code: str, instruction: str) -> tuple[str, bool]:
    """remove <target_name>"""
    m = re.match(r"remove\s+(\w+)", instruction, re.IGNORECASE)
    if not m:
        return code, False

    target_name = m.group(1)

    # Remove self.<target_name> = ... from __init__
    init_pattern = re.compile(
        r"^ *self\." + re.escape(target_name) + r"\s*=.+\n?",
        re.MULTILINE,
    )
    code, n1 = init_pattern.subn("", code)

    # Remove self.<target_name>(...) usage from forward
    fwd_pattern = re.compile(
        r"^ *\w+\s*=\s*self\." + re.escape(target_name) + r"\s*\([^)]*\).*\n?",
        re.MULTILINE,
    )
    code, n2 = fwd_pattern.subn("", code)

    if n1 == 0 and n2 == 0:
        return code, False

    return code, True


def _try_replace(code: str, instruction: str) -> tuple[str, bool]:
    """replace <target_name> with <NewLayerExpr>"""
    m = re.match(
        r"replace\s+(\w+)\s+with\s+([\w\d_.]+(?:\([^)]*\))?)",
        instruction,
        re.IGNORECASE,
    )
    if not m:
        return code, False

    target_name = m.group(1)
    new_expr = m.group(2)

    init_pattern = re.compile(
        r"^( *self\." + re.escape(target_name) + r"\s*=\s*)nn\.\S+",
        re.MULTILINE,
    )
    m_init = init_pattern.search(code)
    if not m_init:
        return code, False

    code = code[: m_init.start()] + m_init.group(1) + f"nn.{new_expr}" + code[m_init.end() :]
    return code, True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_name(target_name: str, layer_expr: str) -> str:
    """Generate a unique attribute name for the injected layer."""
    lower = layer_expr.lower()
    if "batchnorm" in lower or "bn" in lower:
        suffix = "bn"
    elif "dropout" in lower or "drop" in lower:
        suffix = "drop"
    elif "relu" in lower:
        suffix = "relu"
    elif "linear" in lower:
        suffix = "fc"
    elif "conv" in lower:
        suffix = "conv"
    else:
        # Use first word of layer expr as suffix
        suffix = re.split(r"\W", lower)[0]
    return f"{target_name}_{suffix}"


def _undo_add_init(code: str, new_name: str, new_init_line: str) -> str:
    """Remove a previously inserted __init__ line (cleanup on forward-injection failure)."""
    return code.replace("\n" + new_init_line, "")

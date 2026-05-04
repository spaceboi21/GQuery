"""
SandboxExecutor – runs LLM-generated Python analysis in an isolated subprocess.

Security model
--------------
1. AST whitelist: only imports from ALLOWED_MODULES are permitted.
2. Blocked builtins: exec, eval, compile, __import__, open (direct), input.
3. Path injection: DATA_PATH and OUTPUT_DIR are injected as top-level constants
   so the code can read/write without needing raw path strings.
4. Subprocess timeout: hard kill after settings.sandbox_timeout_sec.
5. The executed script must print its result as JSON to stdout; non-JSON
   stdout is returned as a plain string under the 'stdout' key.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent

from .config import Settings
from .schema import AnalysisResult

ALLOWED_MODULES = {
    "pandas", "pd",
    "polars", "pl",
    "numpy", "np",
    "plotly",
    "json",
    "datetime",
    "pathlib",
    "re",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "enum",
    "string",
    "calendar",
}

BLOCKED_NAMES = {"exec", "eval", "compile", "__import__", "input", "open"}

_HEADER_TEMPLATE = dedent("""\
    import pandas as pd
    import polars as pl
    import numpy as np
    import json
    import math
    import re
    from pathlib import Path
    from datetime import datetime, date, timedelta
    from collections import defaultdict, Counter

    DATA_PATH = Path({data_path!r})
    OUTPUT_DIR = Path({output_dir!r})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _xia_charts = []
    _xia_chart_jsons = []

    def save_chart(fig, name: str) -> str:
        # Capture JSON for inline rendering — skip file writes to avoid kaleido hangs
        try:
            _xia_chart_jsons.append(fig.to_json())
        except Exception:
            pass
        return name

""")

_FOOTER_TEMPLATE = dedent("""\

    # XIA auto-footer: emit result
    _xia_result = locals().get("result", {})
    if not isinstance(_xia_result, dict):
        _xia_result = {"value": str(_xia_result)}
    _xia_result["_chart_paths"] = _xia_charts
    _xia_result["_chart_jsons"] = _xia_chart_jsons
    print(json.dumps(_xia_result, default=str))
""")


class SandboxExecutor:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings

    def run(self, code: str, output_dir: Path | None = None) -> dict:
        if output_dir is None:
            output_dir = self.cfg.data_path / "_output"

        valid, reason = self._validate(code)
        if not valid:
            return AnalysisResult(
                success=False, error=f"Code validation failed: {reason}"
            ).model_dump()

        full_script = (
            _HEADER_TEMPLATE.format(
                data_path=str(self.cfg.data_path),
                output_dir=str(output_dir),
            )
            + code
            + _FOOTER_TEMPLATE
        )

        start = time.monotonic()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(full_script)
            script_path = tf.name

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.cfg.sandbox_timeout_sec,
                env={**os.environ, "PYTHONPATH": ""},
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            if proc.returncode != 0:
                return AnalysisResult(
                    success=False,
                    stdout=proc.stdout,
                    error=proc.stderr[:2000],
                    execution_ms=elapsed_ms,
                ).model_dump()

            stdout = proc.stdout.strip()
            try:
                payload = json.loads(stdout)
                chart_paths = payload.pop("_chart_paths", [])
                return AnalysisResult(
                    success=True,
                    result=payload,
                    chart_paths=chart_paths,
                    execution_ms=elapsed_ms,
                ).model_dump()
            except json.JSONDecodeError:
                return AnalysisResult(
                    success=True,
                    stdout=stdout,
                    execution_ms=elapsed_ms,
                ).model_dump()

        except subprocess.TimeoutExpired:
            return AnalysisResult(
                success=False,
                error=f"Execution exceeded {self.cfg.sandbox_timeout_sec}s timeout.",
                execution_ms=self.cfg.sandbox_timeout_sec * 1000,
            ).model_dump()
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # AST validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in ALLOWED_MODULES:
                        return False, f"Blocked import: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                if top not in ALLOWED_MODULES:
                    return False, f"Blocked import: {node.module}"

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                    return False, f"Blocked call: {node.func.id}()"
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"system", "popen", "spawn", "exec_command"}:
                        return False, f"Blocked method: .{node.func.attr}()"

            elif isinstance(node, ast.Attribute):
                if node.attr in {"__class__", "__subclasses__", "__bases__", "__globals__",
                                 "__builtins__", "__code__", "__closure__"}:
                    return False, f"Blocked attribute access: .{node.attr}"

        return True, ""

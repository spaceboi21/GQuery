"""
XiaAgent — ReAct agent loop for the standalone XIA app.

Pattern: Reason → Act (tool call) → Observe (tool result) → repeat until
the LLM emits a plain-text response (no more tool calls).

Events emitted (as dicts, for SSE):
  {"type": "tool_start",  "name": str, "args": dict}
  {"type": "tool_end",    "name": str, "result": dict}
  {"type": "text",        "content": str}          ← streaming chunk
  {"type": "done"}
  {"type": "error",       "message": str}
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from .cells import read_cell_range, write_cells
from .config import Settings
from .indexer import FileIndexer
from .llm import LLMClient, ToolCall
from .sandbox import SandboxExecutor

MAX_TURNS = 10

# Pricing table: model_prefix → (input $/1M tokens, output $/1M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":                  (0.15,   0.60),
    "gpt-4o":                       (2.50,  10.00),
    "gpt-4-turbo":                  (10.00, 30.00),
    "gpt-4":                        (30.00, 60.00),
    "gpt-3.5-turbo":                (0.50,   1.50),
    "o1-mini":                      (3.00,  12.00),
    "o1":                           (15.00, 60.00),
    "o3-mini":                      (1.10,   4.40),
    "claude-3-5-sonnet":            (3.00,  15.00),
    "claude-3-5-haiku":             (0.80,   4.00),
    "claude-3-opus":                (15.00, 75.00),
    "claude-3-sonnet":              (3.00,  15.00),
    "claude-3-haiku":               (0.25,   1.25),
    "claude-opus-4":                (15.00, 75.00),
    "claude-sonnet-4":              (3.00,  15.00),
}


def _estimate_cost(usage: dict, model: str, provider: str) -> float:
    if provider == "ollama":
        return 0.0
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    for prefix, (ip, op) in _PRICING.items():
        if model.lower().startswith(prefix):
            return round(pt / 1_000_000 * ip + ct / 1_000_000 * op, 6)
    return 0.0


# Token-efficiency knobs (tune to trade context size vs. recall)
SCHEMA_MAX_SAMPLES = 2          # samples per column in get_schema (was 5)
SCHEMA_MAX_COLS = 60            # columns returned per sheet; over this we head/tail
HISTORY_MAX_TURNS = 8           # keep last N message pairs in flight
TOOL_RESULT_TRUNC_CHARS = 6000  # cap for tool-result JSON kept in history


def _compact_sheet(s) -> dict:
    """Compact SheetMeta → dict. Drops empty fields, caps samples/columns."""
    cols = s.columns
    truncated = False
    if len(cols) > SCHEMA_MAX_COLS:
        head = SCHEMA_MAX_COLS // 2
        tail = SCHEMA_MAX_COLS - head
        cols = list(cols[:head]) + list(cols[-tail:])
        truncated = True

    out: dict = {
        "sheet_name": s.sheet_name,
        "rows": s.row_count,
        "cols": s.col_count,
        "header_row": s.header_row,
        "columns": [_compact_column(c) for c in cols],
    }
    if s.sheet_type and s.sheet_type != "data":
        out["sheet_type"] = s.sheet_type
    if s.pivot_tables:
        out["pivot_tables"] = s.pivot_tables
    if truncated:
        out["columns_truncated"] = f"showing {SCHEMA_MAX_COLS} of {s.col_count} columns"
    return out


def _compact_column(c) -> dict:
    """Drop verbose dtype names + cap samples; omit empties."""
    dtype = c.dtype or ""
    if dtype in ("object", "unknown"):
        dtype = "str" if dtype == "object" else ""
    out: dict = {"name": c.name}
    if dtype:
        out["dtype"] = dtype.replace("datetime64[ns]", "datetime").replace("64", "")
    samples = (c.sample_values or [])[:SCHEMA_MAX_SAMPLES]
    if samples:
        out["samples"] = samples
    return out


def _truncate_tool_result(payload: dict | str) -> str:
    """Serialise + truncate a tool result before stuffing it into message history.

    Keeps the head of the JSON so the LLM still sees keys and the first rows;
    the user always sees the full untruncated result in the UI.
    """
    s = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    if len(s) <= TOOL_RESULT_TRUNC_CHARS:
        return s
    suffix = f"... [+{len(s) - TOOL_RESULT_TRUNC_CHARS} chars truncated to save tokens]"
    return s[: TOOL_RESULT_TRUNC_CHARS - len(suffix)] + suffix


def _trim_history(messages: list[dict]) -> list[dict]:
    """Bound message growth: keep the system prompt + last HISTORY_MAX_TURNS turns.

    Preserves tool_call ↔ tool result chains (cannot drop a tool result without
    its preceding assistant tool_calls message).
    """
    if len(messages) <= HISTORY_MAX_TURNS * 4 + 1:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]
    # Find a safe cut: not in the middle of a tool_calls→tool sequence
    target = max(1, len(body) - HISTORY_MAX_TURNS * 4)
    while target < len(body) and body[target].get("role") == "tool":
        target += 1
    return system + body[target:]

SYSTEM_PROMPT = """\
You are XIA, a local-first Excel/CSV analyst. Raw data stays on the user's machine; \
you see only schemas, samples, and tool results.

Workflow for any data question:
1. search_data(query) — always call FIRST.
2. get_schema(file_path) — confirm columns + header_row before code.
3. If user @-referenced a range, call read_cell_range BEFORE writing code.
4. run_analysis(code) for computation, modify_sheet(...) for writes.
5. Reply with bullets: headline finding, numbers, assumptions.

Rules for run_analysis:
- Use the exact `file_path` from get_schema (never reconstruct).
- Pass header=N where N = `header_row` from schema.
- Copy column names case-sensitively from schema; never guess.
- Charts: `import plotly.express as px` then `save_chart(fig, "name")`.
- Assign findings to a dict named `result` (auto-returned).

Rules for modify_sheet:
- State the planned writes (cell + formula) in your reply first.
- Prefer formulas (=VLOOKUP, =SUMIF) over literal values — they stay live.
- .xlsx/.xlsm only.

Pivot sheets (sheet_type='pivot_summary' or non-empty pivot_tables):
- Use `source_ref` to read the underlying detail data when asked "why".
- Do not re-read the rendered pivot cells.

Be concise. State assumptions when ambiguous. Never ask the user to paste data — use tools.
"""

# ------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_data",
            "description": "GraphRAG search. Call FIRST for any data question. Returns schema only, no cells.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Exact columns, dtypes, header_row, pivot_tables for a file. Call before writing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "sheet_name": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": "Sandboxed Python (pd, pl, np, plotly). Use absolute file_path + header=N from get_schema. Assign findings to dict `result`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_cell_range",
            "description": "Read an A1 range (≤2000 cells). Use when user @-references a selection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "range": {"type": "string", "description": "e.g. A1:C10"},
                },
                "required": ["file_path", "range"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_sheet",
            "description": "Write values/formulas to .xlsx cells. Formulas start with '='. State plan before calling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "sheet_name": {"type": "string"},
                    "writes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cell": {"type": "string"},
                                "value": {},
                                "formula": {"type": "string"},
                            },
                            "required": ["cell"],
                        },
                    },
                    "create_sheet": {"type": "boolean", "default": False},
                },
                "required": ["file_path", "writes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Enumerate indexed files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class XiaAgent:
    def __init__(self, settings: Settings, indexer: FileIndexer, executor: SandboxExecutor) -> None:
        self.cfg = settings
        self.indexer = indexer
        self.executor = executor
        self.llm = LLMClient(settings)

    async def run(
        self, query: str, history: list[dict]
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator — yields SSE event dicts until the agent is done.
        history is the prior conversation: [{"role": ..., "content": ...}, ...]
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for turn in range(MAX_TURNS):
            # Trim history before each turn to keep per-turn input cost bounded.
            messages = _trim_history(messages)

            # Collect all yielded items from the sync stream into buckets
            text_chunks: list[str] = []
            tool_calls: list[ToolCall] = []

            # Reset per-turn usage so we don't double-count if the provider
            # silently omits usage on a later turn.
            self.llm.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            for item in self.llm.stream_chat(messages, tools=TOOL_SCHEMAS):
                if isinstance(item, str):
                    text_chunks.append(item)
                    yield {"type": "text", "content": item}
                elif isinstance(item, ToolCall):
                    tool_calls.append(item)

            lu = self.llm.last_usage
            total_usage["prompt_tokens"] += lu.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += lu.get("completion_tokens", 0)
            total_usage["total_tokens"] += lu.get("total_tokens", 0)

            # --- Pure text response → we're done ---
            if not tool_calls:
                cost = _estimate_cost(total_usage, self.cfg.llm_model, self.cfg.llm_provider)
                yield {
                    "type": "usage",
                    "prompt_tokens": total_usage["prompt_tokens"],
                    "completion_tokens": total_usage["completion_tokens"],
                    "total_tokens": total_usage["total_tokens"],
                    "cost_usd": cost,
                    "model": self.cfg.llm_model,
                    "provider": self.cfg.llm_provider,
                }
                yield {"type": "done"}
                return

            # --- Suppress any partial text that preceded tool calls ---
            # (models sometimes emit a short preamble before tool_calls)
            full_text = "".join(text_chunks)

            # Append assistant turn (with tool calls) to messages
            messages.append({
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool call
            for tc in tool_calls:
                yield {"type": "tool_start", "name": tc.name, "args": tc.args}
                result = self._dispatch(tc)
                # Frontend gets the full result; the LLM sees a truncated copy
                # so a single huge tool output can't blow up future turns.
                yield {"type": "tool_end", "name": tc.name, "result": result}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate_tool_result(result),
                })

        # Safety: max turns exceeded
        cost = _estimate_cost(total_usage, self.cfg.llm_model, self.cfg.llm_provider)
        yield {
            "type": "usage",
            "prompt_tokens": total_usage["prompt_tokens"],
            "completion_tokens": total_usage["completion_tokens"],
            "total_tokens": total_usage["total_tokens"],
            "cost_usd": cost,
            "model": self.cfg.llm_model,
            "provider": self.cfg.llm_provider,
        }
        yield {
            "type": "error",
            "message": f"Reached maximum tool-call depth ({MAX_TURNS}). Partial results above.",
        }
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tc: ToolCall) -> dict:
        try:
            if tc.name == "search_data":
                return self._search_data(**tc.args)
            if tc.name == "get_schema":
                return self._get_schema(**tc.args)
            if tc.name == "run_analysis":
                return self._run_analysis(**tc.args)
            if tc.name == "read_cell_range":
                return self._read_cell_range(**tc.args)
            if tc.name == "modify_sheet":
                return self._modify_sheet(**tc.args)
            if tc.name == "list_files":
                return self._list_files()
            return {"error": f"Unknown tool: {tc.name}"}
        except Exception as exc:
            return {"error": str(exc)}

    def _search_data(self, query: str) -> dict:
        results = self.indexer.kg.search(
            query=query,
            fts_fn=self.indexer.fts_search,
            hop_limit=self.cfg.kg_hop_limit,
            top_k=5,
        )
        # Compact each hit: drop verbose graph fields, keep only what the LLM needs
        # to pick a file and call get_schema. Saves ~70% of tokens vs. full KG dump.
        compact: list[dict] = []
        for r in results:
            node = r.get("node", {}) or {}
            hit: dict = {
                "file_path": node.get("file_path", ""),
                "sheet": node.get("sheet_name") or node.get("label", ""),
                "rows": node.get("row_count", 0),
                "type": node.get("sheet_type", "data"),
                "matched": r.get("matched_headers", [])[:5],
            }
            related = r.get("related_nodes", []) or []
            if related:
                hit["related"] = [
                    {
                        "file_path": rel.get("file_path", ""),
                        "sheet": rel.get("sheet_name") or rel.get("label", ""),
                        "via": rel.get("shared_columns", [])[:3],
                        "rel": rel.get("edge_type", "SHARES_COLUMN"),
                    }
                    for rel in related[:3]
                ]
            compact.append(hit)
        return {"hits": compact}

    def _get_schema(self, file_path: str, sheet_name: str | None = None) -> dict:
        path = self.cfg.safe_path(file_path)
        meta = self.indexer.get_schema(path)

        # Token-saver: when the caller doesn't specify a sheet, return only an
        # index of sheets so the LLM can pick one and call back. This avoids
        # dumping every column of every sheet for multi-sheet workbooks.
        if not sheet_name and len(meta.sheets) > 1:
            return {
                "file_path": str(path),
                "size_mb": meta.size_mb,
                "sheets_index": [
                    {
                        "sheet_name": s.sheet_name,
                        "sheet_type": s.sheet_type,
                        "rows": s.row_count,
                        "cols": s.col_count,
                        "header_row": s.header_row,
                        "has_pivot": bool(s.pivot_tables),
                    }
                    for s in meta.sheets
                ],
                "note": "Multiple sheets — call get_schema again with sheet_name for full columns.",
            }

        sheets = [s for s in meta.sheets if s.sheet_name == sheet_name] if sheet_name else meta.sheets
        return {
            "file_path": str(path),
            "size_mb": meta.size_mb,
            "sheets": [_compact_sheet(s) for s in sheets],
        }

    def _run_analysis(self, code: str) -> dict:
        return self.executor.run(code=code)

    def _read_cell_range(
        self, file_path: str, range: str, sheet_name: str | None = None
    ) -> dict:
        path = self.cfg.safe_path(file_path)
        return read_cell_range(path, sheet_name=sheet_name, range_str=range)

    def _modify_sheet(
        self,
        file_path: str,
        writes: list[dict],
        sheet_name: str | None = None,
        create_sheet: bool = False,
    ) -> dict:
        path = self.cfg.safe_path(file_path)
        return write_cells(path, sheet_name, writes, create_sheet=create_sheet)

    def _list_files(self) -> dict:
        files = self.indexer.search_files()
        return {"files": files, "total": len(files)}

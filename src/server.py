"""
Excel Intelligence Agent (XIA) – FastMCP server.

Agentic workflow example ("Why did sales drop in Q3?")
-------------------------------------------------------
1. LLM calls generate_dashboard_summary(query="sales drop Q3")
   → KG returns relevant files/sheets + their schemas (no raw data)
2. LLM inspects the schema context and writes a Python analysis script.
3. LLM calls execute_python_analysis(code=<script>)
   → Sandbox runs it locally, returns JSON summary + chart paths.

Transport: stdio (local-only, CFO-safe – raw data never leaves the machine).
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

from .config import settings
from .indexer import FileIndexer
from .sandbox import SandboxExecutor

mcp = FastMCP(
    name="Excel Intelligence Agent (XIA)",
    instructions=(
        "Local-first Excel/CSV analyst. "
        "NEVER send raw cell data to the model – use schema context only. "
        "Workflow: generate_dashboard_summary → inspect schema → execute_python_analysis."
    ),
)

_indexer = FileIndexer(settings)
_executor = SandboxExecutor(settings)


# ──────────────────────────────────────────────
# Tool 1 – list_files
# ──────────────────────────────────────────────

@mcp.tool()
def list_files(query: str = "", file_type: str = "") -> dict:
    """
    Search all indexed Excel/CSV files.

    Args:
        query:     Natural-language search term matched against column headers via FTS5.
                   Leave empty to list all files.
        file_type: Filter by 'xlsx' or 'csv'. Leave empty for both.

    Returns:
        {"files": [...], "total": int}
        Each file entry includes path, name, size_mb, and a list of sheets with
        their headers and row counts (schema only – no cell data).
    """
    files = _indexer.search_files(query=query, file_type=file_type or "")
    return {"files": files, "total": len(files)}


# ──────────────────────────────────────────────
# Tool 2 – get_workbook_schema
# ──────────────────────────────────────────────

@mcp.tool()
def get_workbook_schema(file_path: str) -> dict:
    """
    Return the full schema for a workbook AND its Knowledge Graph relationships.

    Args:
        file_path: Absolute path to an indexed .xlsx or .csv file.

    Returns:
        {
          "schema":  FileMeta dict (sheets → columns → dtypes + 5 sample values),
          "kg_edges": list of SHARES_COLUMN edges linking this file's sheets
                      to related sheets in other files.
        }
        Use this before writing analysis code so you know exact column names and types.
    """
    path = settings.safe_path(file_path)
    schema = _indexer.get_schema(path)
    edges = _indexer.kg.get_related(str(path))
    return {
        "schema": schema.model_dump(mode="json"),
        "kg_edges": edges,
    }


# ──────────────────────────────────────────────
# Tool 3 – execute_python_analysis
# ──────────────────────────────────────────────

@mcp.tool()
def execute_python_analysis(code: str, output_dir: str = "") -> dict:
    """
    Execute a Python analysis script in a sandboxed subprocess.

    Security guarantees
    -------------------
    • Only pandas, polars, numpy, plotly, and stdlib math/datetime/re are importable.
    • exec/eval/open/os/subprocess are blocked at AST level.
    • Script runs in an isolated process with a hard timeout.
    • File I/O must use the injected DATA_PATH and OUTPUT_DIR constants.

    Injected constants (available in your script without importing)
    ---------------------------------------------------------------
        DATA_PATH  – Path to the approved data directory.
        OUTPUT_DIR – Path where charts/outputs should be saved.
        save_chart(fig, name) – helper to persist a Plotly figure.

    Convention
    ----------
    Your script should assign a dict to `result` with its findings.
    Example:
        df = pd.read_csv(DATA_PATH / "sales.csv")
        result = {"mean_sales": float(df["Sales"].mean()), "rows": len(df)}

    Args:
        code:       Python source to execute (see conventions above).
        output_dir: Optional override for where charts are written.

    Returns:
        AnalysisResult dict: {success, result, chart_paths, stdout, error, execution_ms}
    """
    out = settings.safe_path(output_dir) if output_dir else settings.data_path / "_output"
    return _executor.run(code=code, output_dir=out)


# ──────────────────────────────────────────────
# Tool 4 – generate_dashboard_summary
# ──────────────────────────────────────────────

@mcp.tool()
def generate_dashboard_summary(query: str) -> dict:
    """
    GraphRAG search: find files/sheets relevant to a business question.

    Uses FTS5 (keyword match on column headers) + NetworkX BFS expansion
    (multi-hop via shared columns) to surface the most relevant data context.

    This tool returns SCHEMA CONTEXT only – no raw cell data.
    Use the returned file_path + sheet_name values to write analysis code
    and then call execute_python_analysis.

    Args:
        query: A business question, e.g. "Why did sales drop in Q3?"

    Returns:
        {
          "query": str,
          "relevant_context": [
            {
              "node": {file_path, sheet_name, column_count, row_count},
              "relevance_score": float,
              "matched_headers": ["sales", "quarter"],
              "related_nodes": [...]   # sheets linked via shared columns
            }, ...
          ],
          "index_stats": {total_files, total_sheets, total_columns, total_edges}
        }
    """
    results = _indexer.kg.search(
        query=query,
        fts_fn=_indexer.fts_search,
        hop_limit=settings.kg_hop_limit,
        top_k=10,
    )
    return {
        "query": query,
        "relevant_context": results,
        "index_stats": _indexer.stats().model_dump(),
    }


# ──────────────────────────────────────────────
# Tool 5 – index_data_directory  (housekeeping)
# ──────────────────────────────────────────────

@mcp.tool()
def index_data_directory() -> dict:
    """
    (Re)index the data directory: crawl for new/changed .xlsx and .csv files,
    extract metadata, rebuild the Knowledge Graph.

    Call this once after adding new files, or on a schedule.
    Returns indexing summary: {added, updated, skipped, kg_edges}.
    """
    return _indexer.index_directory()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")

"""
XIA FastAPI server — main entry point for the standalone app.

Run:
    python src/api.py

Opens http://127.0.0.1:8000 automatically in the default browser.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import XiaAgent
from .cells import read_cell_range, read_sheet_data, write_cells
from .config import settings
from .indexer import FileIndexer
from .sandbox import SandboxExecutor

# ------------------------------------------------------------------
# App bootstrap
# ------------------------------------------------------------------

app = FastAPI(title="XIA — Excel Intelligence Agent", version="0.1.0")

_indexer = FileIndexer(settings)
_executor = SandboxExecutor(settings)
_agent = XiaAgent(settings, _indexer, _executor)

_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ------------------------------------------------------------------
# Frontend
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    html = _FRONTEND / "index.html"
    if not html.exists():
        raise HTTPException(404, "Frontend not found. Run from the project root.")
    return FileResponse(str(html))


# ------------------------------------------------------------------
# Chat — streaming SSE
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream agent events as Server-Sent Events.
    Event format:  data: <json>\\n\\n
    """
    if not settings.llm_api_key and settings.llm_provider != "ollama":
        async def _no_key():
            yield _sse({"type": "error", "message": (
                f"No API key set. Add XIA_LLM_API_KEY to your .env file "
                f"(provider: {settings.llm_provider})."
            )})
            yield _sse({"type": "done"})
        return StreamingResponse(_no_key(), media_type="text/event-stream")

    async def _stream():
        async for event in _agent.run(req.message, req.history):
            yield _sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------

@app.post("/api/index")
def index_directory(force: bool = Query(False)) -> dict:
    """Crawl data_path, index new/changed files, rebuild KG.
    Pass ?force=true to clear the index and re-process every file from scratch.
    """
    return _indexer.index_directory(force=force)


# ------------------------------------------------------------------
# Files & Schema
# ------------------------------------------------------------------

@app.get("/api/files")
def list_files(q: str = Query(""), type: str = Query("")) -> dict:
    files = _indexer.search_files(query=q, file_type=type)
    return {"files": files, "total": len(files)}


@app.get("/api/schema")
def get_schema(path: str = Query(...)) -> dict:
    try:
        resolved = settings.safe_path(path)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    try:
        meta = _indexer.get_schema(resolved)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return meta.model_dump(mode="json")


# ------------------------------------------------------------------
# Spreadsheet overlay — cell data & writes
# ------------------------------------------------------------------

@app.get("/api/sheet-data")
def sheet_data(
    path: str = Query(...),
    sheet: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> dict:
    """Load a paginated block of cells for the spreadsheet overlay."""
    try:
        resolved = settings.safe_path(path)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not resolved.exists():
        raise HTTPException(404, f"File not found: {resolved}")
    try:
        return read_sheet_data(resolved, sheet_name=sheet, offset=offset, limit=limit)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/cell-range")
def cell_range(
    path: str = Query(...),
    sheet: str | None = Query(None),
    range: str = Query(..., description="A1 notation, e.g. A1:C10"),
) -> dict:
    """Read an exact A1 range — used by chat references."""
    try:
        resolved = settings.safe_path(path)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    try:
        return read_cell_range(resolved, sheet_name=sheet, range_str=range)
    except ValueError as e:
        raise HTTPException(400, str(e))


class CellWrite(BaseModel):
    cell: str
    value: Any | None = None
    formula: str | None = None


class SheetWriteRequest(BaseModel):
    path: str
    sheet: str | None = None
    writes: list[dict]
    create_sheet: bool = False


@app.post("/api/sheet-write")
def sheet_write(body: SheetWriteRequest) -> dict:
    """Apply cell writes directly (used by frontend shortcuts, not agent)."""
    try:
        resolved = settings.safe_path(body.path)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    try:
        return write_cells(resolved, body.sheet, body.writes, body.create_sheet)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------
# Stats & Graph
# ------------------------------------------------------------------

@app.get("/api/stats")
def stats() -> dict:
    return _indexer.stats().model_dump()


@app.get("/api/graph")
def kg_graph() -> dict:
    """Return KG as {nodes, edges} for vis-network visualisation."""
    g = _indexer.kg._g
    nodes = [
        {
            "id": nid,
            "label": data.get("label", nid),
            "type": data.get("node_type", "sheet"),
            "sheet_type": data.get("sheet_type", "data"),
            "file_path": data.get("file_path", ""),
            "sheet_name": data.get("sheet_name"),
            "col_count": data.get("column_count", 0),
            "row_count": data.get("row_count", 0),
        }
        for nid, data in g.nodes(data=True)
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "shared_columns": data.get("shared_columns", []),
            "weight": data.get("weight", 0),
            "type": data.get("edge_type", "SHARES_COLUMN"),
            "relationship": data.get("relationship", ""),
        }
        for u, v, data in g.edges(data=True)
        if data.get("edge_type") in ("SHARES_COLUMN", "AGGREGATES_FROM", "HAS_SHEET")
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "files": sum(1 for n in nodes if n["type"] == "file"),
            "sheets": sum(1 for n in nodes if n["type"] == "sheet"),
            "shares_edges": sum(1 for e in edges if e["type"] == "SHARES_COLUMN"),
            "aggregates_edges": sum(1 for e in edges if e["type"] == "AGGREGATES_FROM"),
        },
    }


# ------------------------------------------------------------------
# Settings (runtime LLM config)
# ------------------------------------------------------------------

@app.get("/api/settings")
def get_settings() -> dict:
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_api_key_set": bool(settings.llm_api_key),
        "llm_base_url": settings.llm_base_url,
        "data_path": str(settings.data_path),
        "app_port": settings.app_port,
    }


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    data_path: str | None = None


@app.post("/api/settings")
def update_settings(body: SettingsUpdate) -> dict:
    """Hot-update settings without restarting. Rebuilds LLM client and indexer if data_path changes."""
    global _indexer, _executor, _agent

    if body.llm_provider is not None:
        settings.llm_provider = body.llm_provider
    if body.llm_model is not None:
        settings.llm_model = body.llm_model
    if body.llm_api_key is not None:
        settings.llm_api_key = body.llm_api_key
    if body.llm_base_url is not None:
        settings.llm_base_url = body.llm_base_url

    if body.data_path is not None:
        new_path = Path(body.data_path).resolve()
        if not new_path.exists():
            raise HTTPException(400, f"Path does not exist: {new_path}")
        new_path.mkdir(parents=True, exist_ok=True)
        settings.data_path = new_path
        # db lives inside data_path — move it to the new location
        settings.db_path = new_path / "xia_knowledge.db"
        # Re-bootstrap the data layer against the new path
        _indexer = FileIndexer(settings)
        _executor = SandboxExecutor(settings)
        _agent = XiaAgent(settings, _indexer, _executor)

    _agent.llm = _agent.llm.__class__(settings)
    return get_settings()


@app.get("/api/browse-folder")
async def browse_folder() -> dict:
    """
    Open the OS-native folder picker dialog (via tkinter) and return the selected path.
    Runs in a thread so it doesn't block the event loop.
    """
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, _open_folder_dialog)
    return {"path": path}


def _open_folder_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select your data folder")
        root.destroy()
        return str(path) if path else ""
    except Exception:
        return ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"



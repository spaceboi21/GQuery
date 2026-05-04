from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


class ColumnMeta(BaseModel):
    name: str
    dtype: str
    null_rate: float = 0.0
    sample_values: list[Any] = Field(default_factory=list)


class SheetMeta(BaseModel):
    sheet_name: str
    row_count: int
    col_count: int
    columns: list[ColumnMeta]
    # Named ranges that reference this sheet (xlsx only)
    named_ranges: dict[str, str] = Field(default_factory=dict)
    # Which 0-based row index was detected as the real header (0 = normal, >0 = skip leading metadata)
    header_row: int = 0
    # Structural classification: "pivot_summary" | "detail" | "data"
    sheet_type: str = "data"
    # Pivot tables defined on this sheet (extracted via openpyxl)
    pivot_tables: list[dict[str, Any]] = Field(default_factory=list)


class FileMeta(BaseModel):
    path: Path
    name: str
    file_type: str  # 'xlsx' | 'csv'
    size_mb: float
    modified_at: float
    sheets: list[SheetMeta]


class GraphNode(BaseModel):
    node_id: str
    node_type: str  # 'file' | 'sheet'
    label: str
    file_path: str
    sheet_name: str | None = None
    column_count: int = 0
    row_count: int = 0
    sheet_type: str = "data"  # 'pivot_summary' | 'detail' | 'data'


class GraphEdge(BaseModel):
    source: str
    target: str
    shared_columns: list[str]
    weight: float
    edge_type: str = "SHARES_COLUMN"
    relationship: str = ""


class KGSearchResult(BaseModel):
    node: GraphNode
    relevance_score: float
    matched_headers: list[str]
    # list of dicts (richer than GraphNode — includes edge_type / shared_columns)
    related_nodes: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    success: bool
    stdout: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    chart_paths: list[str] = Field(default_factory=list)
    error: str | None = None
    execution_ms: int = 0


class IndexStats(BaseModel):
    total_files: int
    total_sheets: int
    total_columns: int
    total_edges: int
    data_path: str
    db_path: str

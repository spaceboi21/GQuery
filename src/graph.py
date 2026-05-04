"""
Knowledge Graph over Excel/CSV files.

Nodes  – one per file, one per sheet/table.
Edges  – weighted by number of shared column headers (the "join keys").

GraphRAG query pattern
----------------------
1. FTS5 returns seed sheet_ids for a query.
2. BFS up to `hop_limit` expands to related sheets via shared-column edges.
3. Nodes are ranked by (hits * edge_weight) and returned as context for the LLM.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from typing import Callable

import networkx as nx

from .schema import GraphEdge, GraphNode, KGSearchResult


class KnowledgeGraph:
    def __init__(self) -> None:
        self._g: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_file_node(self, file_id: int, file_path: str, name: str) -> None:
        nid = f"file::{file_id}"
        self._g.add_node(nid, node_type="file", label=name, file_path=file_path, sheet_name=None,
                         column_count=0, row_count=0)

    def add_sheet_node(self, sheet_id: int, file_id: int, file_path: str,
                       sheet_name: str, col_count: int, row_count: int,
                       sheet_type: str = "data") -> None:
        nid = f"sheet::{sheet_id}"
        file_nid = f"file::{file_id}"
        self._g.add_node(nid, node_type="sheet", label=sheet_name,
                         file_path=file_path, sheet_name=sheet_name,
                         column_count=col_count, row_count=row_count,
                         sheet_type=sheet_type)
        self._g.add_edge(file_nid, nid, shared_columns=[], weight=1.0, edge_type="HAS_SHEET")

    def build_column_edges(self, db: sqlite3.Connection) -> int:
        """
        Derive SHARES_COLUMN and AGGREGATES_FROM edges from the sheets table.

        SHARES_COLUMN  — two sheets share at least one column header (join candidates).
        AGGREGATES_FROM — summary sheet → detail sheet when:
            • they share ≥2 dimension columns, AND
            • the summary has <20 % the rows of the detail, AND
            • the summary has 'pivot_summary' classification OR 'grand total' column.
        """
        rows = db.execute(
            "SELECT id, headers_json, row_count, sheet_type FROM sheets"
        ).fetchall()

        sheet_headers: dict[int, set[str]] = {}
        sheet_rows: dict[int, int] = {}
        sheet_types: dict[int, str] = {}

        for r in rows:
            sid = r[0]
            sheet_headers[sid] = {h.strip().lower() for h in json.loads(r[1]) if h}
            sheet_rows[sid] = r[2] or 0
            sheet_types[sid] = r[3] or "data"

        new_edges = 0
        sheet_ids = list(sheet_headers.keys())

        for i, sid_a in enumerate(sheet_ids):
            for sid_b in sheet_ids[i + 1:]:
                shared = sheet_headers[sid_a] & sheet_headers[sid_b]
                if not shared:
                    continue

                nid_a, nid_b = f"sheet::{sid_a}", f"sheet::{sid_b}"
                weight = len(shared) / max(
                    len(sheet_headers[sid_a]), len(sheet_headers[sid_b]), 1
                )

                # Determine if one sheet aggregates the other
                rows_a, rows_b = sheet_rows[sid_a], sheet_rows[sid_b]
                agg_edge: tuple[str, str] | None = None

                if len(shared) >= 2 and rows_a > 0 and rows_b > 0:
                    ratio = min(rows_a, rows_b) / max(rows_a, rows_b)
                    if ratio < 0.20:
                        small_sid = sid_a if rows_a < rows_b else sid_b
                        large_sid = sid_b if rows_a < rows_b else sid_a
                        small_type = sheet_types[small_sid]
                        small_headers = sheet_headers[small_sid]
                        if (small_type == "pivot_summary"
                                or "grand total" in small_headers
                                or any("total" in h for h in small_headers)):
                            agg_edge = (f"sheet::{small_sid}", f"sheet::{large_sid}")

                if agg_edge:
                    # Add the directed AGGREGATES_FROM edge (small → large)
                    self._g.add_edge(
                        agg_edge[0], agg_edge[1],
                        shared_columns=sorted(shared),
                        weight=round(weight, 4),
                        edge_type="AGGREGATES_FROM",
                        relationship="Summary sheet aggregates from detail sheet",
                    )
                else:
                    self._g.add_edge(nid_a, nid_b,
                                     shared_columns=sorted(shared),
                                     weight=round(weight, 4),
                                     edge_type="SHARES_COLUMN")
                new_edges += 1

        return new_edges

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        fts_fn: Callable[[str], list[dict]],
        hop_limit: int = 2,
        top_k: int = 10,
    ) -> list[dict]:
        """
        GraphRAG search:
        FTS5 seeds → BFS expansion → ranked KGSearchResult list (as dicts).
        """
        seeds = fts_fn(query)
        if not seeds:
            return []

        seed_nids: dict[str, list[str]] = {}
        for hit in seeds:
            nid = f"sheet::{hit['sheet_id']}"
            seed_nids.setdefault(nid, []).append(hit["header"])

        scored: dict[str, float] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        for nid in seed_nids:
            if nid in self._g:
                queue.append((nid, 0))
                scored[nid] = len(seed_nids[nid]) * 2.0

        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if depth >= hop_limit:
                continue
            for neighbor, edge_data in self._g[current].items():
                if self._g.nodes[neighbor].get("node_type") == "sheet":
                    w = edge_data.get("weight", 0.5)
                    scored[neighbor] = scored.get(neighbor, 0.0) + w
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        results: list[KGSearchResult] = []
        for nid, score in sorted(scored.items(), key=lambda x: -x[1])[:top_k]:
            if nid not in self._g:
                continue
            attr = self._g.nodes[nid]
            node = GraphNode(
                node_id=nid,
                node_type=attr.get("node_type", "sheet"),
                label=attr.get("label", ""),
                file_path=attr.get("file_path", ""),
                sheet_name=attr.get("sheet_name"),
                column_count=attr.get("column_count", 0),
                row_count=attr.get("row_count", 0),
                sheet_type=attr.get("sheet_type", "data"),
            )
            # Include neighbor info and the edge relationship type
            related_with_context = []
            for nb in self._g.neighbors(nid):
                if self._g.nodes[nb].get("node_type") != "sheet" or nb == nid:
                    continue
                nb_attr = self._g.nodes[nb]
                edge_data = self._g[nid][nb]
                related_with_context.append({
                    "node_id": nb,
                    "label": nb_attr.get("label", ""),
                    "file_path": nb_attr.get("file_path", ""),
                    "sheet_name": nb_attr.get("sheet_name"),
                    "row_count": nb_attr.get("row_count", 0),
                    "sheet_type": nb_attr.get("sheet_type", "data"),
                    "edge_type": edge_data.get("edge_type", "SHARES_COLUMN"),
                    "shared_columns": edge_data.get("shared_columns", []),
                    "relationship": edge_data.get("relationship", ""),
                })
            results.append(KGSearchResult(
                node=node,
                relevance_score=round(score, 4),
                matched_headers=seed_nids.get(nid, []),
                related_nodes=related_with_context[:5],
            ))

        return [r.model_dump() for r in results]

    def get_related(self, file_path: str, sheet_name: str | None = None) -> list[dict]:
        """Return immediate neighbors for a given file/sheet (for get_workbook_schema)."""
        matches = [
            nid for nid, attr in self._g.nodes(data=True)
            if attr.get("file_path") == file_path
            and (sheet_name is None or attr.get("sheet_name") == sheet_name)
        ]
        edges: list[GraphEdge] = []
        for nid in matches:
            for nb, edata in self._g[nid].items():
                et = edata.get("edge_type", "")
                if et in ("SHARES_COLUMN", "AGGREGATES_FROM"):
                    edges.append(GraphEdge(
                        source=nid,
                        target=nb,
                        shared_columns=edata.get("shared_columns", []),
                        weight=edata.get("weight", 0.0),
                        edge_type=et,
                        relationship=edata.get("relationship", ""),
                    ))
        return [e.model_dump() for e in edges]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, db: sqlite3.Connection) -> None:
        nodes = {n: dict(d) for n, d in self._g.nodes(data=True)}
        edges = [
            {"u": u, "v": v, **dict(d)}
            for u, v, d in self._g.edges(data=True)
        ]
        db.execute(
            "INSERT OR REPLACE INTO graph_state(id, nodes_json, edges_json, updated_at) "
            "VALUES (1, ?, ?, ?)",
            (json.dumps(nodes), json.dumps(edges), time.time()),
        )
        db.commit()

    def load(self, db: sqlite3.Connection) -> bool:
        row = db.execute(
            "SELECT nodes_json, edges_json FROM graph_state WHERE id = 1"
        ).fetchone()
        if not row:
            return False
        self._g = nx.Graph()
        for nid, attrs in json.loads(row[0]).items():
            self._g.add_node(nid, **attrs)
        for e in json.loads(row[1]):
            u, v = e.pop("u"), e.pop("v")
            self._g.add_edge(u, v, **e)
        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def edge_count(self) -> int:
        return sum(
            1 for _, _, d in self._g.edges(data=True)
            if d.get("edge_type") == "SHARES_COLUMN"
        )

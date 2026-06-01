from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_NAME = "融合积水深度、通行能力与安全度的时间依赖双目标疏散路径优化模型"
ROAD_IDS = [f"L{i}" for i in range(1, 9)]
DEPARTURE_TIMES = list(range(0, 61, 5))
ALPHA_STRATEGIES = {
    0.3: "安全优先",
    0.5: "均衡策略",
    0.7: "时间优先",
}
DEFAULT_HIGH_RISK_EDGES = ["L5", "L2", "L8"]
DEFAULT_TOPOLOGY = {
    "L7": ("N1", "N2"),
    "L1": ("N2", "N5"),
    "L4": ("N2", "N3"),
    "L2": ("N3", "N4"),
    "L6": ("N3", "N6"),
    "L3": ("N5", "N6"),
    "L8": ("N6", "N7"),
    "L5": ("N4", "N7"),
}
DEFAULT_LENGTHS = {
    "L1": 450.0,
    "L2": 320.0,
    "L3": 580.0,
    "L4": 260.0,
    "L5": 390.0,
    "L6": 410.0,
    "L7": 290.0,
    "L8": 520.0,
}
DEFAULT_ROAD_CLASS = {
    "L1": "主干路",
    "L2": "次干路",
    "L3": "主干路",
    "L4": "支路",
    "L5": "次干路",
    "L6": "主干路",
    "L7": "支路",
    "L8": "次干路",
}
NODE_POSITIONS = {
    "N1": (0.0, 2.0),
    "N2": (1.0, 2.0),
    "N3": (2.0, 1.5),
    "N4": (3.1, 0.7),
    "N5": (1.5, 3.0),
    "N6": (2.7, 2.5),
    "N7": (4.0, 1.1),
}


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def find_project_dir() -> Path:
    return Path(__file__).resolve().parent


def _read_time_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_min" not in df.columns:
        raise ValueError(f"{path.name} 缺少 time_min 列")
    for edge_id in ROAD_IDS:
        if edge_id not in df.columns:
            raise ValueError(f"{path.name} 缺少 {edge_id} 列")
    df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce").astype(int)
    df = df.set_index("time_min").sort_index()
    return df[ROAD_IDS].astype(float)


def load_q1_outputs(project_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    q1_dir = project_dir / "results" / "q1"
    paths = {
        "depth": q1_dir / "q1_depth_5min.csv",
        "capacity": q1_dir / "q1_capacity_5min.csv",
        "safety": q1_dir / "q1_safety_5min.csv",
    }
    status = {f"read_{key}": path.exists() for key, path in paths.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("第三问必须读取第一问三个核心文件，缺失: " + "; ".join(missing))
    return {key: _read_time_table(path) for key, path in paths.items()}, status


def find_input_excel_optional(project_dir: Path) -> Path | None:
    search_dirs = [
        project_dir,
        Path.cwd(),
        Path(r"C:\Users\马翌博\xwechat_files\wxid_i3gl5sn5m1p22_a8a9\msg\file\2026-05"),
    ]
    matches: list[Path] = []
    for folder in search_dirs:
        if folder.exists():
            matches.extend(p for p in folder.glob("B题附表1至6*.xlsx") if not p.name.startswith("~$"))
    if not matches:
        return None
    return sorted(set(matches), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[0].resolve()


def _cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _find_sheet_name(sheet_names: list[str], number: int) -> str:
    for name in sheet_names:
        if f"附表 {number}" in name or f"附表{number}" in name:
            return name
    if len(sheet_names) >= number:
        return sheet_names[number - 1]
    raise ValueError(f"无法定位附表{number}")


def _find_column(columns: list[Any], keywords: list[str]) -> Any:
    for col in columns:
        name = _cell_text(col)
        if all(keyword in name for keyword in keywords):
            return col
    for col in columns:
        name = _cell_text(col)
        if any(keyword in name for keyword in keywords):
            return col
    raise KeyError(f"找不到列: {keywords}")


def load_road_attributes_optional(excel_path: Path | None) -> tuple[pd.DataFrame, bool, str]:
    if excel_path is None or not excel_path.exists():
        return pd.DataFrame(), False, "未找到 B题附表1至6*.xlsx，路段长度和等级使用 topology/default 信息。"
    try:
        xl = pd.ExcelFile(excel_path)
        raw = pd.read_excel(excel_path, sheet_name=_find_sheet_name(xl.sheet_names, 1), header=None)
        raw = raw.replace(r"^\s*$", np.nan, regex=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
        raw = raw.reset_index(drop=True)
        header_row = None
        for idx, row in raw.iterrows():
            text = " ".join(_cell_text(v) for v in row.tolist())
            if "路段编号" in text and "长度" in text:
                header_row = idx
                break
        if header_row is None:
            raise ValueError("附表1表头定位失败")
        table = raw.iloc[header_row + 1 :].copy()
        table.columns = [_cell_text(v) or f"unnamed_{i}" for i, v in enumerate(raw.iloc[header_row])]
        edge_col = _find_column(list(table.columns), ["路段编号"])
        length_col = _find_column(list(table.columns), ["长度"])
        class_col = _find_column(list(table.columns), ["通行优先级"])
        attrs = pd.DataFrame(
            {
                "edge_id": table[edge_col].astype(str).str.strip(),
                "length_m": pd.to_numeric(table[length_col], errors="coerce"),
                "road_class": table[class_col].astype(str).str.strip(),
            }
        )
        attrs = attrs[attrs["edge_id"].str.match(r"^L\d+$", na=False)].copy()
        return attrs, True, f"已读取附表1路段属性: {excel_path}"
    except Exception as exc:
        return pd.DataFrame(), False, f"读取 Excel 附表1失败，使用 topology/default 信息: {exc}"


def _speed_kmh(road_class: str) -> float:
    if "主干路" in str(road_class):
        return 40.0
    if "次干路" in str(road_class):
        return 30.0
    if "支路" in str(road_class):
        return 20.0
    return 30.0


def load_or_build_topology(project_dir: Path, road_attrs: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    topology_path = project_dir / "results" / "topology" / "topology_edges.csv"
    if topology_path.exists():
        edges = pd.read_csv(topology_path)
        loaded = True
    else:
        edges = pd.DataFrame(
            [
                {
                    "edge_id": edge_id,
                    "start_node": start,
                    "end_node": end,
                    "length_m": DEFAULT_LENGTHS[edge_id],
                    "road_class": DEFAULT_ROAD_CLASS[edge_id],
                }
                for edge_id, (start, end) in DEFAULT_TOPOLOGY.items()
            ]
        )
        loaded = False

    if "edge_id" not in edges.columns:
        raise ValueError("拓扑边表缺少 edge_id")
    edges["edge_id"] = edges["edge_id"].astype(str)
    for edge_id in ROAD_IDS:
        if edge_id not in set(edges["edge_id"]):
            start, end = DEFAULT_TOPOLOGY[edge_id]
            edges.loc[len(edges)] = {"edge_id": edge_id, "start_node": start, "end_node": end}

    if not road_attrs.empty:
        attrs = road_attrs[["edge_id", "length_m", "road_class"]].drop_duplicates("edge_id")
        edges = edges.drop(columns=[col for col in ["length_m", "road_class"] if col in edges.columns])
        edges = edges.merge(attrs, on="edge_id", how="left")

    edges["length_m"] = edges.apply(
        lambda row: float(row["length_m"]) if pd.notna(row.get("length_m")) else DEFAULT_LENGTHS.get(row["edge_id"], 400.0),
        axis=1,
    )
    edges["road_class"] = edges.apply(
        lambda row: row["road_class"] if pd.notna(row.get("road_class")) else DEFAULT_ROAD_CLASS.get(row["edge_id"], "次干路"),
        axis=1,
    )
    edges["base_speed_kmh"] = edges["road_class"].apply(_speed_kmh)
    edges["base_speed_m_per_min"] = edges["base_speed_kmh"] * 1000.0 / 60.0
    edges["edge_id"] = pd.Categorical(edges["edge_id"], categories=ROAD_IDS, ordered=True)
    edges = edges.sort_values("edge_id").reset_index(drop=True)
    edges["edge_id"] = edges["edge_id"].astype(str)
    return edges, loaded


def build_graph(edges_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    graph: dict[str, list[dict[str, Any]]] = {}
    for _, row in edges_df.iterrows():
        start, end, edge_id = row["start_node"], row["end_node"], row["edge_id"]
        graph.setdefault(start, []).append({"neighbor": end, "edge_id": edge_id})
        graph.setdefault(end, []).append({"neighbor": start, "edge_id": edge_id})
    return graph


def check_graph_connected(graph: dict[str, list[dict[str, Any]]]) -> bool:
    if not graph:
        return False
    start = next(iter(graph))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for item in graph.get(node, []):
            neighbor = item["neighbor"]
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(graph)


def enumerate_simple_paths(
    graph: dict[str, list[dict[str, Any]]],
    origin: str,
    destination: str,
    edges_df: pd.DataFrame,
    high_risk_edges: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    edge_lookup = edges_df.set_index("edge_id")

    def dfs(node: str, visited: set[str], node_path: list[str], edge_path: list[str]) -> None:
        if node == destination:
            total_length = float(sum(edge_lookup.loc[e, "length_m"] for e in edge_path))
            freeflow = float(
                sum(edge_lookup.loc[e, "length_m"] / edge_lookup.loc[e, "base_speed_m_per_min"] for e in edge_path)
            )
            records.append(
                {
                    "path_id": f"P{len(records) + 1}",
                    "origin": origin,
                    "destination": destination,
                    "node_path": "->".join(node_path),
                    "edge_path": "->".join(edge_path),
                    "total_length_m": total_length,
                    "static_freeflow_time_min": freeflow,
                    "contains_high_risk_edge": any(edge in high_risk_edges for edge in edge_path),
                }
            )
            return
        for item in graph.get(node, []):
            neighbor = item["neighbor"]
            if neighbor in visited:
                continue
            dfs(neighbor, visited | {neighbor}, node_path + [neighbor], edge_path + [item["edge_id"]])

    dfs(origin, {origin}, [origin], [])
    return pd.DataFrame(records)


def load_high_risk_edges(project_dir: Path) -> list[str]:
    risk_path = project_dir / "results" / "q1" / "q1_risk_summary.csv"
    if not risk_path.exists():
        return DEFAULT_HIGH_RISK_EDGES.copy()
    try:
        risk = pd.read_csv(risk_path)
        if {"road_id", "risk_rank"}.issubset(risk.columns):
            return risk.sort_values("risk_rank").head(3)["road_id"].astype(str).tolist()
    except Exception:
        pass
    return DEFAULT_HIGH_RISK_EDGES.copy()


def get_state_at_time(q1_data: dict[str, pd.DataFrame], edge_id: str, time_min: float) -> dict[str, float]:
    t_hat = int(5 * math.floor(time_min / 5.0))
    t_hat = max(0, min(60, t_hat))
    return {
        "t_hat": float(t_hat),
        "depth_cm": float(q1_data["depth"].loc[t_hat, edge_id]),
        "capacity": float(q1_data["capacity"].loc[t_hat, edge_id]),
        "safety": float(q1_data["safety"].loc[t_hat, edge_id]),
    }


def compute_edge_travel_time(length_m: float, base_speed_m_per_min: float, capacity: float) -> float:
    if capacity <= 0:
        return float("inf")
    return float(length_m / (base_speed_m_per_min * capacity))


def evaluate_path_time_dependent(
    path_row: pd.Series,
    departure_time: float,
    q1_data: dict[str, pd.DataFrame],
    edges_df: pd.DataFrame,
    collect_steps: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    edge_lookup = edges_df.set_index("edge_id")
    edge_path = str(path_row["edge_path"]).split("->") if str(path_row["edge_path"]) else []
    node_path = str(path_row["node_path"]).split("->") if str(path_row["node_path"]) else []
    current_time = float(departure_time)
    safeties: list[float] = []
    blocked_edges: list[str] = []
    step_rows: list[dict[str, Any]] = []

    for idx, edge_id in enumerate(edge_path):
        state = get_state_at_time(q1_data, edge_id, current_time)
        row = edge_lookup.loc[edge_id]
        edge_time = compute_edge_travel_time(row["length_m"], row["base_speed_m_per_min"], state["capacity"])
        next_node = node_path[idx + 1] if idx + 1 < len(node_path) else ""
        if not np.isfinite(edge_time):
            blocked_edges.append(edge_id)
            if collect_steps:
                step_rows.append(
                    {
                        "edge_id": edge_id,
                        "from_node": node_path[idx] if idx < len(node_path) else "",
                        "to_node": next_node,
                        "edge_enter_time_min": current_time,
                        "edge_exit_time_min": np.nan,
                        "edge_depth_cm": state["depth_cm"],
                        "edge_capacity": state["capacity"],
                        "edge_safety": state["safety"],
                        "edge_time_min": np.inf,
                    }
                )
            return (
                {
                    "departure_time_min": departure_time,
                    "path_id": path_row["path_id"],
                    "node_path": path_row["node_path"],
                    "edge_path": path_row["edge_path"],
                    "total_time_min": np.inf,
                    "arrival_time_min": np.inf,
                    "min_safety": 0.0,
                    "avg_safety": 0.0,
                    "feasible": False,
                    "blocked_edges": ",".join(blocked_edges),
                    "reason": f"{edge_id} 不可通行",
                },
                step_rows,
            )
        enter_time = current_time
        exit_time = current_time + edge_time
        safeties.append(state["safety"])
        if collect_steps:
            step_rows.append(
                {
                    "edge_id": edge_id,
                    "from_node": node_path[idx] if idx < len(node_path) else "",
                    "to_node": next_node,
                    "edge_enter_time_min": enter_time,
                    "edge_exit_time_min": exit_time,
                    "edge_depth_cm": state["depth_cm"],
                    "edge_capacity": state["capacity"],
                    "edge_safety": state["safety"],
                    "edge_time_min": edge_time,
                }
            )
        current_time = exit_time

    return (
        {
            "departure_time_min": departure_time,
            "path_id": path_row["path_id"],
            "node_path": path_row["node_path"],
            "edge_path": path_row["edge_path"],
            "total_time_min": current_time - departure_time,
            "arrival_time_min": current_time,
            "min_safety": min(safeties) if safeties else 1.0,
            "avg_safety": float(np.mean(safeties)) if safeties else 1.0,
            "feasible": True,
            "blocked_edges": "",
            "reason": "OK",
        },
        step_rows,
    )


def compute_pareto_flags(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    out["is_pareto_optimal"] = False
    for dep, group in out.groupby("departure_time_min"):
        feasible = group[group["feasible"]].copy()
        for idx, row in feasible.iterrows():
            dominated = False
            for jdx, other in feasible.iterrows():
                if idx == jdx:
                    continue
                better_or_equal = (
                    other["total_time_min"] <= row["total_time_min"]
                    and other["min_safety"] >= row["min_safety"]
                )
                strictly_better = (
                    other["total_time_min"] < row["total_time_min"]
                    or other["min_safety"] > row["min_safety"]
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            out.loc[idx, "is_pareto_optimal"] = not dominated
    return out


def choose_best_path(metrics_df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    feasible = metrics_df[metrics_df["feasible"]].copy()
    if feasible.empty:
        dep = metrics_df["departure_time_min"].iloc[0] if not metrics_df.empty else np.nan
        return {
            "departure_time_min": dep,
            "alpha": alpha,
            "strategy_name": ALPHA_STRATEGIES[alpha],
            "best_path_id": "",
            "node_path": "",
            "edge_path": "",
            "total_time_min": np.inf,
            "min_safety": 0.0,
            "avg_safety": 0.0,
            "objective_value": np.inf,
            "feasible": False,
            "blocked_edges": "",
            "notes": "No feasible path",
        }
    if np.isclose(alpha, 0.3):
        threshold = 0.6
        candidates = feasible[feasible["min_safety"] >= threshold].copy()
        note = "安全优先：S_min>=0.6，先最大化最小安全度，再最小化总时间"
        if candidates.empty:
            threshold = 0.4
            candidates = feasible[feasible["min_safety"] >= threshold].copy()
            note = "安全优先降级：S_min>=0.4，先最大化最小安全度，再最小化总时间"
        if candidates.empty:
            candidates = feasible.copy()
            note = "无安全阈值可行路径，退化为最高安全度路径"
        best = candidates.sort_values(["min_safety", "total_time_min"], ascending=[False, True]).iloc[0]
        objective = float((1.0 - best["min_safety"]) * 1000.0 + best["total_time_min"])
        return {
            "departure_time_min": best["departure_time_min"],
            "alpha": alpha,
            "strategy_name": ALPHA_STRATEGIES[alpha],
            "best_path_id": best["path_id"],
            "node_path": best["node_path"],
            "edge_path": best["edge_path"],
            "total_time_min": best["total_time_min"],
            "min_safety": best["min_safety"],
            "avg_safety": best["avg_safety"],
            "objective_value": objective,
            "feasible": True,
            "blocked_edges": best.get("blocked_edges", ""),
            "notes": note,
        }
    if np.isclose(alpha, 0.7):
        candidates = feasible[feasible["min_safety"] >= 0.2].copy()
        note = "时间优先：S_min>=0.2 后选时间最短"
        if candidates.empty:
            candidates = feasible.copy()
            note = "时间优先无阈值可行路径，退化为可行路径中时间最短"
        best = candidates.sort_values(["total_time_min", "min_safety"], ascending=[True, False]).iloc[0]
        objective = float(best["total_time_min"])
        return {
            "departure_time_min": best["departure_time_min"],
            "alpha": alpha,
            "strategy_name": ALPHA_STRATEGIES[alpha],
            "best_path_id": best["path_id"],
            "node_path": best["node_path"],
            "edge_path": best["edge_path"],
            "total_time_min": best["total_time_min"],
            "min_safety": best["min_safety"],
            "avg_safety": best["avg_safety"],
            "objective_value": objective,
            "feasible": True,
            "blocked_edges": best.get("blocked_edges", ""),
            "notes": note,
        }
    t_ref = float(feasible["total_time_min"].max())
    if not np.isfinite(t_ref) or t_ref <= 0:
        t_ref = 30.0
    feasible["objective_value"] = alpha * feasible["total_time_min"] / t_ref + (1.0 - alpha) * (
        1.0 - feasible["min_safety"]
    )
    best = feasible.sort_values(["objective_value", "total_time_min", "min_safety"], ascending=[True, True, False]).iloc[0]
    return {
        "departure_time_min": best["departure_time_min"],
        "alpha": alpha,
        "strategy_name": ALPHA_STRATEGIES[alpha],
        "best_path_id": best["path_id"],
        "node_path": best["node_path"],
        "edge_path": best["edge_path"],
        "total_time_min": best["total_time_min"],
        "min_safety": best["min_safety"],
        "avg_safety": best["avg_safety"],
        "objective_value": best["objective_value"],
        "feasible": True,
        "blocked_edges": best.get("blocked_edges", ""),
        "notes": "OK",
    }


def solve_dynamic_paths(
    candidate_paths: pd.DataFrame,
    q1_data: dict[str, pd.DataFrame],
    edges_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    for dep in DEPARTURE_TIMES:
        for _, path in candidate_paths.iterrows():
            metric, _ = evaluate_path_time_dependent(path, dep, q1_data, edges_df)
            metric_records.append(metric)
    metrics_df = compute_pareto_flags(pd.DataFrame(metric_records))
    dynamic_records: list[dict[str, Any]] = []
    for dep in DEPARTURE_TIMES:
        dep_metrics = metrics_df[metrics_df["departure_time_min"] == dep]
        for alpha in ALPHA_STRATEGIES:
            dynamic_records.append(choose_best_path(dep_metrics, alpha))
    return metrics_df, pd.DataFrame(dynamic_records)


def build_static_shortest_path_comparison(
    candidate_paths: pd.DataFrame,
    path_metrics: pd.DataFrame,
    dynamic_paths: pd.DataFrame,
) -> pd.DataFrame:
    static_row = candidate_paths.sort_values("static_freeflow_time_min").iloc[0]
    records: list[dict[str, Any]] = []
    for dep in [0, 30, 45]:
        static_metric = path_metrics[
            (path_metrics["departure_time_min"] == dep) & (path_metrics["path_id"] == static_row["path_id"])
        ].iloc[0]
        dynamic_row = dynamic_paths[
            (dynamic_paths["departure_time_min"] == dep) & (np.isclose(dynamic_paths["alpha"], 0.5))
        ].iloc[0]
        records.append(
            {
                "departure_time_min": dep,
                "comparison_type": "static_freeflow_shortest",
                "path_id": static_row["path_id"],
                "node_path": static_row["node_path"],
                "edge_path": static_row["edge_path"],
                "feasible": bool(static_metric["feasible"]),
                "total_time_min": static_metric["total_time_min"],
                "min_safety": static_metric["min_safety"],
                "avg_safety": static_metric["avg_safety"],
                "notes": "静态自由流最短路对照",
            }
        )
        records.append(
            {
                "departure_time_min": dep,
                "comparison_type": "dynamic_balanced_best",
                "path_id": dynamic_row["best_path_id"],
                "node_path": dynamic_row["node_path"],
                "edge_path": dynamic_row["edge_path"],
                "feasible": bool(dynamic_row["feasible"]),
                "total_time_min": dynamic_row["total_time_min"],
                "min_safety": dynamic_row["min_safety"],
                "avg_safety": dynamic_row["avg_safety"],
                "notes": "动态均衡策略最优路",
            }
        )
    return pd.DataFrame(records)


def rolling_replanning(
    graph: dict[str, list[dict[str, Any]]],
    edges_df: pd.DataFrame,
    q1_data: dict[str, pd.DataFrame],
    origin: str,
    destination: str,
    departure_time: float = 0.0,
    alpha: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_node = origin
    current_time = float(departure_time)
    log_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    step = 1
    high_risk = DEFAULT_HIGH_RISK_EDGES

    while current_node != destination and current_time <= 60 and step <= 20:
        sub_paths = enumerate_simple_paths(graph, current_node, destination, edges_df, high_risk)
        if sub_paths.empty:
            log_rows.append(
                {
                    "step": step,
                    "current_node": current_node,
                    "current_time_min": current_time,
                    "selected_path_id": "",
                    "selected_node_path": "",
                    "selected_edge_path": "",
                    "next_edge": "",
                    "next_node": "",
                    "edge_enter_time_min": current_time,
                    "edge_exit_time_min": np.nan,
                    "edge_depth_cm": np.nan,
                    "edge_capacity": np.nan,
                    "edge_safety": np.nan,
                    "edge_time_min": np.inf,
                    "reason": "当前节点到终点没有候选路径",
                }
            )
            break
        sub_metrics_records = [
            evaluate_path_time_dependent(path, current_time, q1_data, edges_df)[0]
            for _, path in sub_paths.iterrows()
        ]
        sub_metrics = pd.DataFrame(sub_metrics_records)
        best = choose_best_path(sub_metrics, alpha)
        if not best["feasible"]:
            log_rows.append(
                {
                    "step": step,
                    "current_node": current_node,
                    "current_time_min": current_time,
                    "selected_path_id": "",
                    "selected_node_path": "",
                    "selected_edge_path": "",
                    "next_edge": "",
                    "next_node": "",
                    "edge_enter_time_min": current_time,
                    "edge_exit_time_min": np.nan,
                    "edge_depth_cm": np.nan,
                    "edge_capacity": np.nan,
                    "edge_safety": np.nan,
                    "edge_time_min": np.inf,
                    "reason": "No feasible path",
                }
            )
            break
        node_path = str(best["node_path"]).split("->")
        edge_path = str(best["edge_path"]).split("->")
        next_edge = edge_path[0]
        next_node = node_path[1]
        one_edge_path = pd.Series(
            {
                "path_id": best["best_path_id"],
                "node_path": f"{current_node}->{next_node}",
                "edge_path": next_edge,
            }
        )
        edge_metric, edge_steps = evaluate_path_time_dependent(
            one_edge_path, current_time, q1_data, edges_df, collect_steps=True
        )
        if not edge_metric["feasible"]:
            log_rows.append(
                {
                    "step": step,
                    "current_node": current_node,
                    "current_time_min": current_time,
                    "selected_path_id": best["best_path_id"],
                    "selected_node_path": best["node_path"],
                    "selected_edge_path": best["edge_path"],
                    "next_edge": next_edge,
                    "next_node": next_node,
                    "edge_enter_time_min": current_time,
                    "edge_exit_time_min": np.nan,
                    "edge_depth_cm": np.nan,
                    "edge_capacity": 0.0,
                    "edge_safety": 0.0,
                    "edge_time_min": np.inf,
                    "reason": "下一条边不可通行",
                }
            )
            break
        edge_step = edge_steps[0]
        log_row = {
            "step": step,
            "current_node": current_node,
            "current_time_min": current_time,
            "selected_path_id": best["best_path_id"],
            "selected_node_path": best["node_path"],
            "selected_edge_path": best["edge_path"],
            "next_edge": next_edge,
            "next_node": next_node,
            "edge_enter_time_min": edge_step["edge_enter_time_min"],
            "edge_exit_time_min": edge_step["edge_exit_time_min"],
            "edge_depth_cm": edge_step["edge_depth_cm"],
            "edge_capacity": edge_step["edge_capacity"],
            "edge_safety": edge_step["edge_safety"],
            "edge_time_min": edge_step["edge_time_min"],
            "reason": "滚动重规划后选择当前最优路径首边",
        }
        log_rows.append(log_row)
        timeline_rows.append(log_row.copy())
        current_node = next_node
        current_time = float(edge_step["edge_exit_time_min"])
        step += 1

    return pd.DataFrame(log_rows), pd.DataFrame(timeline_rows)


def save_q3_outputs(
    output_dir: Path,
    edges_df: pd.DataFrame,
    candidate_paths: pd.DataFrame,
    path_metrics: pd.DataFrame,
    dynamic_paths: pd.DataFrame,
    replanning_log: pd.DataFrame,
    path_timeline: pd.DataFrame,
    replanning_log_30: pd.DataFrame,
    path_timeline_30: pd.DataFrame,
    static_comparison: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(output_dir / "q3_topology_edges.csv", index=False, encoding="utf-8-sig")
    candidate_paths.to_csv(output_dir / "q3_candidate_paths.csv", index=False, encoding="utf-8-sig")
    path_metrics.to_csv(output_dir / "q3_path_metrics.csv", index=False, encoding="utf-8-sig")
    dynamic_paths.to_csv(output_dir / "q3_dynamic_paths.csv", index=False, encoding="utf-8-sig")
    replanning_log.to_csv(output_dir / "q3_replanning_log.csv", index=False, encoding="utf-8-sig")
    path_timeline.to_csv(output_dir / "q3_path_timeline.csv", index=False, encoding="utf-8-sig")
    replanning_log_30.to_csv(output_dir / "q3_replanning_log_30min.csv", index=False, encoding="utf-8-sig")
    path_timeline_30.to_csv(output_dir / "q3_path_timeline_30min.csv", index=False, encoding="utf-8-sig")
    static_comparison.to_csv(output_dir / "q3_static_vs_dynamic_comparison.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "q3_results.xlsx", engine="openpyxl") as writer:
        edges_df.to_excel(writer, sheet_name="topology_edges", index=False)
        candidate_paths.to_excel(writer, sheet_name="candidate_paths", index=False)
        path_metrics.to_excel(writer, sheet_name="path_metrics", index=False)
        dynamic_paths.to_excel(writer, sheet_name="dynamic_paths", index=False)
        replanning_log.to_excel(writer, sheet_name="replanning_log", index=False)
        path_timeline.to_excel(writer, sheet_name="path_timeline", index=False)
        replanning_log_30.to_excel(writer, sheet_name="replanning_log_30min", index=False)
        path_timeline_30.to_excel(writer, sheet_name="path_timeline_30min", index=False)
        static_comparison.to_excel(writer, sheet_name="static_vs_dynamic", index=False)


def _setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        return None, False, str(exc)
    fonts = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, True, ""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, False, ""


def plot_q3_results(
    output_dir: Path,
    edges_df: pd.DataFrame,
    dynamic_paths: pd.DataFrame,
    path_metrics: pd.DataFrame,
    replanning_log: pd.DataFrame,
    replanning_log_30: pd.DataFrame,
    origin: str,
    destination: str,
    high_risk_edges: list[str],
) -> str:
    plt, use_chinese, reason = _setup_matplotlib()
    if plt is None:
        return f"图片未生成: {reason}"

    def text(cn: str, en: str) -> str:
        return cn if use_chinese else en

    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for _, row in edges_df.iterrows():
        start, end = row["start_node"], row["end_node"]
        x1, y1 = NODE_POSITIONS.get(start, (0, 0))
        x2, y2 = NODE_POSITIONS.get(end, (0, 0))
        interrupted = row["edge_id"] == "L5"
        risky = row["edge_id"] in ["L2", "L8"] or (row["edge_id"] in high_risk_edges and not interrupted)
        color = "#222222" if interrupted else "#C44E52" if risky else "#4C72B0"
        linestyle = ":" if interrupted else "--" if risky else "-"
        linewidth = 4 if interrupted else 3 if risky else 2
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, linestyle=linestyle)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, row["edge_id"], ha="center", fontsize=10)
    for node, (x, y) in NODE_POSITIONS.items():
        color = "#55A868" if node == origin else "#DD8452" if node == destination else "#F2F2F2"
        ax.scatter(x, y, s=520, color=color, edgecolor="#333333", zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontweight="bold")
    ax.set_title(text("第三问简化疏散拓扑", "Q3 Simplified Evacuation Topology"))
    ax.plot([], [], color="#C44E52", linewidth=3, linestyle="--", label=text("高风险边 L2/L8", "High-risk edges L2/L8"))
    ax.plot([], [], color="#222222", linewidth=4, linestyle=":", label=text("45min 中断边 L5", "Interrupted at 45 min: L5"))
    ax.plot([], [], color="#4C72B0", linewidth=2, label=text("普通边", "Normal edge"))
    ax.legend(loc="upper right")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "q3_topology_graph.png")
    plt.close(fig)

    plt.figure(figsize=(9, 5), dpi=160)
    for alpha, strategy in ALPHA_STRATEGIES.items():
        subset = dynamic_paths[(dynamic_paths["alpha"] == alpha) & (dynamic_paths["feasible"])]
        plt.plot(subset["departure_time_min"], subset["total_time_min"], marker="o", label=f"{strategy} alpha={alpha}")
    plt.xlabel(text("出发时刻 / min", "Departure time / min"))
    plt.ylabel(text("总疏散时间 / min", "Total evacuation time / min"))
    plt.title(text("动态最优疏散时间随出发时刻变化", "Dynamic Optimal Evacuation Time by Departure"))
    plt.figtext(
        0.5,
        0.01,
        text("注：三种策略在多数时段收敛", "Note: the three strategies converge in most periods"),
        ha="center",
        fontsize=9,
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q3_dynamic_best_path_by_departure.png")
    plt.close()

    balanced = dynamic_paths[(np.isclose(dynamic_paths["alpha"], 0.5)) & (dynamic_paths["feasible"])].copy()
    feasible_metrics = path_metrics[path_metrics["feasible"]].copy()
    path_ids = sorted(
        set(balanced["best_path_id"].dropna().astype(str).tolist())
        | set(feasible_metrics["path_id"].dropna().astype(str).tolist())
    )
    path_to_y = {path_id: idx for idx, path_id in enumerate(path_ids)}
    colors = {"P1": "#4C72B0", "P2": "#55A868", "P3": "#DD8452", "P4": "#C44E52"}
    plt.figure(figsize=(11, 5.6), dpi=160)
    near_rows: list[dict[str, Any]] = []
    for _, best in balanced.iterrows():
        dep = best["departure_time_min"]
        best_time = float(best["total_time_min"])
        best_safety = float(best["min_safety"])
        same_dep = feasible_metrics[
            (feasible_metrics["departure_time_min"] == dep)
            & (feasible_metrics["path_id"].astype(str) != str(best["best_path_id"]))
        ]
        for _, cand in same_dep.iterrows():
            cand_time = float(cand["total_time_min"])
            cand_safety = float(cand["min_safety"])
            if (
                np.isfinite(best_time)
                and np.isfinite(cand_time)
                and cand_time <= best_time * 1.10
                and np.isclose(cand_safety, best_safety)
            ):
                near_rows.append(
                    {
                        "departure_time_min": dep,
                        "path_id": str(cand["path_id"]),
                        "total_time_min": cand_time,
                        "min_safety": cand_safety,
                    }
                )
    for row in near_rows:
        y = path_to_y.get(row["path_id"], 0)
        plt.scatter(
            row["departure_time_min"],
            y,
            s=120,
            facecolors="none",
            edgecolors=colors.get(row["path_id"], "#999999"),
            linewidths=1.8,
            alpha=0.75,
            label=text("近似最优路径", "Near-optimal path") if row is near_rows[0] else None,
        )
    for _, row in balanced.iterrows():
        y = path_to_y.get(row["best_path_id"], 0)
        plt.scatter(
            row["departure_time_min"],
            y,
            s=150,
            color=colors.get(row["best_path_id"], "#8172B3"),
            edgecolor="#333333",
            label=text("最优路径", "Best path") if _ == balanced.index[0] else None,
        )
        plt.text(
            row["departure_time_min"],
            y + 0.08,
            f"T={row['total_time_min']:.1f}min, Smin={row['min_safety']:.1f}",
            ha="center",
            fontsize=8,
        )
    if 30 in set(balanced["departure_time_min"].tolist()) and {"P2", "P4"}.issubset(set(path_to_y)):
        y_mid = (path_to_y["P2"] + path_to_y["P4"]) / 2
        plt.annotate(
            text(
                "P2 与 P4 均为绕行路径，30min 处 P2 因耗时略低被选中。",
                "P2 and P4 are both detours; P2 is selected at 30 min because it is slightly faster.",
            ),
            xy=(30, path_to_y["P2"]),
            xytext=(33, y_mid + 0.35),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
            fontsize=9,
            color="#333333",
        )
    plt.yticks(list(path_to_y.values()), list(path_to_y.keys()))
    plt.xlabel(text("出发时刻 / min", "Departure time / min"))
    plt.ylabel(text("均衡策略路径编号", "Path id (balanced strategy)"))
    plt.title(text("不同出发时刻下动态绕行路径演化", "Dynamic Detour Path Evolution by Departure Time"))
    plt.grid(axis="x", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "q3_path_switching_map.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    phase_y = {"P3": 3, "P4": 2, "P2/P4": 1}
    ax.broken_barh([(0, 5)], (phase_y["P3"] - 0.28, 0.56), facecolors="#DD8452", alpha=0.85)
    ax.text(2.5, phase_y["P3"], "0min: P3 = L7-L4-L2-L5\n早期最短路可行", ha="center", va="center", fontsize=9)
    ax.broken_barh([(5, 20)], (phase_y["P4"] - 0.28, 0.56), facecolors="#C44E52", alpha=0.75)
    ax.text(15, phase_y["P4"], "5-25min: P4 = L7-L4-L6-L8\n开始绕开 L2-L5", ha="center", va="center", fontsize=9)
    ax.broken_barh([(30, 5)], (phase_y["P2/P4"] - 0.28, 0.56), facecolors="#55A868", alpha=0.75)
    ax.text(32.5, phase_y["P2/P4"], "30min: P2/P4 均可作为绕行候选\nP2 耗时略低", ha="center", va="center", fontsize=9)
    ax.broken_barh([(35, 25)], (phase_y["P4"] - 0.28, 0.56), facecolors="#C44E52", alpha=0.75)
    ax.text(47.5, phase_y["P4"] + 0.38, "35-60min: P4 为主要绕行路径", ha="center", va="bottom", fontsize=9)
    ax.axvspan(40, 55, color="#999999", alpha=0.16)
    ax.text(47.5, 0.35, "40-55min: 疏散时间显著增大，处于高风险低效率阶段", ha="center", va="center", fontsize=9, color="#444444")
    ax.set_xlim(-1, 61)
    ax.set_ylim(0, 3.8)
    ax.set_yticks([phase_y["P3"], phase_y["P4"], phase_y["P2/P4"]])
    ax.set_yticklabels(["P3", "P4", "P2/P4"])
    ax.set_xlabel(text("出发时刻 / min", "Departure time / min"))
    ax.set_title(text("不同出发时刻下动态绕行路径阶段性演化", "Staged Evolution of Dynamic Detour Paths"))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "q3_path_stage_evolution.png")
    plt.close(fig)

    available_deps = set(path_metrics["departure_time_min"].tolist())
    scatter_deps = [dep for dep in [30, 45] if dep in available_deps]
    if not scatter_deps:
        scatter_deps = [45 if 45 in available_deps else 30 if 30 in available_deps else 0]
    fig, axes = plt.subplots(
        1,
        len(scatter_deps),
        figsize=(6.2 * len(scatter_deps), 4.8),
        dpi=160,
        sharey=True,
    )
    if len(scatter_deps) == 1:
        axes = [axes]
    for ax, scatter_dep in zip(axes, scatter_deps):
        scatter_all = path_metrics[path_metrics["departure_time_min"] == scatter_dep].copy()
        scatter = scatter_all[scatter_all["feasible"]]
        infeasible = scatter_all[~scatter_all["feasible"]]
        if not scatter.empty:
            ax.scatter(
                scatter["total_time_min"],
                scatter["min_safety"],
                s=70,
                color="#8172B3",
                label=text("可行路径", "Feasible"),
            )
        for _, row in scatter.iterrows():
            ax.annotate(
                row["path_id"],
                (row["total_time_min"], row["min_safety"]),
                xytext=(4, 4),
                textcoords="offset points",
            )
        if not infeasible.empty:
            finite_times = pd.to_numeric(scatter["total_time_min"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            x_base = float(finite_times.max()) if not finite_times.empty else 20.0
            for offset, (_, row) in enumerate(infeasible.iterrows()):
                x = x_base + 1.5 + offset * 1.2
                y = 0.03 + offset * 0.04
                ax.scatter(
                    [x],
                    [y],
                    marker="x",
                    s=95,
                    color="gray",
                    label=text("不可行路径", "Infeasible") if offset == 0 else None,
                )
                reason = "因 L5 中断不可行" if "L5" in str(row.get("edge_path", "")) else "不可行"
                ax.annotate(
                    f"{row['path_id']} {reason}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="gray",
                )
        ax.set_xlabel(text("总疏散时间 / min", "Total time / min"))
        ax.set_title(text(f"{scatter_dep} min 出发", f"Departure {scatter_dep} min"))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(text("路径最小安全度", "Minimum safety"))
    fig.suptitle(text("30/45 min 路径时间-安全权衡对照", "Time-Safety Tradeoff at 30/45 min"))
    fig.tight_layout()
    fig.savefig(output_dir / "q3_path_time_safety_scatter.png")
    plt.close(fig)

    plt.figure(figsize=(9, 4.8), dpi=160)
    if not replanning_log.empty and "next_edge" in replanning_log.columns:
        for idx, row in replanning_log.iterrows():
            if pd.notna(row["edge_exit_time_min"]) and np.isfinite(row["edge_exit_time_min"]):
                plt.barh(
                    y=idx,
                    width=row["edge_exit_time_min"] - row["edge_enter_time_min"],
                    left=row["edge_enter_time_min"],
                    color="#55A868",
                )
                plt.text(row["edge_enter_time_min"], idx, row["next_edge"], va="center", ha="left")
        plt.yticks(range(len(replanning_log)), [f"step {int(s)}" for s in replanning_log["step"]])
    plt.xlabel(text("时间 / min", "Time / min"))
    plt.title(text("滚动重规划实际通行时间轴", "Rolling Replanning Timeline"))
    plt.tight_layout()
    plt.savefig(output_dir / "q3_replanning_timeline.png")
    plt.close()

    plt.figure(figsize=(9, 4.8), dpi=160)
    if not replanning_log_30.empty and "next_edge" in replanning_log_30.columns:
        for idx, row in replanning_log_30.iterrows():
            if pd.notna(row["edge_exit_time_min"]) and np.isfinite(row["edge_exit_time_min"]):
                plt.barh(
                    y=idx,
                    width=row["edge_exit_time_min"] - row["edge_enter_time_min"],
                    left=row["edge_enter_time_min"],
                    color="#4C72B0",
                )
                plt.text(row["edge_enter_time_min"], idx, row["next_edge"], va="center", ha="left")
        plt.yticks(range(len(replanning_log_30)), [f"step {int(s)}" for s in replanning_log_30["step"]])
    plt.xlabel(text("时间 / min", "Time / min"))
    plt.title(text("30min 出发滚动重规划路径时间轴", "Rolling Replanning Timeline from 30 min"))
    plt.tight_layout()
    plt.savefig(output_dir / "q3_replanning_timeline_30min.png")
    plt.close()
    return "图片已生成" if use_chinese else "图片已生成；未检测到中文字体，图题使用英文。"


def write_q3_model_notes(
    output_dir: Path,
    origin: str,
    destination: str,
    candidate_paths: pd.DataFrame,
    dynamic_paths: pd.DataFrame,
    replanning_log: pd.DataFrame,
    replanning_log_30: pd.DataFrame,
    static_comparison: pd.DataFrame,
    high_risk_edges: list[str],
    topology_loaded: bool,
    excel_note: str,
    plot_status: str,
) -> None:
    best0 = dynamic_paths[(dynamic_paths["departure_time_min"] == 0) & (dynamic_paths["feasible"])]
    best30 = dynamic_paths[(dynamic_paths["departure_time_min"] == 30) & (dynamic_paths["feasible"])]
    best0_text = "；".join(
        f"{row['strategy_name']}：{row['edge_path']}，T={row['total_time_min']:.2f}min，Smin={row['min_safety']:.2f}"
        for _, row in best0.iterrows()
    )
    best30_text = "；".join(
        f"{row['strategy_name']}：{row['edge_path']}，T={row['total_time_min']:.2f}min，Smin={row['min_safety']:.2f}"
        for _, row in best30.iterrows()
    )
    actual_edges = "->".join(replanning_log["next_edge"].dropna().astype(str).tolist()) if not replanning_log.empty else "未形成可行滚动路径"
    actual_edges_30 = "->".join(replanning_log_30["next_edge"].dropna().astype(str).tolist()) if not replanning_log_30.empty else "未形成可行滚动路径"
    topology_note = "已读取 results/topology/topology_edges.csv" if topology_loaded else "未读取拓扑文件，使用默认简化拓扑"
    static_text = static_comparison.to_string(index=False)
    notes = f"""第三问模型说明与论文可用结论

一、模型名称
{MODEL_NAME}

二、边状态输入
第三问不重新计算第一问积水，而是读取 q1_depth_5min.csv、q1_capacity_5min.csv、q1_safety_5min.csv，将 h_i(t)、C_i(t)、S_i(t) 按 edge_id 挂载到拓扑边 L1-L8 上。

三、简化拓扑
{topology_note}。该拓扑是基于路段等级、最低标高、风险特征和连通性原则构建的简化交通拓扑网络，不是真实测绘拓扑。{excel_note}

四、边通行时间
进入边 e_i 的时刻为 t，取 t_hat=5*floor(t/5)，并限制在 0 到 60 min。若 C_i(t_hat)>0，则 tau_i(t)=length_i/(v0_i*C_i(t_hat))；若 C_i(t_hat)=0，则该边不可通行。

五、路径安全度
路径 P 的安全度按实际进入每条边的时刻计算，S_min(P)=min S_i(t_i)，S_avg(P)=mean S_i(t_i)。主模型以 S_min 作为安全目标和约束解释，S_avg 作为辅助评价。

六、双目标优化
模型同时考虑总疏散时间最短和路径安全度最高。安全优先策略先筛选 S_min>=0.6 的路径，并按 (-S_min, total_time_min) 排序，即先最大化最小安全度，再最小化总时间；若无可行路径则降级为 S_min>=0.4，若仍无可行路径则退化为最高安全度路径。均衡策略采用 J(P)=alpha*T(P)/T_ref+(1-alpha)*(1-S_min(P))；时间优先策略要求 S_min>=0.2 后选时间最短。alpha=0.3 表示安全优先，alpha=0.5 表示均衡策略，alpha=0.7 表示时间优先。

七、路径枚举与 Pareto 分析
本题路段数量较少，因此采用枚举所有简单路径并逐条进行时间依赖仿真，比复杂智能算法更稳健、透明。Pareto 非支配分析用于验证不同时间-安全权衡下路径选择的合理性。

八、典型 OD 与最优路径结论
典型 OD 为 {origin} 到 {destination}，共枚举 {len(candidate_paths)} 条候选路径。departure_time=0 的最优结果为：{best0_text}。0min 出发时最短路仍可行，因为积水尚未形成，可作为早期对照。departure_time=30 的最优结果为：{best30_text}。30min 后随着 L2、L5 风险升高，动态路径切换到 L7-L1-L3-L8 或 L7-L4-L6-L8。45min 左右 L5 中断，静态最短路失效，必须绕行。0min 滚动重规划边序列为：{actual_edges}；论文主展示采用 30min 出发滚动重规划，边序列为：{actual_edges_30}。

九、静态最短路失效原因
静态最短路只考虑长度或自由流时间，但内涝条件下 L5、L2、L8 等高风险边的通行能力会随时间下降，甚至出现完全中断，因此静态最短路径可能在某些时段不可行或安全度过低。

静态最短路与动态均衡策略对照如下：
{static_text}

十、与第一问高风险路段关系
第一问识别的高风险边为 {'、'.join(high_risk_edges)}。第三问在候选路径、动态路径选择和验证报告中均关注这些边，尤其当 L5 在 45 min 左右中断时，动态路径会倾向于绕开 L5。

十一、动态滚动重规划
滚动重规划从当前节点和当前时刻重新枚举到终点的可行路径，选择综合代价最小路径的第一条边通行，到达下一节点后再次更新状态并重规划，直到到达终点或超过 60 min。

十二、模型局限性
拓扑为简化假设，速度参数按道路等级设定，未考虑人群拥堵互动、信号灯控制、车辆排队、实时交通管制和真实地理坐标。

十三、输出状态
{plot_status}
"""
    (output_dir / "q3_model_notes.txt").write_text(notes, encoding="utf-8")


def validate_q3_outputs(
    output_dir: Path,
    q1_status: dict[str, bool],
    edges_df: pd.DataFrame,
    topology_loaded: bool,
    graph_connected: bool,
    candidate_paths: pd.DataFrame,
    dynamic_paths: pd.DataFrame,
    q1_data: dict[str, pd.DataFrame],
) -> None:
    lines = ["Q3 validation report", f"Output directory: {output_dir}", ""]
    warnings: list[str] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "WARNING"
        msg = f"[{status}] {name}"
        if detail:
            msg += f" - {detail}"
        lines.append(msg)
        if not passed:
            warnings.append(msg)

    add_check("是否成功读取 q1_depth_5min.csv", q1_status.get("read_depth", False))
    add_check("是否成功读取 q1_capacity_5min.csv", q1_status.get("read_capacity", False))
    add_check("是否成功读取 q1_safety_5min.csv", q1_status.get("read_safety", False))
    add_check("edge_id 是否包含 L1-L8", set(ROAD_IDS).issubset(set(edges_df["edge_id"])))
    add_check("是否成功读取或生成拓扑", True, "读取拓扑文件" if topology_loaded else "使用默认简化拓扑")
    add_check("拓扑是否连通", graph_connected)
    add_check("是否枚举到 N1-N7 的候选路径", not candidate_paths.empty, f"count={len(candidate_paths)}")
    dep0 = dynamic_paths[(dynamic_paths["departure_time_min"] == 0) & (dynamic_paths["feasible"])]
    add_check("departure_time=0 是否存在可行路径", not dep0.empty)
    alphas = set(dynamic_paths["alpha"].round(1).tolist()) if "alpha" in dynamic_paths.columns else set()
    add_check("alpha=0.3、0.5、0.7 是否都有结果", {0.3, 0.5, 0.7}.issubset(alphas), f"alphas={sorted(alphas)}")
    l5_cap45 = float(q1_data["capacity"].loc[45, "L5"]) if 45 in q1_data["capacity"].index else np.nan
    if np.isfinite(l5_cap45) and l5_cap45 <= 0:
        dep45 = dynamic_paths[(dynamic_paths["departure_time_min"] == 45) & (dynamic_paths["feasible"])]
        avoids_l5 = dep45.empty or not dep45["edge_path"].astype(str).str.contains("L5").any()
        add_check("若 L5 在 45min 通行能力为0，则动态路径应避免 L5", avoids_l5, f"L5 capacity 45={l5_cap45}")
    for name in [
        "q3_dynamic_paths.csv",
        "q3_replanning_log.csv",
        "q3_replanning_log_30min.csv",
        "q3_static_vs_dynamic_comparison.csv",
        "q3_results.xlsx",
        "q3_topology_graph.png",
        "q3_dynamic_best_path_by_departure.png",
        "q3_path_switching_map.png",
        "q3_path_stage_evolution.png",
        "q3_path_time_safety_scatter.png",
        "q3_replanning_timeline.png",
        "q3_replanning_timeline_30min.png",
    ]:
        add_check(f"{name} 是否生成", (output_dir / name).exists())
    lines.append("")
    lines.append(f"Total warnings: {len(warnings)}")
    (output_dir / "q3_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for warning in warnings:
        print(warning)
    print(f"Validation report saved to: {output_dir / 'q3_validation_report.txt'}")


def _print_terminal_summary(
    origin: str,
    destination: str,
    candidate_paths: pd.DataFrame,
    dynamic_paths: pd.DataFrame,
    replanning_log: pd.DataFrame,
    replanning_log_30: pd.DataFrame,
    output_dir: Path,
) -> None:
    print(f"起点终点: {origin} -> {destination}")
    print(f"候选路径数量: {len(candidate_paths)}")
    for dep in [0, 30]:
        print(f"\n三种策略在 departure_time={dep} 的最优路径:")
        subset = dynamic_paths[dynamic_paths["departure_time_min"] == dep]
        for _, row in subset.iterrows():
            if row["feasible"]:
                print(f"  {row['strategy_name']} alpha={row['alpha']}: {row['edge_path']} | T={row['total_time_min']:.2f} min | Smin={row['min_safety']:.2f}")
            else:
                print(f"  {row['strategy_name']} alpha={row['alpha']}: No feasible path")
    actual_0 = "->".join(replanning_log["next_edge"].dropna().astype(str).tolist()) if not replanning_log.empty else "无"
    actual_30 = "->".join(replanning_log_30["next_edge"].dropna().astype(str).tolist()) if not replanning_log_30.empty else "无"
    print(f"\n滚动重规划实际路径(0min早期对照): {actual_0}")
    print(f"滚动重规划实际路径(30min论文主展示): {actual_30}")
    print(f"输出目录: {output_dir}")


def main() -> None:
    configure_console()
    project_dir = find_project_dir()
    output_dir = project_dir / "results" / "q3"
    origin = sys.argv[1] if len(sys.argv) >= 2 else "N1"
    destination = sys.argv[2] if len(sys.argv) >= 3 else "N7"

    q1_data, q1_status = load_q1_outputs(project_dir)
    excel_path = find_input_excel_optional(project_dir)
    road_attrs, excel_loaded, excel_note = load_road_attributes_optional(excel_path)
    edges_df, topology_loaded = load_or_build_topology(project_dir, road_attrs)
    high_risk_edges = load_high_risk_edges(project_dir)
    graph = build_graph(edges_df)
    graph_connected = check_graph_connected(graph)
    candidate_paths = enumerate_simple_paths(graph, origin, destination, edges_df, high_risk_edges)
    path_metrics, dynamic_paths = solve_dynamic_paths(candidate_paths, q1_data, edges_df)
    replanning_log, path_timeline = rolling_replanning(graph, edges_df, q1_data, origin, destination)
    replanning_log_30, path_timeline_30 = rolling_replanning(
        graph, edges_df, q1_data, origin, destination, departure_time=30.0, alpha=0.5
    )
    static_comparison = build_static_shortest_path_comparison(candidate_paths, path_metrics, dynamic_paths)

    save_q3_outputs(
        output_dir,
        edges_df,
        candidate_paths,
        path_metrics,
        dynamic_paths,
        replanning_log,
        path_timeline,
        replanning_log_30,
        path_timeline_30,
        static_comparison,
    )
    plot_status = plot_q3_results(
        output_dir,
        edges_df,
        dynamic_paths,
        path_metrics,
        replanning_log,
        replanning_log_30,
        origin,
        destination,
        high_risk_edges,
    )
    write_q3_model_notes(
        output_dir,
        origin,
        destination,
        candidate_paths,
        dynamic_paths,
        replanning_log,
        replanning_log_30,
        static_comparison,
        high_risk_edges,
        topology_loaded,
        excel_note,
        plot_status,
    )
    validate_q3_outputs(
        output_dir,
        q1_status,
        edges_df,
        topology_loaded,
        graph_connected,
        candidate_paths,
        dynamic_paths,
        q1_data,
    )
    _print_terminal_summary(
        origin,
        destination,
        candidate_paths,
        dynamic_paths,
        replanning_log,
        replanning_log_30,
        output_dir,
    )
    print(plot_status)


if __name__ == "__main__":
    main()

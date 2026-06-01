from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_NAME = "内涝韧性提升综合改造多目标优化模型"
ROAD_IDS = [f"L{i}" for i in range(1, 9)]
KEY_EDGES = ["L5", "L2", "L8"]
KEY_EDGE_WEIGHTS = {"L5": 0.50, "L2": 0.25, "L8": 0.25}
R0_DEFAULT = 0.5713
DECISION_GRID = {
    "x1_pipe_m": [0, 300, 600, 900, 1230],
    "x2_storage_count": [0, 1, 2],
    "x3_pavement_m2": [0, 3000, 6000, 9000],
    "x4_warning_count": [0, 1, 2],
}
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


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def find_input_file(base_dir: Path) -> Path:
    if len(sys.argv) >= 2 and Path(sys.argv[1]).exists():
        return Path(sys.argv[1]).resolve()
    search_dirs = [
        base_dir,
        Path.cwd(),
        Path(r"C:\Users\马翌博\Documents\New project"),
        Path(r"C:\Users\马翌博\xwechat_files"),
    ]
    matches: list[Path] = []
    for folder in search_dirs:
        if folder.exists():
            try:
                matches.extend(p for p in folder.rglob("B题附表1至6*.xlsx") if not p.name.startswith("~$"))
            except Exception:
                continue
    if not matches:
        raise FileNotFoundError("未找到 B题附表1至6*.xlsx，请把 Excel 放到项目目录，或命令行传入路径。")
    return sorted(
        set(matches),
        key=lambda p: ("(1)" in p.name, p.stat().st_mtime, str(p)),
        reverse=True,
    )[0].resolve()


def cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_sheet_name(sheet_names: list[str], number: int) -> str:
    for name in sheet_names:
        compact = name.replace(" ", "")
        if f"附表{number}" in compact:
            return name
    if len(sheet_names) >= number:
        return sheet_names[number - 1]
    raise ValueError(f"无法定位附表{number}")


def clean_table(raw: pd.DataFrame, required_keywords: list[str]) -> pd.DataFrame:
    raw = raw.replace(r"^\s*$", np.nan, regex=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    raw = raw.reset_index(drop=True)
    header_row = None
    for idx, row in raw.iterrows():
        row_text = " ".join(cell_text(v) for v in row.tolist())
        if all(keyword in row_text for keyword in required_keywords):
            header_row = idx
            break
    if header_row is None:
        raise ValueError(f"无法定位表头，关键词={required_keywords}")
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = [cell_text(v) or f"unnamed_{i}" for i, v in enumerate(raw.iloc[header_row])]
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return table


def find_col(columns: list[str], keywords: list[str]) -> str:
    for col in columns:
        if all(keyword in str(col) for keyword in keywords):
            return col
    for col in columns:
        if any(keyword in str(col) for keyword in keywords):
            return col
    raise KeyError(f"未找到列: {keywords}")


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    if text in {"", "—", "-", "nan", "None"}:
        return np.nan
    import re

    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def load_appendix_data(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(excel_path)

    road_raw = pd.read_excel(excel_path, sheet_name=find_sheet_name(xl.sheet_names, 1), header=None)
    road_table = clean_table(road_raw, ["路段编号", "长度", "排水"])
    road = pd.DataFrame(
        {
            "edge_id": road_table[find_col(list(road_table.columns), ["路段编号"])].astype(str).str.strip(),
            "length_m": pd.to_numeric(road_table[find_col(list(road_table.columns), ["长度"])], errors="coerce"),
            "road_class": road_table[find_col(list(road_table.columns), ["通行优先级"])].astype(str).str.strip(),
            "pipe_diameter_mm": pd.to_numeric(road_table[find_col(list(road_table.columns), ["管径"])], errors="coerce"),
            "drainage_capacity_lps": pd.to_numeric(road_table[find_col(list(road_table.columns), ["排水能力"])], errors="coerce"),
            "min_elevation_m": pd.to_numeric(road_table[find_col(list(road_table.columns), ["最低标高"])], errors="coerce"),
        }
    )
    road = road[road["edge_id"].str.match(r"^L\d+$", na=False)].copy()

    ind_raw = pd.read_excel(excel_path, sheet_name=find_sheet_name(xl.sheet_names, 5), header=None)
    ind_table = clean_table(ind_raw, ["一级指标", "二级指标", "权重"])
    layer_col = find_col(list(ind_table.columns), ["一级指标"])
    indicator_col = find_col(list(ind_table.columns), ["二级指标"])
    unit_col = find_col(list(ind_table.columns), ["计量单位"])
    current_col = find_col(list(ind_table.columns), ["现状"])
    ideal_col = find_col(list(ind_table.columns), ["理想"])
    weight_col = find_col(list(ind_table.columns), ["权重"])
    indicators = pd.DataFrame(
        {
            "layer": ind_table[layer_col].ffill().astype(str).str.strip(),
            "indicator": ind_table[indicator_col].astype(str).str.strip(),
            "unit": ind_table[unit_col].astype(str).str.strip(),
            "current_value": pd.to_numeric(ind_table[current_col], errors="coerce"),
            "ideal_value": pd.to_numeric(ind_table[ideal_col], errors="coerce"),
            "weight": pd.to_numeric(ind_table[weight_col], errors="coerce"),
        }
    )
    indicators = indicators[indicators["indicator"].notna() & indicators["current_value"].notna()].copy()
    indicators["weight"] = indicators["weight"] / indicators["weight"].sum()
    indicators["direction"] = indicators["indicator"].apply(infer_indicator_direction)

    meas_raw = pd.read_excel(excel_path, sheet_name=find_sheet_name(xl.sheet_names, 6), header=None)
    meas_table = clean_table(meas_raw, ["改造措施", "单位成本", "寿命"])
    measures = meas_table.copy()
    measures.columns = [str(c).strip() for c in measures.columns]
    return road, indicators, measures


def infer_indicator_direction(indicator: str) -> str:
    negative_keywords = ["时间", "半衰期", "损失", "风险", "深度", "延误", "成本", "低洼", "占比"]
    return "negative" if any(keyword in indicator for keyword in negative_keywords) else "positive"


def normalize_value(value: float, ideal: float, direction: str, unit: str, indicator: str) -> float:
    if not np.isfinite(value) or not np.isfinite(ideal):
        return 0.0
    if direction == "positive":
        score = value / ideal if ideal != 0 else 1.0 if value >= ideal else 0.0
    else:
        if ideal == 0:
            if "%" in unit or "占比" in indicator:
                score = 1.0 - value / 100.0
            else:
                score = 1.0 if value == 0 else 1.0 / (1.0 + value)
        else:
            score = ideal / value if value != 0 else 1.0
    return float(np.clip(score, 0.0, 1.0))


def compute_resilience(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in indicators.iterrows():
        score = normalize_value(
            float(row["new_value"]),
            float(row["ideal_value"]),
            str(row["direction"]),
            str(row["unit"]),
            str(row["indicator"]),
        )
        rows.append({**row.to_dict(), "normalized_score": score, "weighted_score": score * float(row["weight"])})
    detail = pd.DataFrame(rows)
    total = float(detail["weighted_score"].sum())
    layer_rows = [
        {
            "score_type": "综合韧性",
            "layer": "总体",
            "score": total,
            "weight_sum": float(detail["weight"].sum()),
            "indicator_count": len(detail),
        }
    ]
    for layer, group in detail.groupby("layer"):
        layer_rows.append(
            {
                "score_type": "层级韧性",
                "layer": layer,
                "score": float((group["normalized_score"] * group["weight"]).sum() / group["weight"].sum()),
                "weight_sum": float(group["weight"].sum()),
                "indicator_count": len(group),
            }
        )
    return detail, pd.DataFrame(layer_rows)


def read_time_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_min" not in df.columns:
        raise ValueError(f"{path.name} 缺少 time_min 列")
    df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce").astype(int)
    df = df.set_index("time_min").sort_index()
    return df[[edge for edge in ROAD_IDS if edge in df.columns]].astype(float)


def load_previous_outputs(base_dir: Path) -> dict[str, Any]:
    q1 = base_dir / "results" / "q1"
    q3 = base_dir / "results" / "q3"
    outputs = {
        "depth": read_time_table(q1 / "q1_depth_5min.csv"),
        "capacity": read_time_table(q1 / "q1_capacity_5min.csv"),
        "safety": read_time_table(q1 / "q1_safety_5min.csv"),
        "risk_summary": pd.read_csv(q1 / "q1_risk_summary.csv"),
        "q3_static_dynamic": pd.read_csv(q3 / "q3_static_vs_dynamic_comparison.csv") if (q3 / "q3_static_vs_dynamic_comparison.csv").exists() else pd.DataFrame(),
        "q3_dynamic": pd.read_csv(q3 / "q3_dynamic_paths.csv") if (q3 / "q3_dynamic_paths.csv").exists() else pd.DataFrame(),
        "topology": pd.read_csv(q3 / "q3_topology_edges.csv") if (q3 / "q3_topology_edges.csv").exists() else pd.DataFrame(),
    }
    return outputs


def base_speed_kmh(road_class: str) -> float:
    text = str(road_class)
    if "主干" in text:
        return 40.0
    if "次干" in text:
        return 30.0
    if "支路" in text:
        return 20.0
    return 30.0


def prepare_edges(road: pd.DataFrame, prev: dict[str, Any]) -> pd.DataFrame:
    edges = prev.get("topology", pd.DataFrame()).copy()
    if edges.empty:
        edges = pd.DataFrame(
            [
                {"edge_id": eid, "start_node": s, "end_node": e}
                for eid, (s, e) in DEFAULT_TOPOLOGY.items()
            ]
        )
    edges["edge_id"] = edges["edge_id"].astype(str)
    attrs = road[["edge_id", "length_m", "road_class"]].drop_duplicates("edge_id")
    edges = edges.drop(columns=[c for c in ["length_m", "road_class"] if c in edges.columns], errors="ignore")
    edges = edges.merge(attrs, on="edge_id", how="left")
    edges["length_m"] = edges.apply(lambda r: float(r["length_m"]) if pd.notna(r["length_m"]) else DEFAULT_LENGTHS.get(r["edge_id"], 400.0), axis=1)
    edges["road_class"] = edges.apply(lambda r: r["road_class"] if pd.notna(r["road_class"]) else DEFAULT_ROAD_CLASS.get(r["edge_id"], "次干路"), axis=1)
    edges["base_speed_kmh"] = edges["road_class"].apply(base_speed_kmh)
    edges["base_speed_m_per_min"] = edges["base_speed_kmh"] * 1000 / 60
    return edges


def scheme_flags(x1: float, x2: float, x3: float, x4: float) -> dict[str, int]:
    return {
        "I_pipe": int(x1 > 0),
        "I_storage": int(x2 > 0),
        "I_pavement": int(x3 > 0),
        "I_warning": int(x4 > 0),
    }


def apply_scheme_to_indicators(
    indicators: pd.DataFrame,
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    effect_factor: float = 1.0,
) -> pd.DataFrame:
    flags = scheme_flags(x1, x2, x3, x4)
    updated = indicators.copy()
    updated["new_value"] = updated["current_value"].astype(float)

    def set_indicator(name: str, value: float) -> None:
        mask = updated["indicator"].astype(str).str.contains(name, regex=False)
        if mask.any():
            updated.loc[mask, "new_value"] = value

    def current(name: str) -> float:
        mask = updated["indicator"].astype(str).str.contains(name, regex=False)
        return float(updated.loc[mask, "current_value"].iloc[0]) if mask.any() else np.nan

    drainage_old = current("排水能力达标率")
    storage_old = current("绿色调蓄容积")
    half_life_old = current("积水消退半衰期")
    response_old = current("应急响应时间")
    warning_old = current("预警覆盖率")

    if np.isfinite(drainage_old):
        improvement = effect_factor * (0.25 * flags["I_pipe"] + 0.15 * flags["I_pavement"])
        set_indicator("排水能力达标率", min(100.0, drainage_old * (1.0 + improvement)))
    if np.isfinite(storage_old):
        set_indicator("绿色调蓄容积", storage_old + effect_factor * (800.0 * x2 + 0.1 * x3))
    if np.isfinite(half_life_old):
        half_life_new = half_life_old
        half_life_new *= 1.0 - effect_factor * 0.15 * flags["I_pipe"]
        half_life_new *= 1.0 - effect_factor * 0.20 * flags["I_storage"]
        half_life_new *= 1.0 - effect_factor * 0.10 * flags["I_pavement"]
        set_indicator("积水消退半衰期", max(1.0, half_life_new))
    if np.isfinite(response_old):
        set_indicator("应急响应时间", max(1.0, response_old * (1.0 - effect_factor * 0.40 * flags["I_warning"])))
    if np.isfinite(warning_old):
        set_indicator("预警覆盖率", min(100.0, warning_old + effect_factor * 15.0 * x4))
    return updated


def map_safety_from_depth(depth: pd.DataFrame) -> pd.DataFrame:
    safety = pd.DataFrame(index=depth.index, columns=depth.columns, dtype=float)
    h = depth
    safety[h < 5] = 1.0
    safety[(h >= 5) & (h < 10)] = 0.9
    safety[(h >= 10) & (h < 20)] = 0.6
    safety[(h >= 20) & (h < 30)] = 0.2
    safety[h >= 30] = 0.0
    return safety.astype(float)


def update_water_states(
    prev: dict[str, Any],
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    rain_factor: float = 1.0,
    effect_factor: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    flags = scheme_flags(x1, x2, x3, x4)
    depth = prev["depth"].copy() * rain_factor
    capacity = prev["capacity"].copy()
    factor = 1.0
    factor *= 1.0 - effect_factor * 0.15 * flags["I_pipe"]
    factor *= 1.0 - effect_factor * 0.20 * flags["I_storage"]
    factor *= 1.0 - effect_factor * 0.10 * flags["I_pavement"]
    factor = float(np.clip(factor, 0.30, 1.20))
    cap_gain = effect_factor * (0.25 * flags["I_pipe"] + 0.15 * flags["I_pavement"])
    for edge in KEY_EDGES:
        if edge in depth.columns:
            depth[edge] = np.maximum(0.0, depth[edge] * factor)
        if edge in capacity.columns:
            capacity[edge] = np.minimum(1.0, capacity[edge] + cap_gain * (1.0 - capacity[edge]))
            if flags["I_pipe"] or flags["I_storage"]:
                capacity.loc[capacity[edge] <= 0, edge] = np.maximum(capacity.loc[capacity[edge] <= 0, edge], 0.2)
    safety = map_safety_from_depth(depth)
    return depth, capacity, safety, {"depth_factor": factor, "capacity_gain": cap_gain}


def half_life_time(series: pd.Series) -> float:
    s = series.astype(float).sort_index()
    max_depth = float(s.max())
    if max_depth <= 0:
        return 0.0
    peak_time = int(s.idxmax())
    threshold = 0.5 * max_depth
    after_peak = s[s.index >= peak_time]
    reached = after_peak[after_peak <= threshold]
    return float(reached.index[0]) if not reached.empty else 60.0


def weighted_recovery_time(depth: pd.DataFrame, recovery_factor: float = 1.0) -> tuple[float, pd.DataFrame]:
    rows = []
    total = 0.0
    for edge, weight in KEY_EDGE_WEIGHTS.items():
        raw_t_half = half_life_time(depth[edge]) if edge in depth.columns else 60.0
        t_half = float(raw_t_half * recovery_factor)
        max_depth = float(depth[edge].max()) if edge in depth.columns else np.nan
        rows.append(
            {
                "edge_id": edge,
                "weight": weight,
                "max_depth_cm": max_depth,
                "raw_half_life_min": raw_t_half,
                "recovery_factor": recovery_factor,
                "half_life_min": t_half,
            }
        )
        total += weight * t_half
    return total, pd.DataFrame(rows)


def recovery_factor_from_scheme(x1: float, x2: float, x3: float, effect_factor: float = 1.0) -> float:
    flags = scheme_flags(x1, x2, x3, 0)
    factor = 1.0
    factor *= 1.0 - effect_factor * 0.15 * flags["I_pipe"]
    factor *= 1.0 - effect_factor * 0.20 * flags["I_storage"]
    factor *= 1.0 - effect_factor * 0.10 * flags["I_pavement"]
    return float(np.clip(factor, 0.30, 1.20))


def interruption_duration(capacity: pd.DataFrame, edge: str) -> float:
    if edge not in capacity.columns:
        return 0.0
    return float((capacity[edge] <= 0).sum() * 5)


def build_graph(edges: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    graph: dict[str, list[dict[str, Any]]] = {}
    for _, row in edges.iterrows():
        edge = row["edge_id"]
        start, end = row["start_node"], row["end_node"]
        item = row.to_dict()
        graph.setdefault(start, []).append({**item, "from_node": start, "to_node": end})
        graph.setdefault(end, []).append({**item, "from_node": end, "to_node": start})
    return graph


def enumerate_paths(graph: dict[str, list[dict[str, Any]]], origin: str = "N1", destination: str = "N7") -> pd.DataFrame:
    paths = []

    def dfs(node: str, visited: set[str], node_path: list[str], edge_path: list[str]) -> None:
        if node == destination:
            paths.append({"node_path": node_path.copy(), "edge_path": edge_path.copy()})
            return
        for edge in graph.get(node, []):
            nxt = edge["to_node"]
            if nxt in visited:
                continue
            dfs(nxt, visited | {nxt}, node_path + [nxt], edge_path + [edge["edge_id"]])

    dfs(origin, {origin}, [origin], [])
    rows = []
    for idx, path in enumerate(paths, start=1):
        rows.append({"path_id": f"P{idx}", "node_path": "->".join(path["node_path"]), "edge_path": "->".join(path["edge_path"])})
    return pd.DataFrame(rows)


def state_at_time(table: pd.DataFrame, edge: str, time_min: float) -> float:
    t = int(math.floor(max(0.0, min(60.0, time_min)) / 5.0) * 5)
    if t not in table.index:
        t = int(table.index[table.index <= t].max()) if (table.index <= t).any() else int(table.index.min())
    return float(table.loc[t, edge])


def evaluate_path(
    edge_path: str,
    departure_time: float,
    edges: pd.DataFrame,
    depth: pd.DataFrame,
    capacity: pd.DataFrame,
    safety: pd.DataFrame,
) -> dict[str, Any]:
    current = float(departure_time)
    safeties: list[float] = []
    blocked: list[str] = []
    edge_rows = edges.set_index("edge_id")
    for edge in edge_path.split("->"):
        cap = state_at_time(capacity, edge, current)
        dep = state_at_time(depth, edge, current)
        saf = state_at_time(safety, edge, current)
        if cap <= 0:
            blocked.append(edge)
            return {
                "feasible": False,
                "total_time_min": np.inf,
                "min_safety": 0.0,
                "avg_safety": 0.0,
                "blocked_edges": "->".join(blocked),
                "reason": f"{edge} 不可通行",
            }
        length = float(edge_rows.loc[edge, "length_m"])
        speed = float(edge_rows.loc[edge, "base_speed_m_per_min"])
        edge_time = length / (speed * cap)
        current += edge_time
        safeties.append(saf)
    return {
        "feasible": True,
        "total_time_min": current - departure_time,
        "min_safety": float(min(safeties)) if safeties else 0.0,
        "avg_safety": float(np.mean(safeties)) if safeties else 0.0,
        "blocked_edges": "",
        "reason": "OK",
    }


def choose_dynamic_path(paths: pd.DataFrame, departure_time: float, edges: pd.DataFrame, depth: pd.DataFrame, capacity: pd.DataFrame, safety: pd.DataFrame) -> dict[str, Any]:
    evaluated = []
    for _, path in paths.iterrows():
        result = evaluate_path(path["edge_path"], departure_time, edges, depth, capacity, safety)
        evaluated.append({**path.to_dict(), **result})
    feasible = [r for r in evaluated if r["feasible"]]
    if not feasible:
        return {"path_id": "", "edge_path": "", "feasible": False, "total_time_min": np.inf, "min_safety": 0.0, "avg_safety": 0.0, "blocked_edges": ";".join(r["blocked_edges"] for r in evaluated), "reason": "No feasible path"}
    t_ref = max(r["total_time_min"] for r in feasible) or 30.0
    for r in feasible:
        r["objective"] = 0.5 * r["total_time_min"] / t_ref + 0.5 * (1.0 - r["min_safety"])
    return min(feasible, key=lambda r: (r["objective"], r["total_time_min"]))


def static_shortest_path(paths: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    edge_rows = edges.set_index("edge_id")
    rows = []
    for _, path in paths.iterrows():
        total = 0.0
        for edge in path["edge_path"].split("->"):
            total += float(edge_rows.loc[edge, "length_m"]) / float(edge_rows.loc[edge, "base_speed_m_per_min"])
        rows.append(total)
    paths = paths.copy()
    paths["freeflow_time_min"] = rows
    return paths.sort_values("freeflow_time_min").iloc[0]


def evaluate_evacuation(
    x4: float,
    paths: pd.DataFrame,
    edges: pd.DataFrame,
    depth: pd.DataFrame,
    capacity: pd.DataFrame,
    safety: pd.DataFrame,
) -> dict[str, Any]:
    rows = {}
    static_path = static_shortest_path(paths, edges)
    for scene_time in [30, 45]:
        effective_departure = scene_time - 0.40 * 12.0 if x4 > 0 else float(scene_time)
        effective_departure = max(0.0, effective_departure)
        dyn = choose_dynamic_path(paths, effective_departure, edges, depth, capacity, safety)
        sta = evaluate_path(static_path["edge_path"], effective_departure, edges, depth, capacity, safety)
        prefix = f"evac_{scene_time}"
        rows[f"{prefix}_effective_departure_min"] = effective_departure
        rows[f"{prefix}_dynamic_path"] = dyn.get("edge_path", "")
        rows[f"{prefix}_dynamic_time_min"] = dyn.get("total_time_min", np.inf)
        rows[f"{prefix}_dynamic_min_safety"] = dyn.get("min_safety", 0.0)
        rows[f"{prefix}_dynamic_feasible"] = dyn.get("feasible", False)
        rows[f"{prefix}_static_path"] = static_path["edge_path"]
        rows[f"{prefix}_static_time_min"] = sta.get("total_time_min", np.inf)
        rows[f"{prefix}_static_min_safety"] = sta.get("min_safety", 0.0)
        rows[f"{prefix}_static_feasible"] = sta.get("feasible", False)
    rows["L5_interrupted_45min"] = bool(capacity.loc[45, "L5"] <= 0) if 45 in capacity.index and "L5" in capacity.columns else False
    return rows


def enumerate_candidate_schemes(indicators: pd.DataFrame, prev: dict[str, Any], edges: pd.DataFrame, paths: pd.DataFrame, r0: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_trec, baseline_recovery = weighted_recovery_time(prev["depth"])
    baseline_interrupt = {edge: interruption_duration(prev["capacity"], edge) for edge in KEY_EDGES}
    rows = []
    scheme_id = 1
    for x1 in DECISION_GRID["x1_pipe_m"]:
        for x2 in DECISION_GRID["x2_storage_count"]:
            for x3 in DECISION_GRID["x3_pavement_m2"]:
                for x4 in DECISION_GRID["x4_warning_count"]:
                    total_cost = 1.8 * x1 + 1200 * x2 + 0.35 * x3 + 80 * x4
                    annual_cost = 1.8 * x1 / 20 + 1200 * x2 / 30 + 0.35 * x3 / 15 + 80 * x4 / 8
                    updated_indicators = apply_scheme_to_indicators(indicators, x1, x2, x3, x4)
                    _, score_df = compute_resilience(updated_indicators)
                    r_new = float(score_df.loc[score_df["layer"] == "总体", "score"].iloc[0])
                    depth_new, cap_new, safety_new, effect_info = update_water_states(prev, x1, x2, x3, x4)
                    rec_factor = recovery_factor_from_scheme(x1, x2, x3)
                    trec, _ = weighted_recovery_time(depth_new, recovery_factor=rec_factor)
                    max_reduction = []
                    duration_changes = []
                    for edge in KEY_EDGES:
                        before = float(prev["depth"][edge].max())
                        after = float(depth_new[edge].max())
                        max_reduction.append((before - after) / before if before > 0 else 0.0)
                        duration_changes.append(baseline_interrupt[edge] - interruption_duration(cap_new, edge))
                    evac = evaluate_evacuation(x4, paths, edges, depth_new, cap_new, safety_new)
                    rows.append(
                        {
                            "scheme_id": f"S{scheme_id:03d}",
                            "x1_pipe_m": x1,
                            "x2_storage_count": x2,
                            "x3_pavement_m2": x3,
                            "x4_warning_count": x4,
                            "TotalCost_wan": total_cost,
                            "AnnualCost_wan_per_year": annual_cost,
                            "R_new": r_new,
                            "Delta_R": r_new - r0,
                            "T_rec_min": trec,
                            "T_rec_reduction_min": baseline_trec - trec,
                            "recovery_factor": rec_factor,
                            "avg_key_max_depth_reduction_rate": float(np.mean(max_reduction)),
                            "avg_interruption_duration_reduction_min": float(np.mean(duration_changes)),
                            **effect_info,
                            **evac,
                        }
                    )
                    scheme_id += 1
    meta = {"baseline_T_rec": baseline_trec, "baseline_recovery": baseline_recovery, "baseline_interrupt": baseline_interrupt}
    return pd.DataFrame(rows), meta


def compute_pareto(candidates: pd.DataFrame) -> pd.DataFrame:
    is_pareto = np.ones(len(candidates), dtype=bool)
    arr = candidates[["AnnualCost_wan_per_year", "Delta_R", "T_rec_min"]].to_numpy(dtype=float)
    for i, a in enumerate(arr):
        for j, b in enumerate(arr):
            if i == j:
                continue
            dominates = (b[0] <= a[0] and b[1] >= a[1] and b[2] <= a[2]) and (b[0] < a[0] or b[1] > a[1] or b[2] < a[2])
            if dominates:
                is_pareto[i] = False
                break
    pareto = candidates[is_pareto].copy()
    return pareto.sort_values(["AnnualCost_wan_per_year", "T_rec_min", "Delta_R"], ascending=[True, True, False]).reset_index(drop=True)


def minmax_score(series: pd.Series, positive: bool) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if np.isclose(vmax, vmin):
        return pd.Series(1.0, index=series.index)
    score = (vals - vmin) / (vmax - vmin)
    return score if positive else 1.0 - score


def compute_topsis(pareto: pd.DataFrame) -> pd.DataFrame:
    if pareto.empty:
        return pareto.copy()
    df = pareto.copy()
    benefit = pd.DataFrame(
        {
            "cost_score": minmax_score(df["AnnualCost_wan_per_year"], positive=False),
            "resilience_score": minmax_score(df["Delta_R"], positive=True),
            "recovery_score": minmax_score(df["T_rec_min"], positive=False),
        }
    )
    weights = np.array([0.30, 0.40, 0.30])
    weighted = benefit.to_numpy(dtype=float) * weights
    ideal = weights
    anti = np.zeros(3)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
    df["cost_score"] = benefit["cost_score"].values
    df["resilience_score"] = benefit["resilience_score"].values
    df["recovery_score"] = benefit["recovery_score"].values
    df["TOPSIS_closeness"] = d_neg / (d_pos + d_neg + 1e-12)
    df["TOPSIS_rank"] = df["TOPSIS_closeness"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("TOPSIS_rank").reset_index(drop=True)


def topsis_score_from_ranges(row: pd.Series, ranges: dict[str, tuple[float, float]]) -> float:
    def score(value: float, key: str, positive: bool) -> float:
        lo, hi = ranges[key]
        if np.isclose(hi, lo):
            return 1.0
        s = (value - lo) / (hi - lo)
        if not positive:
            s = 1.0 - s
        return float(np.clip(s, 0.0, 1.0))

    benefit = np.array(
        [
            score(float(row["AnnualCost_wan_per_year"]), "AnnualCost_wan_per_year", False),
            score(float(row["Delta_R"]), "Delta_R", True),
            score(float(row["T_rec_min"]), "T_rec_min", False),
        ]
    )
    weights = np.array([0.30, 0.40, 0.30])
    weighted = benefit * weights
    d_pos = np.sqrt(((weighted - weights) ** 2).sum())
    d_neg = np.sqrt((weighted**2).sum())
    return float(d_neg / (d_pos + d_neg + 1e-12))


def robustness_analysis(
    top_schemes: pd.DataFrame,
    indicators: pd.DataFrame,
    prev: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
    r0: float,
) -> pd.DataFrame:
    rows = []
    for _, scheme in top_schemes.iterrows():
        for rain_factor in [0.9, 1.0, 1.1, 1.2]:
            for effect_factor in [0.8, 1.0, 1.2]:
                for cost_factor in [0.9, 1.0, 1.1]:
                    x1, x2, x3, x4 = (scheme["x1_pipe_m"], scheme["x2_storage_count"], scheme["x3_pavement_m2"], scheme["x4_warning_count"])
                    annual = float(scheme["AnnualCost_wan_per_year"]) * cost_factor
                    updated = apply_scheme_to_indicators(indicators, x1, x2, x3, x4, effect_factor=effect_factor)
                    _, score_df = compute_resilience(updated)
                    r_new = float(score_df.loc[score_df["layer"] == "总体", "score"].iloc[0])
                    depth, _, _, _ = update_water_states(prev, x1, x2, x3, x4, rain_factor=rain_factor, effect_factor=effect_factor)
                    trec, _ = weighted_recovery_time(
                        depth,
                        recovery_factor=recovery_factor_from_scheme(x1, x2, x3, effect_factor=effect_factor),
                    )
                    temp = pd.Series({"AnnualCost_wan_per_year": annual, "Delta_R": r_new - r0, "T_rec_min": trec})
                    rows.append(
                        {
                            "scheme_id": scheme["scheme_id"],
                            "rain_factor": rain_factor,
                            "effect_factor": effect_factor,
                            "cost_factor": cost_factor,
                            "AnnualCost_wan_per_year": annual,
                            "R_new": r_new,
                            "Delta_R": r_new - r0,
                            "T_rec_min": trec,
                            "TOPSIS_score": topsis_score_from_ranges(temp, ranges),
                        }
                    )
    robust = pd.DataFrame(rows)
    summary = robust.groupby("scheme_id")["TOPSIS_score"].agg(["mean", "std"]).reset_index()
    summary["RobustScore"] = summary["mean"] - 0.5 * summary["std"].fillna(0)
    robust = robust.merge(summary[["scheme_id", "RobustScore"]], on="scheme_id", how="left")
    return robust


def build_output_tables(
    recommended: pd.Series,
    indicators: pd.DataFrame,
    prev: dict[str, Any],
    edges: pd.DataFrame,
    paths: pd.DataFrame,
    r0: float,
) -> dict[str, pd.DataFrame]:
    x1, x2, x3, x4 = (recommended["x1_pipe_m"], recommended["x2_storage_count"], recommended["x3_pavement_m2"], recommended["x4_warning_count"])
    depth_new, cap_new, safety_new, _ = update_water_states(prev, x1, x2, x3, x4)
    rec_factor = recovery_factor_from_scheme(x1, x2, x3)
    updated_ind = apply_scheme_to_indicators(indicators, x1, x2, x3, x4)
    detail_after, score_after = compute_resilience(updated_ind)
    base_ind = indicators.copy()
    base_ind["new_value"] = base_ind["current_value"]
    detail_before, score_before = compute_resilience(base_ind)

    depth_rows = []
    rec_rows = []
    for edge in KEY_EDGES:
        before_max = float(prev["depth"][edge].max())
        after_max = float(depth_new[edge].max())
        depth_rows.append(
            {
                "edge_id": edge,
                "before_max_depth_cm": before_max,
                "after_max_depth_cm": after_max,
                "reduction_cm": before_max - after_max,
                "reduction_rate": (before_max - after_max) / before_max if before_max > 0 else 0.0,
                "before_interruption_duration_min": interruption_duration(prev["capacity"], edge),
                "after_interruption_duration_min": interruption_duration(cap_new, edge),
            }
        )
        rec_rows.append(
            {
                "edge_id": edge,
                "before_half_life_min": half_life_time(prev["depth"][edge]),
                "after_raw_half_life_min": half_life_time(depth_new[edge]),
                "recovery_factor": rec_factor,
                "after_half_life_min": half_life_time(depth_new[edge]) * rec_factor,
            }
        )

    resilience = detail_before[["layer", "indicator", "unit", "current_value", "ideal_value", "weight", "normalized_score"]].rename(columns={"normalized_score": "before_score"})
    resilience = resilience.merge(
        detail_after[["indicator", "new_value", "normalized_score"]].rename(columns={"normalized_score": "after_score"}),
        on="indicator",
        how="left",
    )
    resilience["score_improvement"] = resilience["after_score"] - resilience["before_score"]

    evac_rows = []
    base_evac = evaluate_evacuation(0, paths, edges, prev["depth"], prev["capacity"], prev["safety"])
    new_evac = evaluate_evacuation(x4, paths, edges, depth_new, cap_new, safety_new)
    for scene in [30, 45]:
        for label, data in [("before", base_evac), ("after", new_evac)]:
            prefix = f"evac_{scene}"
            evac_rows.append(
                {
                    "scene_time_min": scene,
                    "stage": label,
                    "effective_departure_min": data[f"{prefix}_effective_departure_min"],
                    "dynamic_path": data[f"{prefix}_dynamic_path"],
                    "dynamic_time_min": data[f"{prefix}_dynamic_time_min"],
                    "dynamic_min_safety": data[f"{prefix}_dynamic_min_safety"],
                    "dynamic_feasible": data[f"{prefix}_dynamic_feasible"],
                    "static_path": data[f"{prefix}_static_path"],
                    "static_time_min": data[f"{prefix}_static_time_min"],
                    "static_min_safety": data[f"{prefix}_static_min_safety"],
                    "static_feasible": data[f"{prefix}_static_feasible"],
                    "L5_interrupted_45min": data["L5_interrupted_45min"],
                }
            )

    scenario = pd.DataFrame(
        [
            {
                "item": "recommended_scheme",
                "value": recommended["scheme_id"],
                "description": "TOPSIS 折中推荐方案",
            },
            {"item": "baseline_R0", "value": r0, "description": "第二问综合韧性基准"},
            {"item": "recommended_R_new", "value": recommended["R_new"], "description": "推荐方案改造后综合韧性"},
            {"item": "recommended_Delta_R", "value": recommended["Delta_R"], "description": "韧性提升量"},
            {"item": "recommended_AnnualCost", "value": recommended["AnnualCost_wan_per_year"], "description": "年均成本，万元/年"},
            {"item": "recommended_T_rec", "value": recommended["T_rec_min"], "description": "重点路段加权消退时间"},
        ]
    )

    return {
        "depth_reduction": pd.DataFrame(depth_rows),
        "recovery_time": pd.DataFrame(rec_rows),
        "resilience_improvement": resilience,
        "evacuation_effect": pd.DataFrame(evac_rows),
        "scenario_summary": scenario,
        "score_before": score_before,
        "score_after": score_after,
    }


def save_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_map = {
        "candidate_schemes": "q4_candidate_schemes.csv",
        "pareto": "q4_pareto_solutions.csv",
        "topsis": "q4_topsis_ranking.csv",
        "scenario_summary": "q4_scenario_summary.csv",
        "depth_reduction": "q4_depth_reduction.csv",
        "resilience_improvement": "q4_resilience_improvement.csv",
        "recovery_time": "q4_recovery_time_comparison.csv",
        "evacuation_effect": "q4_evacuation_effect.csv",
        "robustness": "q4_robustness_analysis.csv",
    }
    for key, filename in csv_map.items():
        tables[key].to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "q4_results.xlsx") as writer:
        for key, df in tables.items():
            safe = key[:31]
            df.to_excel(writer, sheet_name=safe, index=False)


def setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        return None, False, str(exc)
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, True, ""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, False, ""


def plot_results(output_dir: Path, tables: dict[str, pd.DataFrame]) -> str:
    plt, chinese, reason = setup_matplotlib()
    if plt is None:
        return f"图片未生成：{reason}"
    candidates = tables["candidate_schemes"]
    pareto = tables["pareto"]
    topsis = tables["topsis"]
    recommended = topsis.iloc[0]
    depth = tables["depth_reduction"]
    resilience = tables["resilience_improvement"]
    recovery = tables["recovery_time"]
    robust = tables["robustness"]

    plt.figure(figsize=(8, 5), dpi=160)
    plt.scatter(candidates["AnnualCost_wan_per_year"], candidates["Delta_R"], c=candidates["T_rec_min"], cmap="viridis", alpha=0.45, label="候选方案")
    plt.scatter(pareto["AnnualCost_wan_per_year"], pareto["Delta_R"], color="#C44E52", s=55, label="Pareto方案")
    plt.scatter([recommended["AnnualCost_wan_per_year"]], [recommended["Delta_R"]], color="#DD8452", marker="*", s=180, label="推荐方案")
    plt.xlabel("年均成本 / 万元年")
    plt.ylabel("韧性提升 Delta_R")
    plt.title("Pareto 前沿：成本-韧性-消退时间")
    plt.colorbar(label="T_rec / min")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q4_pareto_front.png")
    plt.close()

    top10 = topsis.head(10).copy()
    plt.figure(figsize=(9, 5), dpi=160)
    plt.bar(top10["scheme_id"], top10["TOPSIS_closeness"], color="#4C72B0")
    plt.xlabel("方案编号")
    plt.ylabel("TOPSIS贴近度")
    plt.title("Pareto方案 TOPSIS 综合排序")
    plt.tight_layout()
    plt.savefig(output_dir / "q4_topsis_ranking_bar.png")
    plt.close()

    score_before = tables["score_before"][["layer", "score"]].rename(columns={"score": "before"})
    score_after = tables["score_after"][["layer", "score"]].rename(columns={"score": "after"})
    score = score_before.merge(score_after, on="layer")
    x = np.arange(len(score))
    plt.figure(figsize=(8, 5), dpi=160)
    plt.bar(x - 0.18, score["before"], width=0.36, label="改造前")
    plt.bar(x + 0.18, score["after"], width=0.36, label="改造后")
    plt.xticks(x, score["layer"])
    plt.ylim(0, 1)
    plt.ylabel("韧性得分")
    plt.title("综合韧性与三层能力改造前后对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q4_resilience_before_after.png")
    plt.close()

    x = np.arange(len(depth))
    plt.figure(figsize=(8, 5), dpi=160)
    plt.bar(x - 0.18, depth["before_max_depth_cm"], width=0.36, label="改造前")
    plt.bar(x + 0.18, depth["after_max_depth_cm"], width=0.36, label="改造后")
    plt.xticks(x, depth["edge_id"])
    plt.ylabel("最大积水深度 / cm")
    plt.title("重点路段最大积水深度削减")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q4_key_segments_depth_before_after.png")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=160)
    plt.bar(x - 0.18, recovery["before_half_life_min"], width=0.36, label="改造前")
    plt.bar(x + 0.18, recovery["after_half_life_min"], width=0.36, label="改造后")
    plt.xticks(x, recovery["edge_id"])
    plt.ylabel("半衰期 / min")
    plt.title("重点路段消退时间对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q4_recovery_time_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=160)
    plt.scatter(candidates["TotalCost_wan"], candidates["R_new"], alpha=0.5, label="候选方案")
    plt.scatter([recommended["TotalCost_wan"]], [recommended["R_new"]], marker="*", s=180, color="#C44E52", label="推荐方案")
    plt.xlabel("总建设成本 / 万元")
    plt.ylabel("改造后综合韧性")
    plt.title("成本-韧性散点关系")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q4_cost_resilience_scatter.png")
    plt.close()

    plt.figure(figsize=(9, 5), dpi=160)
    robust.boxplot(column="TOPSIS_score", by="scheme_id", ax=plt.gca())
    plt.suptitle("")
    plt.title("TOP 方案鲁棒性扰动分布")
    plt.xlabel("方案编号")
    plt.ylabel("扰动下 TOPSIS_score")
    plt.tight_layout()
    plt.savefig(output_dir / "q4_robustness_boxplot.png")
    plt.close()

    comp = pd.Series(
        {
            "管网扩径": 1.8 * recommended["x1_pipe_m"],
            "新建调蓄池": 1200 * recommended["x2_storage_count"],
            "透水路面": 0.35 * recommended["x3_pavement_m2"],
            "智慧预警": 80 * recommended["x4_warning_count"],
        }
    )
    plt.figure(figsize=(8, 5), dpi=160)
    plt.bar(comp.index, comp.values, color=["#4C72B0", "#55A868", "#8172B3", "#DD8452"])
    plt.ylabel("建设成本 / 万元")
    plt.title("推荐方案成本组成")
    plt.tight_layout()
    plt.savefig(output_dir / "q4_recommended_scheme_composition.png")
    plt.close()
    return "图片已生成" if chinese else "图片已生成；未检测到中文字体时部分环境可能以默认字体显示。"


def write_model_notes(output_dir: Path, excel_path: Path, tables: dict[str, pd.DataFrame], plot_status: str) -> None:
    rec = tables["topsis"].iloc[0]
    top5 = tables["topsis"].head(5)[["scheme_id", "AnnualCost_wan_per_year", "Delta_R", "T_rec_min", "TOPSIS_closeness"]]
    text = f"""# 第四问模型说明

## 模型名称
{MODEL_NAME}

## 数据来源
本问读取 Excel 文件：{excel_path}
其中附表6给出了四类韧性提升措施的真实成本、效果和寿命：管网扩径、新建调蓄池、透水路面改造、智慧预警系统。前三问结果用于识别重点治理对象和评估疏散效果，重点路段为 L5、L2、L8。

## 建模路线
第四问采用“离散组合枚举 + Pareto 非支配排序 + TOPSIS 折中决策”的多目标优化路线。决策变量为 x1 管网扩径长度、x2 调蓄池数量、x3 透水路面面积、x4 智慧预警系统数量。目标包括：年均成本最小、韧性提升最大、重点路段加权消退时间最短。

## 韧性与积水效果评估
综合韧性基准来自第二问，R0={R0_DEFAULT:.4f}。措施对附表5中的排水能力达标率、绿色调蓄容积、积水消退半衰期、应急响应时间和预警覆盖率进行修正，并按第二问归一化与权重重新计算 R_new。对第一问输出的 L5、L2、L8 积水、通行能力和安全度进行改造后更新，并重新评估 30min 和 45min 疏散效果。

消退时间指标以重点路段 L5、L2、L8 的半衰期为基础。由于题目给出的积水过程只到 60min，部分路段在观测期内尚未降至峰值一半，因此以 60min 作为原始半衰期上限，并进一步根据附表6中“消退时间缩短”效果计算改造后的等效半衰期，用于多目标优化中的 T_rec。

## 推荐方案
推荐方案为 {rec['scheme_id']}：
- 管网扩径 {rec['x1_pipe_m']} m；
- 新建调蓄池 {rec['x2_storage_count']} 座；
- 透水路面改造 {rec['x3_pavement_m2']} m²；
- 智慧预警系统 {rec['x4_warning_count']} 套。

该方案总建设成本为 {rec['TotalCost_wan']:.2f} 万元，年均成本为 {rec['AnnualCost_wan_per_year']:.2f} 万元/年，改造后综合韧性 R_new={rec['R_new']:.4f}，韧性提升 Delta_R={rec['Delta_R']:.4f}，重点路段加权消退时间 T_rec={rec['T_rec_min']:.2f} min。

推荐方案不是单纯成本最低方案，而是在成本、韧性提升和消退时间缩短之间的折中最优方案。TOPSIS 排名前五如下：
{top5.to_string(index=False)}

## 与前三问衔接
第一问和第三问表明 L5、L2、L8 是内涝与疏散薄弱环节，因此第四问把这些路段作为积水削减和通行恢复的重点对象。第二问给出系统韧性基准，第四问进一步把工程措施转化为指标提升量，为“韧性提升方案设计”提供可落地的数量化依据。

## 鲁棒性
对推荐方案和 TOP 3 Pareto 方案，在降雨强度、措施效果和成本参数扰动下进行鲁棒性分析，计算扰动下 TOPSIS_score 和 RobustScore。若推荐方案在扰动下仍保持较高 RobustScore，则说明其对成本和效果不确定性具有较强稳定性。

## 局限性
本问使用题给附表6的线性/乘性效果估计措施收益，未进一步进行详细工程水力校核；疏散效果基于第三问简化拓扑和时间依赖边权计算，未考虑施工期交通影响、人群拥堵互动和更精细的管网节点调度。

{plot_status}
"""
    (output_dir / "q4_model_notes.txt").write_text(text, encoding="utf-8-sig")


def validate_outputs(output_dir: Path, candidates: pd.DataFrame, pareto: pd.DataFrame, topsis: pd.DataFrame, r0: float) -> None:
    lines = ["Q4 validation report", f"Output directory: {output_dir}", ""]
    warnings = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "WARNING"
        msg = f"[{status}] {name}"
        if detail:
            msg += f" - {detail}"
        lines.append(msg)
        if not passed:
            warnings.append(msg)

    check("候选方案数量是否为180", len(candidates) == 180, f"count={len(candidates)}")
    check("Pareto方案是否非空", not pareto.empty, f"count={len(pareto)}")
    check("TOPSIS推荐方案是否存在", not topsis.empty)
    check("综合韧性基准是否约为0.5713", abs(r0 - 0.5713) < 0.005, f"R0={r0:.4f}")
    if not topsis.empty:
        rec = topsis.iloc[0]
        check("推荐方案韧性提升是否为正", rec["Delta_R"] > 0, f"Delta_R={rec['Delta_R']:.4f}")
        check("推荐方案年均成本是否为正", rec["AnnualCost_wan_per_year"] > 0, f"AnnualCost={rec['AnnualCost_wan_per_year']:.2f}")
    required = [
        "q4_candidate_schemes.csv",
        "q4_pareto_solutions.csv",
        "q4_topsis_ranking.csv",
        "q4_scenario_summary.csv",
        "q4_depth_reduction.csv",
        "q4_resilience_improvement.csv",
        "q4_recovery_time_comparison.csv",
        "q4_evacuation_effect.csv",
        "q4_robustness_analysis.csv",
        "q4_results.xlsx",
        "q4_model_notes.txt",
        "q4_pareto_front.png",
        "q4_topsis_ranking_bar.png",
        "q4_resilience_before_after.png",
        "q4_key_segments_depth_before_after.png",
        "q4_recovery_time_comparison.png",
        "q4_cost_resilience_scatter.png",
        "q4_robustness_boxplot.png",
        "q4_recommended_scheme_composition.png",
    ]
    for filename in required:
        check(f"{filename} 是否生成", (output_dir / filename).exists())
    lines.append("")
    lines.append(f"Total warnings: {len(warnings)}")
    (output_dir / "q4_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    for warning in warnings:
        print(warning)


def main() -> None:
    configure_console()
    base = project_dir()
    output_dir = base / "results" / "q4"
    excel_path = find_input_file(base)
    road, indicators, measures = load_appendix_data(excel_path)
    prev = load_previous_outputs(base)
    edges = prepare_edges(road, prev)
    paths = enumerate_paths(build_graph(edges))

    base_ind = indicators.copy()
    base_ind["new_value"] = base_ind["current_value"]
    _, base_score = compute_resilience(base_ind)
    r0 = float(base_score.loc[base_score["layer"] == "总体", "score"].iloc[0])
    if (base / "results" / "q2" / "q2_resilience_score.csv").exists():
        q2_score = pd.read_csv(base / "results" / "q2" / "q2_resilience_score.csv")
        q2_total = q2_score[q2_score["layer"] == "总体"]
        if not q2_total.empty:
            r0 = float(q2_total["score"].iloc[0])

    candidates, meta = enumerate_candidate_schemes(indicators, prev, edges, paths, r0)
    pareto = compute_pareto(candidates)
    topsis = compute_topsis(pareto)
    recommended = topsis.iloc[0]
    top_for_robust = topsis.head(3).copy()
    if recommended["scheme_id"] not in set(top_for_robust["scheme_id"]):
        top_for_robust = pd.concat([top_for_robust, recommended.to_frame().T], ignore_index=True)
    ranges = {
        "AnnualCost_wan_per_year": (float(candidates["AnnualCost_wan_per_year"].min()), float(candidates["AnnualCost_wan_per_year"].max())),
        "Delta_R": (float(candidates["Delta_R"].min()), float(candidates["Delta_R"].max())),
        "T_rec_min": (float(candidates["T_rec_min"].min()), float(candidates["T_rec_min"].max())),
    }
    robustness = robustness_analysis(top_for_robust, indicators, prev, ranges, r0)
    detail_tables = build_output_tables(recommended, indicators, prev, edges, paths, r0)

    tables = {
        "candidate_schemes": candidates,
        "pareto": pareto,
        "topsis": topsis,
        "scenario_summary": detail_tables["scenario_summary"],
        "depth_reduction": detail_tables["depth_reduction"],
        "resilience_improvement": detail_tables["resilience_improvement"],
        "recovery_time": detail_tables["recovery_time"],
        "evacuation_effect": detail_tables["evacuation_effect"],
        "robustness": robustness,
        "score_before": detail_tables["score_before"],
        "score_after": detail_tables["score_after"],
        "appendix6_measures": measures,
        "topology_edges": edges,
        "candidate_paths": paths,
    }
    save_outputs(output_dir, tables)
    plot_status = plot_results(output_dir, tables)
    write_model_notes(output_dir, excel_path, tables, plot_status)
    validate_outputs(output_dir, candidates, pareto, topsis, r0)

    rec = topsis.iloc[0]
    print(f"输入文件: {excel_path}")
    print(f"输出目录: {output_dir}")
    print(f"候选方案数量: {len(candidates)}")
    print(f"Pareto方案数量: {len(pareto)}")
    print(f"推荐方案: {rec['scheme_id']} | x1={rec['x1_pipe_m']}m, x2={rec['x2_storage_count']}座, x3={rec['x3_pavement_m2']}m2, x4={rec['x4_warning_count']}套")
    print(f"年均成本: {rec['AnnualCost_wan_per_year']:.2f} 万元/年")
    print(f"R_new={rec['R_new']:.4f}, Delta_R={rec['Delta_R']:.4f}, T_rec={rec['T_rec_min']:.2f} min")
    print(f"TOPSIS贴近度: {rec['TOPSIS_closeness']:.4f}")
    print(plot_status)


if __name__ == "__main__":
    main()

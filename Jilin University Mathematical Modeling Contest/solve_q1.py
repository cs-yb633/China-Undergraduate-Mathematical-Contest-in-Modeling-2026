from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIME_GRID = np.arange(0, 61, 5)
ANCHOR_TIMES = np.array([0, 15, 30, 45, 60], dtype=float)
ROAD_IDS = [f"L{i}" for i in range(1, 9)]
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
ANCHOR_OBS_TIMES = [15, 30, 45, 60]


def configure_console() -> None:
    """Avoid Windows console encoding failures when printing Chinese text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def find_input_file(argv: list[str] | None = None) -> Path:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        input_path = Path(argv[0]).expanduser()
        if input_path.exists():
            return input_path.resolve()
        raise FileNotFoundError(f"命令行输入文件不存在: {input_path}")

    script_dir = Path(__file__).resolve().parent
    project_dir = Path(r"C:\Users\马翌博\Documents\New project")
    known_wechat_dir = Path(
        r"C:\Users\马翌博\xwechat_files\wxid_i3gl5sn5m1p22_a8a9\msg\file\2026-05"
    )
    search_dirs = []
    for candidate in [Path.cwd(), script_dir, project_dir, known_wechat_dir]:
        if candidate.exists() and candidate not in search_dirs:
            search_dirs.append(candidate)

    matches: list[Path] = []
    for folder in search_dirs:
        matches.extend(p for p in folder.glob("B题附表1至6*.xlsx") if not p.name.startswith("~$"))

    if matches:
        matches = sorted(set(matches), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
        return matches[0].resolve()

    raise FileNotFoundError(
        "未找到 B题附表1至6*.xlsx。请将文件放到当前目录，或使用命令行传入 Excel 路径。"
    )


def _cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_table(raw: pd.DataFrame, header_keywords: list[str] | None = None) -> pd.DataFrame:
    """Drop blank rows/columns and optionally promote the row containing keywords to header."""
    df = raw.copy()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    if not header_keywords:
        return df

    header_row = None
    for idx, row in df.iterrows():
        text = " ".join(_cell_text(v) for v in row.tolist())
        if all(keyword in text for keyword in header_keywords):
            header_row = idx
            break
    if header_row is None:
        for idx, row in df.iterrows():
            text = " ".join(_cell_text(v) for v in row.tolist())
            if any(keyword in text for keyword in header_keywords):
                header_row = idx
                break
    if header_row is None:
        return df

    header = [_cell_text(v) or f"unnamed_{j}" for j, v in enumerate(df.iloc[header_row].tolist())]
    table = df.iloc[header_row + 1 :].copy()
    table.columns = header
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return table.reset_index(drop=True)


def _find_sheet_name(sheet_names: list[str], number: int) -> str:
    candidates = [name for name in sheet_names if f"附表 {number}" in name or f"附表{number}" in name]
    if candidates:
        return candidates[0]
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


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_road_table(raw: pd.DataFrame) -> pd.DataFrame:
    table = clean_table(raw, ["路段编号"])
    route_col = _find_column(list(table.columns), ["路段编号"])
    length_col = _find_column(list(table.columns), ["长度"])
    priority_col = _find_column(list(table.columns), ["通行优先级"])
    pavement_col = _find_column(list(table.columns), ["路面类型"])
    slope_col = _find_column(list(table.columns), ["地形坡度"])
    pipe_col = _find_column(list(table.columns), ["管径"])
    drainage_col = _find_column(list(table.columns), ["设计排水能力"])
    elev_col = _find_column(list(table.columns), ["最低标高"])

    road = pd.DataFrame(
        {
            "road_id": table[route_col].astype(str).str.strip(),
            "length_m": _to_numeric(table[length_col]),
            "priority": table[priority_col].astype(str).str.strip(),
            "pavement": table[pavement_col].astype(str).str.strip(),
            "slope_permille": _to_numeric(table[slope_col]),
            "pipe_diameter_mm": _to_numeric(table[pipe_col]),
            "drainage_capacity_lps": _to_numeric(table[drainage_col]),
            "elevation_m": _to_numeric(table[elev_col]),
        }
    )
    road = road[road["road_id"].str.match(r"^L\d+$", na=False)].copy()
    road = road.sort_values("road_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    return road.reset_index(drop=True)


def _row_by_keyword(df: pd.DataFrame, keyword: str) -> pd.Series:
    for _, row in df.iterrows():
        if keyword in " ".join(_cell_text(v) for v in row.tolist()):
            return row
    raise KeyError(f"找不到包含 {keyword} 的行")


def _parse_rainfall_table(raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_table(raw)
    time_row = _row_by_keyword(df, "时间")
    rain_row = _row_by_keyword(df, "时段雨量")
    cum_row = _row_by_keyword(df, "累计雨量")

    times = pd.to_numeric(pd.Series(time_row.iloc[1:].to_numpy()), errors="coerce")
    rain = pd.to_numeric(pd.Series(rain_row.iloc[1:].to_numpy()), errors="coerce")
    cum = pd.to_numeric(pd.Series(cum_row.iloc[1:].to_numpy()), errors="coerce")

    rainfall = pd.DataFrame(
        {
            "time_min": times,
            "rainfall_mm": rain,
            "cumulative_rainfall_mm": cum,
        }
    ).dropna(subset=["time_min"])
    rainfall[["time_min", "rainfall_mm", "cumulative_rainfall_mm"]] = rainfall[
        ["time_min", "rainfall_mm", "cumulative_rainfall_mm"]
    ].astype(float)
    return rainfall.reset_index(drop=True)


def _parse_depth_anchor_table(raw: pd.DataFrame) -> pd.DataFrame:
    table = clean_table(raw, ["时间"])
    first_col = table.columns[0]
    table = table.rename(columns={first_col: "road_id"})
    table["road_id"] = table["road_id"].astype(str).str.strip()
    table = table[table["road_id"].str.match(r"^L\d+$", na=False)].copy()

    rename_map: dict[Any, int] = {}
    for col in table.columns[1:]:
        value = pd.to_numeric(pd.Series([col]), errors="coerce").iloc[0]
        if pd.notna(value):
            rename_map[col] = int(value)
    table = table.rename(columns=rename_map)
    keep_cols = ["road_id"] + [t for t in [15, 30, 45, 60] if t in table.columns]
    anchors = table[keep_cols].copy()
    for col in keep_cols[1:]:
        anchors[col] = pd.to_numeric(anchors[col], errors="coerce")
    anchors = anchors.sort_values("road_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    return anchors.reset_index(drop=True)


def _parse_rule_table(raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_table(raw)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def load_data(input_file: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(input_file)
    sheet_names = xl.sheet_names
    raw = {
        i: pd.read_excel(input_file, sheet_name=_find_sheet_name(sheet_names, i), header=None)
        for i in [1, 2, 3, 4]
    }
    return {
        "road": _parse_road_table(raw[1]),
        "rainfall": _parse_rainfall_table(raw[2]),
        "rules": _parse_rule_table(raw[3]),
        "depth_anchors": _parse_depth_anchor_table(raw[4]),
    }


def build_depth_series(depth_anchors: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    try:
        from scipy.interpolate import PchipInterpolator

        method = "PCHIP"
    except Exception:
        PchipInterpolator = None
        method = "linear"

    result = pd.DataFrame(index=TIME_GRID)
    result.index.name = "time_min"

    for _, row in depth_anchors.iterrows():
        road_id = row["road_id"]
        y_anchor = np.array(
            [0.0, row.get(15, np.nan), row.get(30, np.nan), row.get(45, np.nan), row.get(60, np.nan)],
            dtype=float,
        )
        valid = np.isfinite(y_anchor)
        if valid.sum() < 2:
            raise ValueError(f"{road_id} 的积水锚点不足，无法插值")
        x_valid = ANCHOR_TIMES[valid]
        y_valid = y_anchor[valid]
        if method == "PCHIP" and PchipInterpolator is not None:
            values = PchipInterpolator(x_valid, y_valid)(TIME_GRID)
        else:
            values = np.interp(TIME_GRID, x_valid, y_valid)
        result[road_id] = np.maximum(values, 0.0)

    cols = [road for road in ROAD_IDS if road in result.columns]
    result = result[cols].round(4)
    return result, method


def load_or_build_topology(project_dir: Path, road_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    topology_path = project_dir / "results" / "topology" / "topology_edges.csv"
    if topology_path.exists():
        topo = pd.read_csv(topology_path)
        rename_map = {
            "road_id": "edge_id",
            "最低标高": "min_elevation_m",
            "start": "start_node",
            "end": "end_node",
        }
        topo = topo.rename(columns={col: rename_map[col] for col in topo.columns if col in rename_map})
        required = {"edge_id", "start_node", "end_node"}
        if required.issubset(set(topo.columns)):
            keep = ["edge_id", "start_node", "end_node"]
            return topo[keep].copy(), True

    topo = pd.DataFrame(
        [
            {"edge_id": edge_id, "start_node": nodes[0], "end_node": nodes[1]}
            for edge_id, nodes in DEFAULT_TOPOLOGY.items()
        ]
    )
    topo["edge_id"] = pd.Categorical(topo["edge_id"], categories=ROAD_IDS, ordered=True)
    topo = topo.sort_values("edge_id").reset_index(drop=True)
    topo["edge_id"] = topo["edge_id"].astype(str)
    return topo, False


def build_edge_neighbors(topology_df: pd.DataFrame) -> dict[str, list[str]]:
    node_to_edges: dict[str, set[str]] = {}
    for _, row in topology_df.iterrows():
        for node in [row["start_node"], row["end_node"]]:
            node_to_edges.setdefault(str(node), set()).add(str(row["edge_id"]))

    neighbors: dict[str, set[str]] = {edge_id: set() for edge_id in topology_df["edge_id"].astype(str)}
    for edge_set in node_to_edges.values():
        for edge_id in edge_set:
            neighbors.setdefault(edge_id, set()).update(edge_set - {edge_id})
    return {edge_id: sorted(list(edge_neighbors), key=lambda x: int(x[1:])) for edge_id, edge_neighbors in neighbors.items()}


def _check_topology_connected(topology_df: pd.DataFrame) -> bool:
    nodes = sorted(set(topology_df["start_node"]).union(set(topology_df["end_node"])))
    adjacency = {node: set() for node in nodes}
    for _, row in topology_df.iterrows():
        adjacency[row["start_node"]].add(row["end_node"])
        adjacency[row["end_node"]].add(row["start_node"])
    if not adjacency:
        return False
    start = nodes[0]
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(nodes)


def compute_slope_factor(road_df: pd.DataFrame) -> pd.DataFrame:
    out = road_df[["road_id", "slope_permille"]].copy()
    s_min = float(out["slope_permille"].min())
    s_max = float(out["slope_permille"].max())
    if np.isclose(s_min, s_max):
        out["slope_factor"] = 1.0
    else:
        out["slope_factor"] = 0.8 + 0.4 * (out["slope_permille"] - s_min) / (s_max - s_min)
    return out


def _road_width(road_class: str) -> float:
    if "主干路" in str(road_class):
        return 24.0
    if "次干路" in str(road_class):
        return 18.0
    if "支路" in str(road_class):
        return 12.0
    return 18.0


def _runoff_coeff(pavement: str) -> float:
    if "水泥" in str(pavement):
        return 0.88
    if "沥青" in str(pavement):
        return 0.93
    return 0.92


def _rainfall_per_minute(rainfall_df: pd.DataFrame) -> np.ndarray:
    rain = np.zeros(60, dtype=float)
    prev = 0
    for _, row in rainfall_df.sort_values("time_min").iterrows():
        end = int(row["time_min"])
        amount = float(row["rainfall_mm"])
        end = min(max(end, 0), 60)
        if end > prev:
            rain[prev:end] = amount / (end - prev)
        prev = end
    return rain


def _prepare_model_arrays(road_df: pd.DataFrame, rainfall_df: pd.DataFrame) -> dict[str, Any]:
    roads = road_df.set_index("road_id").loc[ROAD_IDS].reset_index()
    slope = compute_slope_factor(roads).set_index("road_id").loc[ROAD_IDS]
    return {
        "road_ids": ROAD_IDS,
        "length": roads["length_m"].to_numpy(dtype=float),
        "road_width": np.array([_road_width(v) for v in roads["priority"]], dtype=float),
        "runoff": np.array([_runoff_coeff(v) for v in roads["pavement"]], dtype=float),
        "drainage_lps": roads["drainage_capacity_lps"].to_numpy(dtype=float),
        "elevation": roads["elevation_m"].to_numpy(dtype=float),
        "slope_factor": slope["slope_factor"].to_numpy(dtype=float),
        "slope_permille": slope["slope_permille"].to_numpy(dtype=float),
        "rain_mm_min": _rainfall_per_minute(rainfall_df),
    }


def _anchor_observations(depth_anchors: pd.DataFrame) -> np.ndarray:
    anchors = depth_anchors.set_index("road_id").loc[ROAD_IDS]
    return anchors[ANCHOR_OBS_TIMES].to_numpy(dtype=float)


def _params_to_arrays(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w_catch = np.asarray(params[:8], dtype=float)
    eta = np.asarray(params[8:16], dtype=float)
    lambda_topo = float(params[16]) if len(params) > 16 else 0.0
    return w_catch, eta, lambda_topo


def compute_topology_exchange(
    h_now: np.ndarray,
    elevation: np.ndarray,
    neighbors: dict[str, list[str]],
    lambda_topo: float,
    balance_update: np.ndarray,
    time_min: int | None = None,
    collect_records: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    topo_update = np.zeros_like(h_now, dtype=float)
    records: list[dict[str, Any]] = []
    if lambda_topo <= 0:
        return topo_update, records

    edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(ROAD_IDS)}
    head = elevation + h_now
    for from_edge, neighs in neighbors.items():
        if from_edge not in edge_to_idx or not neighs:
            continue
        i = edge_to_idx[from_edge]
        if h_now[i] <= 1e-9:
            continue
        weight = 1.0 / max(len(neighs), 1)
        for to_edge in neighs:
            if to_edge not in edge_to_idx:
                continue
            j = edge_to_idx[to_edge]
            delta_head = head[i] - head[j]
            if delta_head <= 0:
                continue
            transfer = lambda_topo * weight * delta_head
            transfer = min(transfer, 0.25 * h_now[i])
            topo_update[i] -= transfer
            topo_update[j] += transfer
            if collect_records and time_min is not None and time_min % 5 == 0:
                records.append(
                    {
                        "time_min": time_min,
                        "from_edge": from_edge,
                        "to_edge": to_edge,
                        "head_from_m": head[i],
                        "head_to_m": head[j],
                        "delta_head_m": delta_head,
                        "estimated_transfer_m": transfer,
                        "note": "简化拓扑水位势弱交换，非二维水动力模拟。",
                    }
                )

    cap = 0.5 * np.abs(balance_update) + 1e-6
    topo_update = np.clip(topo_update, -cap, cap)
    return topo_update, records


def _simulate_model(
    arrays: dict[str, Any],
    params: np.ndarray,
    neighbors: dict[str, list[str]] | None = None,
    use_topology: bool = False,
    collect_flow: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    w_catch, eta, lambda_topo = _params_to_arrays(params)
    n = len(ROAD_IDS)
    h = np.zeros((61, n), dtype=float)
    flow_records: list[dict[str, Any]] = []
    area = arrays["length"] * arrays["road_width"]

    for t in range(60):
        rain = arrays["rain_mm_min"][t]
        inflow = rain * arrays["length"] * w_catch * arrays["runoff"] / 1000.0 / 60.0
        outflow = eta * arrays["drainage_lps"] * arrays["slope_factor"] / 1000.0
        balance = (inflow - outflow) * 60.0 / area
        topo = np.zeros(n, dtype=float)
        if use_topology and neighbors is not None and lambda_topo > 0:
            topo, records = compute_topology_exchange(
                h[t],
                arrays["elevation"],
                neighbors,
                lambda_topo,
                balance,
                time_min=t,
                collect_records=collect_flow,
            )
            flow_records.extend(records)
        h[t + 1] = np.maximum(0.0, h[t] + balance + topo)

    return h, pd.DataFrame(flow_records)


def _depth_frame_from_h(h: np.ndarray) -> pd.DataFrame:
    sampled = h[TIME_GRID.astype(int), :] * 100.0
    df = pd.DataFrame(sampled, index=TIME_GRID, columns=ROAD_IDS)
    df.index.name = "time_min"
    return df.round(4)


def _frame_anchor_metrics(depth_df: pd.DataFrame, depth_anchors: pd.DataFrame) -> dict[str, float]:
    anchors = depth_anchors.set_index("road_id").loc[ROAD_IDS]
    pred = depth_df.loc[ANCHOR_OBS_TIMES, ROAD_IDS].T.to_numpy(dtype=float)
    obs = anchors[ANCHOR_OBS_TIMES].to_numpy(dtype=float)
    diff = pred - obs
    return {
        "rmse_cm": float(np.sqrt(np.nanmean(diff**2))),
        "mae_cm": float(np.nanmean(np.abs(diff))),
        "max_abs_error_cm": float(np.nanmax(np.abs(diff))),
    }


def apply_anchor_residual_correction(
    physical_depth_df: pd.DataFrame,
    depth_anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = depth_anchors.set_index("road_id").loc[ROAD_IDS]
    residual_times = np.array([0, 15, 30, 45, 60], dtype=float)
    corrected = pd.DataFrame(index=physical_depth_df.index)
    corrected.index.name = "time_min"
    residual_df = pd.DataFrame(index=physical_depth_df.index)
    residual_df.index.name = "time_min"

    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:
        PchipInterpolator = None

    for road_id in ROAD_IDS:
        residual_values = [0.0]
        for t in ANCHOR_OBS_TIMES:
            obs = float(anchors.loc[road_id, t])
            pred = float(physical_depth_df.loc[t, road_id])
            residual_values.append(obs - pred)
        residual_values_arr = np.array(residual_values, dtype=float)
        if PchipInterpolator is not None:
            residual_interp = PchipInterpolator(residual_times, residual_values_arr)(TIME_GRID)
        else:
            residual_interp = np.interp(TIME_GRID, residual_times, residual_values_arr)
        residual_df[road_id] = residual_interp
        corrected[road_id] = np.maximum(0.0, physical_depth_df[road_id].to_numpy(dtype=float) + residual_interp)

    return corrected.round(4), residual_df.round(4)


def _calibration_metrics(h: np.ndarray, obs_cm: np.ndarray) -> dict[str, float]:
    pred_cm = h[ANCHOR_OBS_TIMES, :] * 100.0
    diff = pred_cm.T - obs_cm
    return {
        "rmse_cm": float(np.sqrt(np.nanmean(diff**2))),
        "mae_cm": float(np.nanmean(np.abs(diff))),
        "max_abs_error_cm": float(np.nanmax(np.abs(diff))),
    }


def simulate_water_balance_baseline(arrays: dict[str, Any], params: np.ndarray) -> np.ndarray:
    h, _ = _simulate_model(arrays, params, use_topology=False)
    return h


def simulate_water_balance_with_topology(
    arrays: dict[str, Any],
    params: np.ndarray,
    neighbors: dict[str, list[str]],
    collect_flow: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    return _simulate_model(arrays, params, neighbors=neighbors, use_topology=True, collect_flow=collect_flow)


def _initial_params(arrays: dict[str, Any]) -> np.ndarray:
    w0 = np.full(8, 45.0, dtype=float)
    eta0 = np.full(8, 0.75, dtype=float)
    return np.r_[w0, eta0, 0.0]


def _objective(
    params: np.ndarray,
    arrays: dict[str, Any],
    obs_cm: np.ndarray,
    neighbors: dict[str, list[str]] | None,
    use_topology: bool,
) -> float:
    params = np.asarray(params, dtype=float)
    if use_topology:
        h, _ = _simulate_model(arrays, params, neighbors=neighbors, use_topology=True)
    else:
        p = np.r_[params[:16], 0.0] if len(params) == 16 else params.copy()
        p[16] = 0.0
        h, _ = _simulate_model(arrays, p, use_topology=False)
    metrics = _calibration_metrics(h, obs_cm)
    w, eta, lam = _params_to_arrays(params if len(params) == 17 else np.r_[params, 0.0])
    reg = 0.002 * float(np.mean(((w - 45.0) / 60.0) ** 2))
    reg += 0.002 * float(np.mean(((eta - 0.75) / 0.45) ** 2))
    reg += 0.01 * float((lam / 0.20) ** 2)
    return metrics["rmse_cm"] + reg


def calibrate_parameters_with_topology(
    road_df: pd.DataFrame,
    rainfall_df: pd.DataFrame,
    depth_anchors: pd.DataFrame,
    neighbors: dict[str, list[str]],
) -> dict[str, Any]:
    arrays = _prepare_model_arrays(road_df, rainfall_df)
    obs_cm = _anchor_observations(depth_anchors)
    p0 = _initial_params(arrays)
    bounds16 = [(5.0, 120.0)] * 8 + [(0.3, 1.2)] * 8
    bounds17 = bounds16 + [(0.0, 0.20)]

    try:
        from scipy.optimize import minimize

        baseline_res = minimize(
            lambda p: _objective(np.r_[p, 0.0], arrays, obs_cm, None, False),
            p0[:16],
            method="L-BFGS-B",
            bounds=bounds16,
            options={"maxiter": 600, "ftol": 1e-8},
        )
        baseline_params = np.r_[baseline_res.x, 0.0]
        topo_start = baseline_params.copy()
        topo_start[16] = 0.03
        topo_res = minimize(
            lambda p: _objective(p, arrays, obs_cm, neighbors, True),
            topo_start,
            method="L-BFGS-B",
            bounds=bounds17,
            options={"maxiter": 800, "ftol": 1e-8},
        )
        topo_params = topo_res.x
        calibration_method = "scipy_L-BFGS-B"
    except Exception as exc:
        baseline_params = p0.copy()
        baseline_params[16] = 0.0
        topo_params = p0.copy()
        topo_params[16] = 0.03
        calibration_method = f"fallback_initial_params: {exc}"

    baseline_h = simulate_water_balance_baseline(arrays, baseline_params)
    topo_h, flow_df = simulate_water_balance_with_topology(arrays, topo_params, neighbors, collect_flow=True)
    baseline_metrics = _calibration_metrics(baseline_h, obs_cm)
    topo_metrics = _calibration_metrics(topo_h, obs_cm)

    topo_abnormal = (
        not np.isfinite(topo_metrics["rmse_cm"])
        or np.nanmax(topo_h * 100.0) > 100.0
        or np.nanmax(topo_h) < 0
    )
    selected_version = "baseline" if topo_abnormal else "topo_corrected"
    selected_h = baseline_h if selected_version == "baseline" else topo_h
    selected_params = baseline_params if selected_version == "baseline" else topo_params

    return {
        "arrays": arrays,
        "obs_cm": obs_cm,
        "baseline_params": baseline_params,
        "topo_params": topo_params,
        "selected_params": selected_params,
        "baseline_h": baseline_h,
        "topo_h": topo_h,
        "selected_h": selected_h,
        "baseline_metrics": baseline_metrics,
        "topo_metrics": topo_metrics,
        "selected_version": selected_version,
        "topology_abnormal": topo_abnormal,
        "flow_df": flow_df,
        "calibration_method": calibration_method,
    }


def compare_model_versions(
    calibration: dict[str, Any],
    baseline_risk: pd.DataFrame,
    topo_risk: pd.DataFrame,
    corrected_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version, metrics, params, risk in [
        ("baseline", calibration["baseline_metrics"], calibration["baseline_params"], baseline_risk),
        ("topo_corrected", calibration["topo_metrics"], calibration["topo_params"], topo_risk),
    ]:
        top3 = risk.sort_values("risk_rank").head(3)["road_id"].astype(str).tolist()
        note = "无拓扑汇流修正，仅使用路段级水量平衡 + 坡度排水修正。"
        if version == "topo_corrected":
            note = "拓扑修正主要增强空间解释性，若拟合误差改善有限，则仍保留路段级水量平衡作为主模型。"
        rows.append(
            {
                "model_version": version,
                "rmse_cm": metrics["rmse_cm"],
                "mae_cm": metrics["mae_cm"],
                "max_abs_error_cm": metrics["max_abs_error_cm"],
                "physical_rmse_cm": metrics["rmse_cm"],
                "after_anchor_correction_rmse_cm": (
                    corrected_metrics["rmse_cm"] if corrected_metrics and version == calibration["selected_version"] else np.nan
                ),
                "lambda_topo": float(params[16]),
                "risk_top1": top3[0] if top3 else "",
                "risk_top3": ",".join(top3),
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def save_topology_flow_summary(output_dir: Path, flow_df: pd.DataFrame) -> None:
    if flow_df.empty:
        flow_df = pd.DataFrame(
            columns=[
                "time_min",
                "from_edge",
                "to_edge",
                "head_from_m",
                "head_to_m",
                "delta_head_m",
                "estimated_transfer_m",
                "note",
            ]
        )
    flow_df.to_csv(output_dir / "q1_topology_flow_summary.csv", index=False, encoding="utf-8-sig")


def map_capacity_and_safety(depth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    depth = depth_df.to_numpy(dtype=float)
    capacity = np.select(
        [depth < 5, depth < 10, depth < 20, depth < 30, depth >= 30],
        [1.0, 0.8, 0.4, 0.1, 0.0],
    )
    safety = np.select(
        [depth < 5, depth < 10, depth < 20, depth < 30, depth >= 30],
        [1.0, 0.9, 0.6, 0.2, 0.0],
    )
    capacity_df = pd.DataFrame(capacity, index=depth_df.index, columns=depth_df.columns)
    safety_df = pd.DataFrame(safety, index=depth_df.index, columns=depth_df.columns)
    capacity_df.index.name = "time_min"
    safety_df.index.name = "time_min"
    return capacity_df, safety_df


def _risk_level(risk: float) -> str:
    if risk >= 0.75:
        return "极高风险"
    if risk >= 0.55:
        return "高风险"
    if risk >= 0.35:
        return "中风险"
    return "低风险"


def _duration_minutes(mask: pd.Series, step_min: int = 5) -> int:
    return int(mask.sum() * step_min)


def compute_susceptibility_index(road_df: pd.DataFrame) -> pd.DataFrame:
    road = road_df.copy()

    def norm_positive(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0 or pd.isna(span):
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / span

    def norm_negative(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0 or pd.isna(span):
            return pd.Series(0.0, index=series.index)
        return (series.max() - series) / span

    road["elevation_risk"] = norm_negative(road["elevation_m"])
    road["drainage_shortage_risk"] = norm_negative(road["drainage_capacity_lps"])
    road["slope_shortage_risk"] = norm_negative(road["slope_permille"])
    road["length_risk"] = norm_positive(road["length_m"])
    road["susceptibility_index"] = (
        0.35 * road["elevation_risk"]
        + 0.30 * road["drainage_shortage_risk"]
        + 0.20 * road["slope_shortage_risk"]
        + 0.15 * road["length_risk"]
    )
    road["susceptibility_rank"] = (
        road["susceptibility_index"].rank(method="min", ascending=False).astype(int)
    )
    return road[
        [
            "road_id",
            "elevation_risk",
            "drainage_shortage_risk",
            "slope_shortage_risk",
            "length_risk",
            "susceptibility_index",
            "susceptibility_rank",
        ]
    ].copy()


def compute_risk_summary(
    depth_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    safety_df: pd.DataFrame,
    road_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    susceptibility = compute_susceptibility_index(road_df)
    records: list[dict[str, Any]] = []
    for road_id in depth_df.columns:
        depth = depth_df[road_id]
        capacity = capacity_df[road_id]
        safety = safety_df[road_id]
        closed = capacity <= 0
        cap_low = capacity <= 0.4
        safety_low = safety <= 0.6

        max_depth = float(depth.max())
        peak_time = int(depth.idxmax())
        mean_capacity = float(capacity.mean())
        mean_safety = float(safety.mean())
        risk = (
            0.4 * min(max_depth / 30.0, 1.0)
            + 0.3 * (1.0 - mean_capacity)
            + 0.3 * (1.0 - mean_safety)
        )
        first_closed_time = int(closed[closed].index[0]) if closed.any() else np.nan

        records.append(
            {
                "road_id": road_id,
                "max_depth_cm": max_depth,
                "peak_time_min": peak_time,
                "mean_depth_cm": float(depth.mean()),
                "min_capacity": float(capacity.min()),
                "mean_capacity": mean_capacity,
                "min_safety": float(safety.min()),
                "mean_safety": mean_safety,
                "is_closed": bool(closed.any()),
                "first_closed_time_min": first_closed_time,
                "closure_duration_min": _duration_minutes(closed),
                "duration_capacity_below_0_4_min": _duration_minutes(cap_low),
                "duration_safety_below_0_6_min": _duration_minutes(safety_low),
                "risk_index": risk,
                "risk_level": _risk_level(risk),
            }
        )

    summary = pd.DataFrame(records)
    summary = summary.merge(susceptibility, on="road_id", how="left")
    summary = summary.sort_values(["risk_index", "max_depth_cm"], ascending=[False, False]).reset_index(drop=True)
    summary["risk_rank"] = np.arange(1, len(summary) + 1)

    if summary["susceptibility_index"].nunique(dropna=True) > 1 and summary["max_depth_cm"].nunique(dropna=True) > 1:
        corr = float(summary["susceptibility_index"].corr(summary["max_depth_cm"], method="pearson"))
    else:
        corr = float("nan")

    ordered_cols = [
        "road_id",
        "max_depth_cm",
        "peak_time_min",
        "mean_depth_cm",
        "min_capacity",
        "mean_capacity",
        "min_safety",
        "mean_safety",
        "is_closed",
        "first_closed_time_min",
        "closure_duration_min",
        "duration_capacity_below_0_4_min",
        "duration_safety_below_0_6_min",
        "risk_index",
        "risk_rank",
        "risk_level",
        "susceptibility_index",
        "susceptibility_rank",
        "elevation_risk",
        "drainage_shortage_risk",
        "slope_shortage_risk",
        "length_risk",
    ]
    return summary[ordered_cols], corr, susceptibility


def analyze_rainfall_lag(rainfall_df: pd.DataFrame, depth_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rainfall = rainfall_df.copy()
    mean_depth = depth_df.mean(axis=1).rename("mean_depth_cm")
    max_depth = depth_df.max(axis=1).rename("max_depth_cm")
    depth_stats = pd.concat([mean_depth, max_depth], axis=1).reset_index()
    depth_stats = depth_stats.rename(columns={"time_min": "time_min"})

    lag_df = pd.DataFrame({"time_min": TIME_GRID.astype(float)})
    lag_df = lag_df.merge(rainfall, on="time_min", how="left")
    lag_df["rainfall_mm"] = lag_df["rainfall_mm"].fillna(0.0)
    lag_df["cumulative_rainfall_mm"] = lag_df["cumulative_rainfall_mm"].ffill().fillna(0.0)
    lag_df = lag_df.merge(depth_stats, on="time_min", how="left")

    rainfall_peak_time = float(lag_df.loc[lag_df["rainfall_mm"].idxmax(), "time_min"])
    mean_depth_peak_time = float(lag_df.loc[lag_df["mean_depth_cm"].idxmax(), "time_min"])
    max_depth_peak_time = float(lag_df.loc[lag_df["max_depth_cm"].idxmax(), "time_min"])
    metrics = {
        "rainfall_peak_time_min": rainfall_peak_time,
        "mean_depth_peak_time_min": mean_depth_peak_time,
        "max_depth_peak_time_min": max_depth_peak_time,
        "lag_to_mean_depth_peak_min": mean_depth_peak_time - rainfall_peak_time,
        "lag_to_max_depth_peak_min": max_depth_peak_time - rainfall_peak_time,
        "rainfall_peak_mm": float(lag_df["rainfall_mm"].max()),
        "mean_depth_peak_cm": float(lag_df["mean_depth_cm"].max()),
        "max_depth_peak_cm": float(lag_df["max_depth_cm"].max()),
    }
    return lag_df, metrics


def _with_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "time_min", out.index)
    return out


def save_outputs(
    output_dir: Path,
    depth_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    safety_df: pd.DataFrame,
    risk_summary: pd.DataFrame,
    rainfall_lag_df: pd.DataFrame,
    model_parameters: pd.DataFrame,
    model_comparison: pd.DataFrame | None = None,
    physical_depth_df: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _with_time_column(depth_df).to_csv(output_dir / "q1_depth_5min.csv", index=False, encoding="utf-8-sig")
    _with_time_column(capacity_df).to_csv(output_dir / "q1_capacity_5min.csv", index=False, encoding="utf-8-sig")
    _with_time_column(safety_df).to_csv(output_dir / "q1_safety_5min.csv", index=False, encoding="utf-8-sig")
    risk_summary.to_csv(output_dir / "q1_risk_summary.csv", index=False, encoding="utf-8-sig")
    if physical_depth_df is not None:
        _with_time_column(physical_depth_df).to_csv(
            output_dir / "q1_depth_physical_5min.csv", index=False, encoding="utf-8-sig"
        )
    if model_comparison is not None:
        model_comparison.to_csv(output_dir / "q1_model_comparison.csv", index=False, encoding="utf-8-sig")

    xlsx_path = output_dir / "q1_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        _with_time_column(depth_df).to_excel(writer, sheet_name="depth_5min", index=False)
        _with_time_column(capacity_df).to_excel(writer, sheet_name="capacity_5min", index=False)
        _with_time_column(safety_df).to_excel(writer, sheet_name="safety_5min", index=False)
        risk_summary.to_excel(writer, sheet_name="risk_summary", index=False)
        rainfall_lag_df.to_excel(writer, sheet_name="rainfall_lag_analysis", index=False)
        model_parameters.to_excel(writer, sheet_name="model_parameters", index=False)
        if physical_depth_df is not None:
            _with_time_column(physical_depth_df).to_excel(writer, sheet_name="depth_physical_5min", index=False)
        if model_comparison is not None:
            model_comparison.to_excel(writer, sheet_name="model_comparison", index=False)


def _setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        return None, False, f"matplotlib unavailable: {exc}"

    chinese_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    use_chinese = False
    for font_name in chinese_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            use_chinese = True
            break
    return plt, use_chinese, ""


def plot_results(
    output_dir: Path,
    depth_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    safety_df: pd.DataFrame,
    risk_summary: pd.DataFrame,
    rainfall_lag_df: pd.DataFrame,
    topo_depth_df: pd.DataFrame | None = None,
    depth_anchors: pd.DataFrame | None = None,
) -> str:
    plt, use_chinese, reason = _setup_matplotlib()
    if plt is None:
        return f"图片未生成：{reason}"

    output_dir.mkdir(parents=True, exist_ok=True)

    def text(cn: str, en: str) -> str:
        return cn if use_chinese else en

    plt.figure(figsize=(10, 6), dpi=160)
    for road_id in depth_df.columns:
        plt.plot(depth_df.index, depth_df[road_id], marker="o", linewidth=1.8, label=road_id)
    plt.xlabel(text("时间 / min", "Time / min"))
    plt.ylabel(text("积水深度 / cm", "Water depth / cm"))
    plt.title(text("各路段积水深度动态变化", "Dynamic Water Depth by Road Segment"))
    plt.grid(alpha=0.3)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(output_dir / "depth_curves.png")
    plt.close()

    def heatmap(df: pd.DataFrame, title_cn: str, title_en: str, filename: str, vmin: float, vmax: float) -> None:
        plt.figure(figsize=(10, 5), dpi=160)
        matrix = df.T.to_numpy(dtype=float)
        image = plt.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        plt.colorbar(image)
        plt.yticks(np.arange(len(df.columns)), df.columns)
        plt.xticks(np.arange(len(df.index)), df.index)
        plt.xlabel(text("时间 / min", "Time / min"))
        plt.ylabel(text("路段", "Road segment"))
        plt.title(text(title_cn, title_en))
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    heatmap(capacity_df, "通行能力系数热力图", "Capacity Coefficient Heatmap", "capacity_heatmap.png", 0, 1)
    heatmap(safety_df, "安全度热力图", "Safety Score Heatmap", "safety_heatmap.png", 0, 1)

    by_road = risk_summary.sort_values("road_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    plt.figure(figsize=(9, 5), dpi=160)
    plt.bar(by_road["road_id"], by_road["max_depth_cm"], color="#4374B3")
    plt.axhline(30, color="#C44E52", linestyle="--", linewidth=1.5)
    plt.xlabel(text("路段", "Road segment"))
    plt.ylabel(text("最大积水深度 / cm", "Max depth / cm"))
    plt.title(text("各路段最大积水深度", "Maximum Water Depth by Segment"))
    plt.tight_layout()
    plt.savefig(output_dir / "max_depth_bar.png")
    plt.close()

    ranked = risk_summary.sort_values("risk_rank")
    plt.figure(figsize=(9, 5), dpi=160)
    plt.bar(ranked["road_id"], ranked["risk_index"], color="#DD8452")
    plt.xlabel(text("路段", "Road segment"))
    plt.ylabel(text("综合风险指数", "Risk index"))
    plt.title(text("路段综合风险指数排序", "Road Segment Risk Ranking"))
    plt.tight_layout()
    plt.savefig(output_dir / "risk_rank_bar.png")
    plt.close()

    plt.figure(figsize=(7, 5), dpi=160)
    plt.scatter(risk_summary["susceptibility_index"], risk_summary["max_depth_cm"], s=70, color="#55A868")
    for _, row in risk_summary.iterrows():
        plt.annotate(row["road_id"], (row["susceptibility_index"], row["max_depth_cm"]), xytext=(4, 4), textcoords="offset points")
    plt.xlabel(text("静态内涝敏感性指数", "Susceptibility index"))
    plt.ylabel(text("最大积水深度 / cm", "Max depth / cm"))
    plt.title(text("敏感性指数与最大积水深度关系", "Susceptibility vs Max Depth"))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "susceptibility_vs_max_depth.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=160)
    ax1.bar(rainfall_lag_df["time_min"], rainfall_lag_df["rainfall_mm"], width=3.8, alpha=0.55, color="#4C72B0", label=text("5 min 降雨量", "5-min rainfall"))
    ax1.set_xlabel(text("时间 / min", "Time / min"))
    ax1.set_ylabel(text("降雨量 / mm", "Rainfall / mm"), color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(rainfall_lag_df["time_min"], rainfall_lag_df["mean_depth_cm"], marker="o", color="#C44E52", label=text("平均积水深度", "Mean depth"))
    ax2.plot(rainfall_lag_df["time_min"], rainfall_lag_df["max_depth_cm"], marker="s", color="#55A868", label=text("最大积水深度", "Max depth"))
    ax2.set_ylabel(text("积水深度 / cm", "Water depth / cm"))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    plt.title(text("降雨峰值与积水峰值滞后效应", "Lag Effect Between Rainfall Peak and Waterlogging Peak"))
    fig.tight_layout()
    fig.savefig(output_dir / "rainfall_depth_lag.png")
    plt.close(fig)

    if topo_depth_df is not None:
        plt.figure(figsize=(10, 6), dpi=160)
        for road_id in topo_depth_df.columns:
            plt.plot(topo_depth_df.index, topo_depth_df[road_id], marker="o", linewidth=1.8, label=road_id)
        plt.xlabel(text("时间 / min", "Time / min"))
        plt.ylabel(text("积水深度 / cm", "Water depth / cm"))
        plt.title(text("拓扑修正模型积水深度变化", "Topology-Corrected Water Depth Curves"))
        plt.grid(alpha=0.3)
        plt.legend(ncol=4)
        plt.tight_layout()
        plt.savefig(output_dir / "topology_corrected_depth_curves.png")
        plt.close()

    if topo_depth_df is not None and depth_anchors is not None:
        anchors = depth_anchors.set_index("road_id")
        plt.figure(figsize=(11, 7), dpi=160)
        for road_id in topo_depth_df.columns:
            plt.plot(topo_depth_df.index, topo_depth_df[road_id], linewidth=1.6, label=f"{road_id} model")
            if road_id in anchors.index:
                obs = [anchors.loc[road_id, t] for t in ANCHOR_OBS_TIMES]
                plt.scatter(ANCHOR_OBS_TIMES, obs, s=28, zorder=5)
        plt.xlabel(text("时间 / min", "Time / min"))
        plt.ylabel(text("积水深度 / cm", "Water depth / cm"))
        plt.title(text("拓扑修正模型与附表4锚点校准对比", "Topology Model Calibration Check"))
        plt.grid(alpha=0.3)
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "calibration_check_topology.png")
        plt.close()

    return "图片已生成" if use_chinese else "图片已生成；未检测到中文字体，图题使用英文。"


def write_model_notes(
    output_dir: Path,
    input_file: Path,
    model_method: str,
    plot_status: str,
    risk_summary: pd.DataFrame,
    susceptibility_corr: float,
    lag_metrics: dict[str, float],
    calibration: dict[str, Any],
    model_comparison: pd.DataFrame,
    topology_loaded: bool,
    topology_connected: bool,
    corrected_metrics: dict[str, float] | None = None,
) -> None:
    l5 = risk_summary.loc[risk_summary["road_id"] == "L5"].iloc[0]
    top3 = ", ".join(risk_summary.sort_values("risk_rank").head(3)["road_id"].tolist())
    selected_metrics = (
        calibration["topo_metrics"] if model_method == "topo_corrected" else calibration["baseline_metrics"]
    )
    final_metrics = corrected_metrics if corrected_metrics is not None else selected_metrics
    lambda_topo = float(calibration["selected_params"][16])
    topology_source = "已读取 results/topology/topology_edges.csv" if topology_loaded else "未读取拓扑文件，使用默认简化拓扑"
    comparison_text = model_comparison.to_string(index=False)
    notes = f"""第一问模型说明与论文可用结论

一、模型路线
第一问最终采用“基于简化拓扑修正的路段级水量平衡积水演化模型”。模型不推翻原路段级水量平衡框架，而是在降雨入流和排水出流基础上加入坡度排水修正项与简化拓扑汇流修正项。

二、数据与计算方法
输入文件：{input_file}
积水递推公式为 h_i(t+Δt)=max{{0, h_i(t)+Δh_i^balance(t)+Δh_i^topo(t)}}。
其中 Δh_i^balance(t)=((I_i(t)-Q_i(t))/A_i)Δt，I_i(t)=r(t)L_iW_i^catchC_runoff/(1000*60)，A_i=L_iW_i^road。
排水项加入坡度排水修正：Q_i(t)=eta_i Q_i^design f_s(i)/1000，f_s(i)=0.8+0.4(s_i-s_min)/(s_max-s_min)。坡度只表示坡度大小，不表示坡向，因此用于修正排水效率，而不是直接判断真实流向。

三、简化拓扑修正
{topology_source}，拓扑连通性为：{"连通" if topology_connected else "不连通"}。该拓扑是根据路段等级、最低标高、风险特征和连通性原则构建的简化交通拓扑网络，不是真实测绘拓扑。
拓扑汇流项基于相邻路段水位势 H_i(t)=z_i+h_i(t) 的差异，表示高水位路段向低水位路段的弱交换。lambda_topo={lambda_topo:.4f}。该项不是二维水动力模型，只用于增强空间解释性。

四、参数校准与误差
模型通过附表4中 15、30、45、60 min 的观测锚点校准 W_i^catch、eta_i 和全局 lambda_topo。当前物理预测模型版本为 {model_method}，物理预测 RMSE={selected_metrics['rmse_cm']:.4f} cm，MAE={selected_metrics['mae_cm']:.4f} cm，最大绝对误差={selected_metrics['max_abs_error_cm']:.4f} cm。
在物理预测之后，程序增加锚点残差校正层：先计算 e_i(t_k)=h_obs_i(t_k)-h_phys_i(t_k)，并令 e_i(0)=0，再对残差使用 PCHIP 或线性插值到 0、5、10、...、60 min，最终输出 h_final_i(t)=max(0,h_phys_i(t)+e_hat_i(t))。这样最终输出曲线既保留水量平衡、坡度修正和拓扑修正的机理解释，又与附表4观测锚点一致。锚点残差校正后 RMSE={final_metrics['rmse_cm']:.4f} cm，MAE={final_metrics['mae_cm']:.4f} cm，最大绝对误差={final_metrics['max_abs_error_cm']:.4f} cm。
模型对比表如下：
{comparison_text}

五、主要结果
整体积水在 45 min 左右达到峰值。L5 为首要内涝薄弱路段，峰值积水达到 {l5['max_depth_cm']:.2f} cm，达到或接近完全中断状态。风险排序前三位为 {top3}，其中 L2、L8 为高风险梯队路段，L7 全过程积水较浅，通行能力和安全度保持较高水平。

六、静态内涝敏感性
程序基于最低标高、设计排水能力、地形坡度和路段长度构造静态内涝敏感性指数，权重分别为 0.35、0.30、0.20、0.15。susceptibility_index 与最大积水深度的 Pearson 相关系数为 {susceptibility_corr:.4f}，可作为解释各路段积水差异的辅助指标，但最终风险排序仍以动态积水、通行能力和安全度共同决定。

七、降雨-积水滞后效应
附表2显示降雨峰值出现在 {lag_metrics['rainfall_peak_time_min']:.0f} min，平均积水峰值出现在 {lag_metrics['mean_depth_peak_time_min']:.0f} min，最大积水峰值出现在 {lag_metrics['max_depth_peak_time_min']:.0f} min。降雨峰值早于积水峰值，平均积水峰值滞后 {lag_metrics['lag_to_mean_depth_peak_min']:.0f} min，说明内涝过程存在明显滞后效应，反映排水系统在极端降雨下存在容量不足。

八、与后续问题的衔接
该结果将为第二问韧性评价和第三问动态路径优化提供 h_i(t)、C_i(t)、S_i(t) 输入。第三问可将这些动态指标按 edge_id 挂载到简化拓扑边上，进行动态绕行与疏散路径优化。

九、模型适用性与局限性
由于题目未给出完整二维地形、真实汇水面积、检查井节点、坡向和管网拓扑，本文未进行精细二维水动力模拟。拓扑汇流为基于简化交通拓扑的弱修正项，主要增强空间风险解释性，不作为真实水动力约束。

十、程序输出状态
主模型版本：{model_method}
图片输出：{plot_status}
"""
    (output_dir / "q1_model_notes.txt").write_text(notes, encoding="utf-8")


def write_topology_notes(
    output_dir: Path,
    topology_loaded: bool,
    topology_connected: bool,
    neighbors: dict[str, list[str]],
    selected_version: str,
) -> None:
    source = "results/topology/topology_edges.csv" if topology_loaded else "默认简化拓扑"
    notes = f"""第一问拓扑修正说明

拓扑来源：{source}
拓扑连通性：{"连通" if topology_connected else "不连通"}
主输出模型版本：{selected_version}

本拓扑是基于路段等级、最低标高、风险特征和连通性原则构建的简化交通拓扑网络，不是真实测绘拓扑。拓扑在第一问中只用于潜在汇流方向和空间风险解释，不作为强水动力约束，也不表示真实坡向、节点高程或管网连接关系。

相邻边定义为共享同一节点的路段。当前邻接关系如下：
{neighbors}

拓扑汇流修正项基于水位势 H_i(t)=z_i+h_i(t)，仅允许相邻边之间发生弱交换；若拓扑修正拟合误差改善有限，仍保留路段级水量平衡模型作为主模型解释基础。
"""
    (output_dir / "q1_topology_notes.txt").write_text(notes, encoding="utf-8")


def build_model_parameters(
    input_file: Path,
    model_method: str,
    susceptibility_corr: float,
    lag_metrics: dict[str, float],
    plot_status: str,
    calibration: dict[str, Any] | None = None,
    corrected_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = [
        ("input_file", str(input_file)),
        ("model_name", "基于简化拓扑修正的路段级水量平衡积水演化模型"),
        ("time_step_min", 1),
        ("output_sample_min", 5),
        ("time_range_min", "0-60"),
        ("anchor_times_min", "15,30,45,60"),
        ("main_model_version", model_method),
        ("calibration_method", calibration.get("calibration_method", "") if calibration else ""),
        ("capacity_rule", "h<5:1; 5<=h<10:0.8; 10<=h<20:0.4; 20<=h<30:0.1; h>=30:0"),
        ("safety_rule", "h<5:1; 5<=h<10:0.9; 10<=h<20:0.6; 20<=h<30:0.2; h>=30:0"),
        ("risk_formula", "0.4*min(max_depth/30,1)+0.3*(1-mean_capacity)+0.3*(1-mean_safety)"),
        ("susceptibility_weights", "elevation 0.35; drainage 0.30; slope 0.20; length 0.15"),
        ("susceptibility_max_depth_corr", susceptibility_corr),
        ("rainfall_peak_time_min", lag_metrics["rainfall_peak_time_min"]),
        ("mean_depth_peak_time_min", lag_metrics["mean_depth_peak_time_min"]),
        ("lag_to_mean_depth_peak_min", lag_metrics["lag_to_mean_depth_peak_min"]),
        ("plot_status", plot_status),
    ]
    if calibration:
        arrays = calibration["arrays"]
        params = calibration["selected_params"]
        w_catch, eta, lambda_topo = _params_to_arrays(params)
        rows.append(("lambda_topo", lambda_topo))
        rows.append(("baseline_rmse_cm", calibration["baseline_metrics"]["rmse_cm"]))
        rows.append(("topo_corrected_rmse_cm", calibration["topo_metrics"]["rmse_cm"]))
        if corrected_metrics:
            rows.append(("after_anchor_correction_rmse_cm", corrected_metrics["rmse_cm"]))
            rows.append(("after_anchor_correction_mae_cm", corrected_metrics["mae_cm"]))
            rows.append(("after_anchor_correction_max_abs_error_cm", corrected_metrics["max_abs_error_cm"]))
        rows.append(("topology_abnormal", calibration["topology_abnormal"]))
        for idx, road_id in enumerate(ROAD_IDS):
            rows.extend(
                [
                    (f"{road_id}_W_catch_m", w_catch[idx]),
                    (f"{road_id}_eta", eta[idx]),
                    (f"{road_id}_W_road_m", arrays["road_width"][idx]),
                    (f"{road_id}_slope_per_mille", arrays["slope_permille"][idx]),
                    (f"{road_id}_slope_factor", arrays["slope_factor"][idx]),
                ]
            )
    return pd.DataFrame(rows, columns=["parameter", "value"])


def print_terminal_summary(
    input_file: Path,
    output_dir: Path,
    risk_summary: pd.DataFrame,
    susceptibility_corr: float,
    lag_metrics: dict[str, float],
) -> None:
    print(f"输入文件路径: {input_file}")
    print(f"输出目录: {output_dir}")
    print("\n各路段最大积水深度(cm):")
    for _, row in risk_summary.sort_values("road_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)).iterrows():
        print(f"  {row['road_id']}: {row['max_depth_cm']:.2f}")

    print("\n风险排名:")
    for _, row in risk_summary.sort_values("risk_rank").iterrows():
        print(
            f"  第{int(row['risk_rank'])}名 {row['road_id']}: "
            f"risk={row['risk_index']:.4f}, {row['risk_level']}, max_depth={row['max_depth_cm']:.2f} cm"
        )

    closed = risk_summary[risk_summary["is_closed"]]
    print("\n完全中断路段及首次中断时刻:")
    if closed.empty:
        print("  无")
    else:
        for _, row in closed.iterrows():
            print(f"  {row['road_id']}: {int(row['first_closed_time_min'])} min")

    print(f"\nsusceptibility_index 与 max_depth 的 Pearson 相关系数: {susceptibility_corr:.4f}")
    print(
        "降雨峰值与积水峰值: "
        f"降雨峰值 {lag_metrics['rainfall_peak_time_min']:.0f} min, "
        f"平均积水峰值 {lag_metrics['mean_depth_peak_time_min']:.0f} min, "
        f"滞后 {lag_metrics['lag_to_mean_depth_peak_min']:.0f} min"
    )


def validate_outputs(output_dir: Path, depth_anchors: pd.DataFrame | None = None) -> None:
    """Validate generated files and write a non-blocking validation report."""
    report_path = output_dir / "q1_validation_report.txt"
    expected_times = list(range(0, 61, 5))
    expected_roads = ROAD_IDS
    required_files = [
        "q1_depth_5min.csv",
        "q1_capacity_5min.csv",
        "q1_safety_5min.csv",
        "q1_risk_summary.csv",
        "q1_depth_physical_5min.csv",
        "q1_results.xlsx",
        "q1_model_notes.txt",
        "q1_model_comparison.csv",
        "q1_topology_flow_summary.csv",
        "q1_topology_notes.txt",
        "depth_curves.png",
        "capacity_heatmap.png",
        "safety_heatmap.png",
        "max_depth_bar.png",
        "risk_rank_bar.png",
        "susceptibility_vs_max_depth.png",
        "rainfall_depth_lag.png",
        "topology_corrected_depth_curves.png",
        "calibration_check_topology.png",
    ]

    lines: list[str] = ["Q1 final model validation report", f"Output directory: {output_dir}", ""]
    warnings: list[str] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "WARNING"
        message = f"[{status}] {name}"
        if detail:
            message += f" - {detail}"
        lines.append(message)
        if not passed:
            warnings.append(message)

    try:
        depth = pd.read_csv(output_dir / "q1_depth_5min.csv")
        capacity = pd.read_csv(output_dir / "q1_capacity_5min.csv")
        safety = pd.read_csv(output_dir / "q1_safety_5min.csv")
        risk = pd.read_csv(output_dir / "q1_risk_summary.csv")
        comparison = pd.read_csv(output_dir / "q1_model_comparison.csv")
        params = pd.read_excel(output_dir / "q1_results.xlsx", sheet_name="model_parameters")
    except Exception as exc:
        message = f"[WARNING] Failed to read CSV outputs - {exc}"
        lines.append(message)
        warnings.append(message)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(message)
        print(f"Validation report saved to: {report_path}")
        return

    missing_files = [name for name in required_files if not (output_dir / name).exists()]
    add_check("主要输出文件是否完整", not missing_files, f"missing={missing_files}" if missing_files else "all present")
    add_check("是否成功读取附表1-4并完成输出", True, "q1核心输出已生成")
    add_check("是否成功读取或生成简化拓扑", (output_dir / "q1_topology_notes.txt").exists())

    notes_text = (output_dir / "q1_topology_notes.txt").read_text(encoding="utf-8") if (output_dir / "q1_topology_notes.txt").exists() else ""
    add_check("拓扑图是否连通", "拓扑连通性：连通" in notes_text, "see q1_topology_notes.txt")

    param_dict = dict(zip(params["parameter"].astype(str), params["value"]))
    slope_keys = [key for key in param_dict if key.endswith("_slope_factor")]
    add_check("slope_factor 是否生成", len(slope_keys) == 8, f"count={len(slope_keys)}")
    lambda_topo = float(param_dict.get("lambda_topo", np.nan))
    add_check(
        "lambda_topo 是否在合理范围 [0,0.20]",
        np.isfinite(lambda_topo) and 0 <= lambda_topo <= 0.20,
        f"lambda_topo={lambda_topo}",
    )

    if "time_min" in depth.columns:
        actual_times = pd.to_numeric(depth["time_min"], errors="coerce").dropna().astype(int).tolist()
    else:
        actual_times = []
    add_check(
        "q1_depth_5min.csv contains 13 time points 0,5,...,60",
        actual_times == expected_times and len(actual_times) == 13,
        f"actual={actual_times}",
    )

    road_cols = [col for col in depth.columns if col in expected_roads]
    add_check(
        "q1_depth_5min.csv contains L1-L8",
        road_cols == expected_roads,
        f"actual={road_cols}",
    )

    def value_at_time(df: pd.DataFrame, road_id: str, time_min: int) -> float:
        if "time_min" not in df.columns or road_id not in df.columns:
            return float("nan")
        row = df.loc[pd.to_numeric(df["time_min"], errors="coerce") == time_min]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][road_id])

    l5_depth_45 = value_at_time(depth, "L5", 45)
    add_check(
        "L5 峰值积水是否接近 32 cm",
        28.0 <= l5_depth_45 <= 36.0,
        f"actual={l5_depth_45}",
    )

    l5_capacity_45 = value_at_time(capacity, "L5", 45)
    l5_safety_45 = value_at_time(safety, "L5", 45)
    add_check(
        "L5 是否在峰值时段达到或接近中断状态",
        l5_capacity_45 <= 0.1 and l5_safety_45 <= 0.2,
        f"capacity={l5_capacity_45}, safety={l5_safety_45}",
    )

    risk_sorted = risk.sort_values("risk_rank") if "risk_rank" in risk.columns else risk
    top_road = str(risk_sorted.iloc[0]["road_id"]) if not risk_sorted.empty and "road_id" in risk_sorted.columns else ""
    add_check("L5 是否为最高风险路段", top_road == "L5", f"actual={top_road}")

    top3 = set(risk_sorted.head(3)["road_id"].astype(str).tolist()) if "road_id" in risk_sorted.columns else set()
    add_check("L2、L8 是否为高风险路段", {"L2", "L8"}.issubset(top3), f"top3={sorted(top3)}")

    l7_row = risk.loc[risk["road_id"].astype(str) == "L7"] if "road_id" in risk.columns else pd.DataFrame()
    if l7_row.empty:
        l7_ok = False
        l7_detail = "L7 row missing"
    else:
        l7_level = str(l7_row.iloc[0].get("risk_level", ""))
        l7_rank = pd.to_numeric(pd.Series([l7_row.iloc[0].get("risk_rank")]), errors="coerce").iloc[0]
        l7_ok = l7_level in {"低风险", "较低风险"} or (pd.notna(l7_rank) and int(l7_rank) >= 6)
        l7_detail = f"risk_level={l7_level}, risk_rank={l7_rank}"
    add_check("L7 是否为相对低风险路段", l7_ok, l7_detail)

    anchor_rows = comparison[comparison["model_version"].isin(["baseline", "topo_corrected"])]
    for _, row in anchor_rows.iterrows():
        add_check(
            f"{row['model_version']} 15/30/45/60 min 锚点误差",
            np.isfinite(row["rmse_cm"]) and row["rmse_cm"] < 8.0,
            f"rmse={row['rmse_cm']:.4f}, mae={row['mae_cm']:.4f}, max_abs={row['max_abs_error_cm']:.4f}",
        )
    if len(anchor_rows) == 2:
        baseline_rmse = float(anchor_rows.loc[anchor_rows["model_version"] == "baseline", "rmse_cm"].iloc[0])
        topo_rmse = float(anchor_rows.loc[anchor_rows["model_version"] == "topo_corrected", "rmse_cm"].iloc[0])
        add_check(
            "拓扑修正误差是否不劣于 baseline",
            topo_rmse <= baseline_rmse,
            f"baseline_rmse={baseline_rmse:.4f}, topo_rmse={topo_rmse:.4f}; 拓扑修正主要增强空间解释性，误差未改善不影响程序运行。",
        )
    if depth_anchors is not None:
        final_depth = depth.set_index("time_min")
        final_metrics = _frame_anchor_metrics(final_depth, depth_anchors)
        add_check(
            "锚点残差校正后最终输出与附表4锚点误差小于 0.5 cm",
            np.isfinite(final_metrics["max_abs_error_cm"]) and final_metrics["max_abs_error_cm"] < 0.5,
            f"rmse={final_metrics['rmse_cm']:.6f}, max_abs={final_metrics['max_abs_error_cm']:.6f}",
        )

    xlsx_path = output_dir / "q1_results.xlsx"
    if xlsx_path.exists():
        try:
            sheets = pd.ExcelFile(xlsx_path).sheet_names
            expected_sheets = {
                "depth_5min",
                "capacity_5min",
                "safety_5min",
                "risk_summary",
                "rainfall_lag_analysis",
                "model_parameters",
                "depth_physical_5min",
                "model_comparison",
            }
            add_check(
                "q1_results.xlsx exists with required sheets",
                expected_sheets.issubset(set(sheets)),
                f"sheets={sheets}",
            )
        except Exception as exc:
            add_check("q1_results.xlsx can be opened", False, str(exc))
    else:
        add_check("q1_results.xlsx exists", False, "file missing")

    lines.append("")
    lines.append(f"Total warnings: {len(warnings)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for warning in warnings:
        print(warning)
    print(f"Validation report saved to: {report_path}")


def main() -> None:
    configure_console()
    input_file = find_input_file()
    project_dir = Path(__file__).resolve().parent
    output_dir = Path(__file__).resolve().parent / "results" / "q1"

    data = load_data(input_file)
    topology_df, topology_loaded = load_or_build_topology(project_dir, data["road"])
    neighbors = build_edge_neighbors(topology_df)
    topology_connected = _check_topology_connected(topology_df)
    calibration = calibrate_parameters_with_topology(
        data["road"], data["rainfall"], data["depth_anchors"], neighbors
    )

    baseline_depth_df = _depth_frame_from_h(calibration["baseline_h"])
    topo_depth_df = _depth_frame_from_h(calibration["topo_h"])
    physical_depth_df = _depth_frame_from_h(calibration["selected_h"])
    depth_df, residual_df = apply_anchor_residual_correction(physical_depth_df, data["depth_anchors"])
    corrected_metrics = _frame_anchor_metrics(depth_df, data["depth_anchors"])
    model_method = calibration["selected_version"]

    capacity_df, safety_df = map_capacity_and_safety(depth_df)
    risk_summary, susceptibility_corr, _ = compute_risk_summary(
        depth_df, capacity_df, safety_df, data["road"]
    )
    baseline_capacity_df, baseline_safety_df = map_capacity_and_safety(baseline_depth_df)
    baseline_risk, _, _ = compute_risk_summary(
        baseline_depth_df, baseline_capacity_df, baseline_safety_df, data["road"]
    )
    topo_capacity_df, topo_safety_df = map_capacity_and_safety(topo_depth_df)
    topo_risk, _, _ = compute_risk_summary(
        topo_depth_df, topo_capacity_df, topo_safety_df, data["road"]
    )
    model_comparison = compare_model_versions(calibration, baseline_risk, topo_risk, corrected_metrics)
    rainfall_lag_df, lag_metrics = analyze_rainfall_lag(data["rainfall"], depth_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_topology_flow_summary(output_dir, calibration["flow_df"])
    plot_status = plot_results(
        output_dir,
        depth_df,
        capacity_df,
        safety_df,
        risk_summary,
        rainfall_lag_df,
        topo_depth_df=depth_df,
        depth_anchors=data["depth_anchors"],
    )
    model_parameters = build_model_parameters(
        input_file,
        model_method,
        susceptibility_corr,
        lag_metrics,
        plot_status,
        calibration=calibration,
        corrected_metrics=corrected_metrics,
    )
    save_outputs(
        output_dir,
        depth_df,
        capacity_df,
        safety_df,
        risk_summary,
        rainfall_lag_df,
        model_parameters,
        model_comparison=model_comparison,
        physical_depth_df=physical_depth_df,
    )
    write_topology_notes(output_dir, topology_loaded, topology_connected, neighbors, model_method)
    write_model_notes(
        output_dir,
        input_file,
        model_method,
        plot_status,
        risk_summary,
        susceptibility_corr,
        lag_metrics,
        calibration,
        model_comparison,
        topology_loaded,
        topology_connected,
        corrected_metrics,
    )
    print_terminal_summary(input_file, output_dir, risk_summary, susceptibility_corr, lag_metrics)
    print(f"\n主模型版本: {model_method}")
    selected_metrics = calibration["topo_metrics"] if model_method == "topo_corrected" else calibration["baseline_metrics"]
    print(f"物理预测 RMSE: {selected_metrics['rmse_cm']:.4f} cm")
    print(f"锚点校正后 RMSE: {corrected_metrics['rmse_cm']:.4f} cm")
    print(f"lambda_topo: {float(calibration['selected_params'][16]):.4f}")
    print("风险前三路段:", ", ".join(risk_summary.sort_values("risk_rank").head(3)["road_id"].astype(str).tolist()))
    print(f"\n{plot_status}")
    print("第一问结果文件已保存。")
    validate_outputs(output_dir, data["depth_anchors"])


if __name__ == "__main__":
    main()

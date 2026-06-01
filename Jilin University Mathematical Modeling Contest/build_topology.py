from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EDGE_NODE_MAP = {
    "L7": ("N1", "N2"),
    "L1": ("N2", "N5"),
    "L4": ("N2", "N3"),
    "L2": ("N3", "N4"),
    "L6": ("N3", "N6"),
    "L3": ("N5", "N6"),
    "L8": ("N6", "N7"),
    "L5": ("N4", "N7"),
}
EDGE_ORDER = [f"L{i}" for i in range(1, 9)]
NODE_ORDER = [f"N{i}" for i in range(1, 8)]


def configure_console() -> None:
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
    search_dirs: list[Path] = []
    for candidate in [Path.cwd(), script_dir, project_dir, known_wechat_dir]:
        if candidate.exists() and candidate not in search_dirs:
            search_dirs.append(candidate)

    matches: list[Path] = []
    for folder in search_dirs:
        matches.extend(p for p in folder.glob("B题附表1至6*.xlsx") if not p.name.startswith("~$"))

    if matches:
        matches = sorted(set(matches), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
        return matches[0].resolve()

    raise FileNotFoundError("未找到 B题附表1至6*.xlsx，请传入 Excel 路径。")


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


def _find_header_row(raw: pd.DataFrame, keywords: list[str]) -> int:
    for idx, row in raw.iterrows():
        text = " ".join(_cell_text(v) for v in row.tolist())
        if all(keyword in text for keyword in keywords):
            return int(idx)
    raise ValueError(f"无法定位表头行，关键字段: {keywords}")


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


def load_road_data(input_file: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(input_file)
    sheet_name = _find_sheet_name(xl.sheet_names, 1)
    raw = pd.read_excel(input_file, sheet_name=sheet_name, header=None)
    raw = raw.replace(r"^\s*$", np.nan, regex=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    raw = raw.reset_index(drop=True)

    header_row = _find_header_row(raw, ["路段编号", "长度"])
    headers = [_cell_text(v) or f"unnamed_{idx}" for idx, v in enumerate(raw.iloc[header_row].tolist())]
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = headers
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

    road_col = _find_column(list(table.columns), ["路段编号"])
    length_col = _find_column(list(table.columns), ["长度"])
    class_col = _find_column(list(table.columns), ["通行优先级"])
    pavement_col = _find_column(list(table.columns), ["路面类型"])
    slope_col = _find_column(list(table.columns), ["地形坡度"])
    pipe_col = _find_column(list(table.columns), ["管径"])
    drainage_col = _find_column(list(table.columns), ["设计排水能力"])
    elevation_col = _find_column(list(table.columns), ["最低标高"])

    roads = pd.DataFrame(
        {
            "edge_id": table[road_col].astype(str).str.strip(),
            "length_m": pd.to_numeric(table[length_col], errors="coerce"),
            "road_class": table[class_col].astype(str).str.strip(),
            "pavement_type": table[pavement_col].astype(str).str.strip(),
            "slope_per_mille": pd.to_numeric(table[slope_col], errors="coerce"),
            "pipe_diameter_mm": pd.to_numeric(table[pipe_col], errors="coerce"),
            "drainage_capacity_lps": pd.to_numeric(table[drainage_col], errors="coerce"),
            "min_elevation_m": pd.to_numeric(table[elevation_col], errors="coerce"),
        }
    )
    roads = roads[roads["edge_id"].str.match(r"^L\d+$", na=False)].copy()
    roads["edge_order"] = roads["edge_id"].str.extract(r"(\d+)")[0].astype(int)
    roads = roads.sort_values("edge_order").drop(columns=["edge_order"]).reset_index(drop=True)
    return roads


def build_topology_edges(road_df: pd.DataFrame) -> pd.DataFrame:
    edges = road_df.copy()
    starts: list[str] = []
    ends: list[str] = []
    for edge_id in edges["edge_id"]:
        if edge_id not in EDGE_NODE_MAP:
            raise ValueError(f"未定义拓扑连接关系: {edge_id}")
        start_node, end_node = EDGE_NODE_MAP[edge_id]
        starts.append(start_node)
        ends.append(end_node)
    edges.insert(1, "start_node", starts)
    edges.insert(2, "end_node", ends)
    edges["edge_id"] = pd.Categorical(edges["edge_id"], categories=EDGE_ORDER, ordered=True)
    edges = edges.sort_values("edge_id").reset_index(drop=True)
    edges["edge_id"] = edges["edge_id"].astype(str)
    return edges[
        [
            "edge_id",
            "start_node",
            "end_node",
            "length_m",
            "road_class",
            "pavement_type",
            "slope_per_mille",
            "pipe_diameter_mm",
            "drainage_capacity_lps",
            "min_elevation_m",
        ]
    ]


def build_topology_nodes() -> pd.DataFrame:
    rows = [
        ("N1", "origin_candidate", "高地边缘节点，可作为疏散起点候选"),
        ("N2", "transfer", "高地连接节点，连接支路与主干路"),
        ("N3", "transfer", "中部换乘节点，连接多条替代路径"),
        ("N4", "normal", "低洼连接节点，靠近 L5 低洼瓶颈"),
        ("N5", "transfer", "主干路连接节点，连接 L1 与 L3"),
        ("N6", "transfer", "主干路换乘节点，连接 L3、L6、L8"),
        ("N7", "destination_candidate", "低洼出口节点，可作为疏散出口或风险出口节点"),
    ]
    return pd.DataFrame(rows, columns=["node_id", "node_type", "description"])


def build_adjacency_table(edges_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in edges_df.iterrows():
        rows.append(
            {
                "node_id": row["start_node"],
                "neighbor_node": row["end_node"],
                "edge_id": row["edge_id"],
                "length_m": row["length_m"],
            }
        )
        rows.append(
            {
                "node_id": row["end_node"],
                "neighbor_node": row["start_node"],
                "edge_id": row["edge_id"],
                "length_m": row["length_m"],
            }
        )
    adjacency = pd.DataFrame(rows)
    adjacency["node_id"] = pd.Categorical(adjacency["node_id"], categories=NODE_ORDER, ordered=True)
    adjacency = adjacency.sort_values(["node_id", "neighbor_node", "edge_id"]).reset_index(drop=True)
    adjacency["node_id"] = adjacency["node_id"].astype(str)
    return adjacency


def check_graph_connected(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> bool:
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(nodes_df["node_id"])
        graph.add_edges_from(edges_df[["start_node", "end_node"]].itertuples(index=False, name=None))
        return bool(nx.is_connected(graph))
    except Exception:
        adjacency: dict[str, set[str]] = {node: set() for node in nodes_df["node_id"]}
        for _, row in edges_df.iterrows():
            adjacency[row["start_node"]].add(row["end_node"])
            adjacency[row["end_node"]].add(row["start_node"])
        if not adjacency:
            return False
        start = next(iter(adjacency))
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return len(visited) == len(adjacency)


def load_q1_risk_optional(project_dir: Path) -> tuple[pd.DataFrame, bool]:
    q1_path = project_dir / "results" / "q1" / "q1_risk_summary.csv"
    if not q1_path.exists():
        return pd.DataFrame(), False
    risk = pd.read_csv(q1_path)
    rename_map = {
        "路段编号": "edge_id",
        "road_id": "edge_id",
        "risk_index": "risk_index",
        "综合风险指数": "risk_index",
        "risk_rank": "risk_rank",
        "风险排名": "risk_rank",
        "risk_level": "risk_level",
    }
    risk = risk.rename(columns={col: rename_map[col] for col in risk.columns if col in rename_map})
    keep = [col for col in ["edge_id", "risk_index", "risk_rank", "risk_level"] if col in risk.columns]
    if "edge_id" not in keep:
        return pd.DataFrame(), False
    risk = risk[keep].copy()
    if "risk_rank" in risk.columns:
        risk = risk.sort_values("risk_rank")
    return risk.reset_index(drop=True), True


def _build_graph_summary(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    is_connected: bool,
    q1_risk_df: pd.DataFrame,
    q1_loaded: bool,
) -> pd.DataFrame:
    main_roads = edges_df.loc[edges_df["road_class"].astype(str).str.contains("主干路"), "edge_id"].tolist()
    lowest_row = edges_df.loc[edges_df["min_elevation_m"].idxmin()]
    if q1_loaded and "edge_id" in q1_risk_df.columns:
        high_risk_edges = q1_risk_df.head(3)["edge_id"].astype(str).tolist()
        high_risk_text = ",".join(high_risk_edges)
    else:
        high_risk_text = "未读取第一问风险结果"
    notes = (
        "基于路段等级、最低标高、风险等级和连通性原则构建的简化交通拓扑网络；"
        "该拓扑服务第三问动态疏散路径优化，不代表真实测绘拓扑。"
    )
    if not is_connected:
        notes = "WARNING: 图不连通；" + notes
    return pd.DataFrame(
        [
            {
                "node_count": len(nodes_df),
                "edge_count": len(edges_df),
                "is_connected": bool(is_connected),
                "main_roads": ",".join(main_roads),
                "high_risk_edges_if_available": high_risk_text,
                "lowest_elevation_edge": lowest_row["edge_id"],
                "notes": notes,
            }
        ]
    )


def save_topology_outputs(
    output_dir: Path,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    adjacency_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(output_dir / "topology_edges.csv", index=False, encoding="utf-8-sig")
    nodes_df.to_csv(output_dir / "topology_nodes.csv", index=False, encoding="utf-8-sig")
    adjacency_df.to_csv(output_dir / "topology_adjacency.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "topology_graph_summary.csv", index=False, encoding="utf-8-sig")


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
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, True, ""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, False, ""


def plot_topology_graph(output_dir: Path, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> str:
    plt, use_chinese, reason = _setup_matplotlib()
    if plt is None:
        return f"拓扑图未生成：{reason}"

    positions = {
        "N1": (0.0, 2.0),
        "N2": (1.0, 2.0),
        "N3": (2.0, 1.5),
        "N4": (3.1, 0.7),
        "N5": (1.5, 3.0),
        "N6": (2.7, 2.5),
        "N7": (4.0, 1.1),
    }
    class_colors = {"主干路": "#4C72B0", "次干路": "#DD8452", "支路": "#55A868"}
    class_widths = {"主干路": 3.0, "次干路": 2.3, "支路": 1.8}

    def text(cn: str, en: str) -> str:
        return cn if use_chinese else en

    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for _, edge in edges_df.iterrows():
        start = edge["start_node"]
        end = edge["end_node"]
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        road_class = str(edge["road_class"])
        color = class_colors.get(road_class, "#777777")
        width = class_widths.get(road_class, 2.0)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, solid_capstyle="round")
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.08, edge["edge_id"], ha="center", va="center", fontsize=10, fontweight="bold")

    for _, node in nodes_df.iterrows():
        x, y = positions[node["node_id"]]
        node_type = node["node_type"]
        face = "#F2F2F2"
        if node_type == "origin_candidate":
            face = "#C7E9C0"
        elif node_type == "destination_candidate":
            face = "#FDD0A2"
        elif node_type == "transfer":
            face = "#C6DBEF"
        ax.scatter(x, y, s=520, color=face, edgecolor="#333333", zorder=3)
        ax.text(x, y, node["node_id"], ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

    for road_class, color in class_colors.items():
        label = road_class if use_chinese else {"主干路": "arterial", "次干路": "secondary", "支路": "branch"}[road_class]
        ax.plot([], [], color=color, linewidth=class_widths[road_class], label=label)

    ax.text(3.5, 0.35, text("低洼出口方向", "Low-lying exit"), color="#C44E52", ha="center")
    ax.text(0.1, 2.28, text("高地边缘", "Highland edge"), color="#2E7D32", ha="left")
    ax.set_title(text("简化交通拓扑网络", "Simplified Traffic Topology"))
    ax.legend(loc="upper right")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "topology_graph.png")
    plt.close(fig)
    return "拓扑图已生成" if use_chinese else "拓扑图已生成；未检测到中文字体，图题使用英文。"


def write_topology_notes(
    output_dir: Path,
    input_file: Path,
    is_connected: bool,
    q1_risk_df: pd.DataFrame,
    q1_loaded: bool,
    plot_status: str,
) -> None:
    if q1_loaded and "edge_id" in q1_risk_df.columns:
        top_edges = q1_risk_df.head(3)["edge_id"].astype(str).tolist()
        risk_note = f"已读取第一问风险结果，风险排序靠前路段为 {'、'.join(top_edges)}。结合第一问路段运行风险结果，L5、L2、L8 为重点薄弱路段。"
    else:
        risk_note = "暂未读取第一问风险结果，拓扑仍按附表1和简化连通性原则生成。"

    warning = "" if is_connected else "\n警告：当前简化拓扑图不连通，需要在第三问路径规划前修正。\n"
    notes = f"""简化路网拓扑构建说明

输入文件：{input_file}

一、拓扑性质
本拓扑是基于路段等级、最低标高、风险等级和连通性原则构建的简化交通拓扑网络，不是真实测绘拓扑，也不代表真实节点坐标、坡向、节点标高或管网连接关系。

二、构建依据
1. L1、L3、L6 为主干路，作为交通骨架；
2. L5 为最低标高路段，作为低洼瓶颈路段；
3. L2、L8 靠近 L5，作为次高风险通道；
4. L7 标高最高，设置在边缘高地位置，作为相对安全支路；
5. 整个图按连通性原则构建，并保留多条替代路径，便于第三问动态绕行优化。

三、与第一问和第三问的关系
第一问仍以路段级积水演化模型为主。本拓扑在第一问中只作为潜在汇流方向和空间风险解释，不作为强水动力约束，也不强行加入复杂路段间水流交换。第三问可直接读取 topology_edges.csv，并按 edge_id 将 q1_depth_5min.csv、q1_capacity_5min.csv、q1_safety_5min.csv 挂载到拓扑边上。

四、第一问风险衔接
{risk_note}

五、连通性与输出
图连通性检查结果：{"连通" if is_connected else "不连通"}。
{warning}拓扑图输出状态：{plot_status}
"""
    (output_dir / "topology_notes.txt").write_text(notes, encoding="utf-8")


def main() -> None:
    configure_console()
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "results" / "topology"
    input_file = find_input_file()

    road_df = load_road_data(input_file)
    edges_df = build_topology_edges(road_df)
    nodes_df = build_topology_nodes()
    adjacency_df = build_adjacency_table(edges_df)
    is_connected = check_graph_connected(nodes_df, edges_df)
    q1_risk_df, q1_loaded = load_q1_risk_optional(project_dir)
    summary_df = _build_graph_summary(nodes_df, edges_df, is_connected, q1_risk_df, q1_loaded)

    save_topology_outputs(output_dir, edges_df, nodes_df, adjacency_df, summary_df)
    plot_status = plot_topology_graph(output_dir, edges_df, nodes_df)
    write_topology_notes(output_dir, input_file, is_connected, q1_risk_df, q1_loaded, plot_status)

    if not is_connected:
        print("WARNING: 简化拓扑图不连通，请检查边-节点设定。")
    print(f"输出目录: {output_dir}")
    print(f"节点数: {len(nodes_df)}")
    print(f"边数: {len(edges_df)}")
    print(f"是否连通: {is_connected}")
    print(f"是否读取到第一问风险结果: {q1_loaded}")
    print(f"拓扑图是否生成: {(output_dir / 'topology_graph.png').exists()}")
    print(plot_status)


if __name__ == "__main__":
    main()

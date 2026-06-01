from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POSITIVE_KEYWORDS = [
    "达标率",
    "覆盖率",
    "连通度",
    "冗余度",
    "密度",
    "容积",
    "能力",
    "比例",
    "效率",
]
NEGATIVE_KEYWORDS = [
    "时间",
    "半衰期",
    "损失",
    "风险",
    "深度",
    "延误",
    "成本",
    "低洼",
]
LAYER_ORDER = ["抵御能力", "恢复能力", "适应能力"]


def classify_resilience(score: float) -> str:
    if score < 0.4:
        return "低韧性"
    if score < 0.6:
        return "中低韧性"
    if score < 0.8:
        return "中高韧性"
    return "高韧性"


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


def load_indicator_data(input_file: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(input_file)
    sheet_name = _find_sheet_name(xl.sheet_names, 5)
    raw = pd.read_excel(input_file, sheet_name=sheet_name, header=None)
    raw = raw.replace(r"^\s*$", np.nan, regex=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    raw = raw.reset_index(drop=True)

    header_row = _find_header_row(raw, ["一级指标", "二级指标", "权重"])
    headers = [_cell_text(v) or f"unnamed_{i}" for i, v in enumerate(raw.iloc[header_row].tolist())]
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = headers
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

    layer_col = _find_column(list(table.columns), ["一级指标"])
    indicator_col = _find_column(list(table.columns), ["二级指标"])
    unit_col = _find_column(list(table.columns), ["计量单位"])
    current_col = _find_column(list(table.columns), ["现状值"])
    ideal_col = _find_column(list(table.columns), ["理想"])
    weight_col = _find_column(list(table.columns), ["权重"])

    indicator_df = pd.DataFrame(
        {
            "layer": table[layer_col],
            "indicator": table[indicator_col],
            "unit": table[unit_col],
            "current_value": pd.to_numeric(table[current_col], errors="coerce"),
            "ideal_value": pd.to_numeric(table[ideal_col], errors="coerce"),
            "weight": pd.to_numeric(table[weight_col], errors="coerce"),
        }
    )
    indicator_df["layer"] = indicator_df["layer"].ffill().astype(str).str.strip()
    indicator_df["indicator"] = indicator_df["indicator"].astype(str).str.strip()
    indicator_df["unit"] = indicator_df["unit"].astype(str).str.strip()
    indicator_df = indicator_df.dropna(subset=["indicator", "current_value", "ideal_value", "weight"])
    indicator_df = indicator_df[indicator_df["indicator"].ne("")]
    indicator_df = indicator_df.reset_index(drop=True)
    return indicator_df


def infer_indicator_direction(indicator_name: str) -> str:
    name = str(indicator_name)
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in name:
            return "negative"
    for keyword in POSITIVE_KEYWORDS:
        if keyword in name:
            return "positive"
    return "positive"


def normalize_indicators(indicator_df: pd.DataFrame) -> pd.DataFrame:
    df = indicator_df.copy()
    df["direction"] = df["indicator"].apply(infer_indicator_direction)
    weight_sum = df["weight"].sum()
    if weight_sum <= 0 or pd.isna(weight_sum):
        raise ValueError("附表5权重总和无效，无法计算韧性评分。")
    df["normalized_weight"] = df["weight"] / weight_sum

    scores: list[float] = []
    for _, row in df.iterrows():
        current = float(row["current_value"])
        ideal = float(row["ideal_value"])
        unit = str(row.get("unit", ""))
        indicator = str(row.get("indicator", ""))
        if row["direction"] == "positive":
            score = 1.0 if ideal == 0 and current >= ideal else current / ideal if ideal != 0 else 0.0
        else:
            if ideal == 0:
                if "%" in unit or "占比" in indicator:
                    score = 1.0 - current / 100.0
                elif current == 0:
                    score = 1.0
                else:
                    score = 1.0 / (1.0 + current)
            else:
                score = 1.0 if current == 0 else ideal / current
        scores.append(float(np.clip(score, 0.0, 1.0)))

    df["normalized_score"] = scores
    df["deviation"] = 1.0 - df["normalized_score"]
    df["weighted_score"] = df["normalized_weight"] * df["normalized_score"]
    return df


def compute_resilience_scores(normalized_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall_score = float(normalized_df["weighted_score"].sum())
    rows.append(
        {
            "score_type": "综合韧性",
            "layer": "总体",
            "score": overall_score,
            "weight_sum": float(normalized_df["normalized_weight"].sum()),
            "indicator_count": int(len(normalized_df)),
        }
    )

    for layer in LAYER_ORDER:
        layer_df = normalized_df[normalized_df["layer"] == layer].copy()
        if layer_df.empty:
            continue
        layer_weight_sum = layer_df["normalized_weight"].sum()
        layer_score = (
            float((layer_df["normalized_weight"] * layer_df["normalized_score"]).sum() / layer_weight_sum)
            if layer_weight_sum > 0
            else float("nan")
        )
        rows.append(
            {
                "score_type": "层级韧性",
                "layer": layer,
                "score": layer_score,
                "weight_sum": float(layer_weight_sum),
                "indicator_count": int(len(layer_df)),
            }
        )

    score_df = pd.DataFrame(rows)
    score_df["resilience_level"] = score_df["score"].apply(classify_resilience)
    return score_df


def compute_obstacle_degree(normalized_df: pd.DataFrame) -> pd.DataFrame:
    obstacle = normalized_df.copy()
    obstacle["obstacle_numerator"] = obstacle["normalized_weight"] * obstacle["deviation"]
    denom = obstacle["obstacle_numerator"].sum()
    obstacle["obstacle_degree"] = obstacle["obstacle_numerator"] / denom if denom > 0 else 0.0
    obstacle = obstacle.sort_values("obstacle_degree", ascending=False).reset_index(drop=True)
    obstacle["obstacle_rank"] = np.arange(1, len(obstacle) + 1)
    cols = [
        "obstacle_rank",
        "layer",
        "indicator",
        "unit",
        "direction",
        "current_value",
        "ideal_value",
        "weight",
        "normalized_weight",
        "normalized_score",
        "deviation",
        "obstacle_degree",
    ]
    return obstacle[cols]


def compute_weight_sensitivity(normalized_df: pd.DataFrame) -> pd.DataFrame:
    base_weights = normalized_df["normalized_weight"].to_numpy(dtype=float)
    scores = normalized_df["normalized_score"].to_numpy(dtype=float)
    base_score = float(np.sum(base_weights * scores))
    records: list[dict[str, Any]] = []

    for idx, row in normalized_df.reset_index(drop=True).iterrows():
        for factor, label in [(0.9, "weight_down_10pct"), (1.1, "weight_up_10pct")]:
            adjusted_weights = base_weights.copy()
            adjusted_weights[idx] *= factor
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            adjusted_score = float(np.sum(adjusted_weights * scores))
            records.append(
                {
                    "layer": row["layer"],
                    "indicator": row["indicator"],
                    "scenario": label,
                    "weight_factor": factor,
                    "original_weight": row["normalized_weight"],
                    "adjusted_weight": adjusted_weights[idx],
                    "base_score": base_score,
                    "adjusted_score": adjusted_score,
                    "score_delta": adjusted_score - base_score,
                    "abs_score_delta": abs(adjusted_score - base_score),
                }
            )

    sensitivity = pd.DataFrame(records)
    return sensitivity.sort_values("abs_score_delta", ascending=False).reset_index(drop=True)


def _diagnosis_direction(indicator: str) -> str:
    mapping = [
        ("积水消退半衰期", "排水管网扩径、增设调蓄池、提升泵站排涝效率"),
        ("绿色调蓄容积", "建设调蓄池、下沉式绿地、透水路面和雨水花园"),
        ("应急响应时间", "建设智慧预警系统、优化应急联动流程和抢险调度"),
        ("疏散点位密度", "增设疏散点、完善避险空间布局和引导标识"),
        ("排水能力达标率", "推进排水管网改造、清淤维护和易涝节点提标"),
        ("低洼路段占比", "低洼路段抬升改造、局部排水改造和交通绕行组织"),
        ("路网连通度", "打通断点路段、优化路网微循环和应急通道"),
        ("抢修覆盖效率", "增加抢修布点、优化物资储备和抢险队伍调度"),
        ("预警覆盖率", "扩展监测预警终端、接入多源雨水情数据"),
    ]
    for keyword, direction in mapping:
        if keyword in indicator:
            return direction
    return "结合指标短板开展针对性工程改造与管理优化"


def build_diagnosis_table(obstacle_df: pd.DataFrame) -> pd.DataFrame:
    diagnosis = obstacle_df.head(5).copy()
    diagnosis["governance_direction"] = diagnosis["indicator"].apply(_diagnosis_direction)
    diagnosis["priority"] = diagnosis["obstacle_rank"].apply(lambda rank: f"P{int(rank)}")
    return diagnosis[
        [
            "priority",
            "obstacle_rank",
            "layer",
            "indicator",
            "normalized_score",
            "deviation",
            "obstacle_degree",
            "governance_direction",
        ]
    ]


def load_q1_road_risk_optional(project_dir: Path) -> tuple[pd.DataFrame, bool]:
    q1_path = project_dir / "results" / "q1" / "q1_risk_summary.csv"
    if not q1_path.exists():
        return (
            pd.DataFrame(
                [
                    {
                        "status": "未读取第一问结果",
                        "message": "results/q1/q1_risk_summary.csv 不存在，第二问仅完成系统层面韧性评价。",
                    }
                ]
            ),
            False,
        )

    risk = pd.read_csv(q1_path)
    rename_candidates = {
        "road_id": "road_id",
        "路段编号": "road_id",
        "risk_index": "risk_index",
        "综合风险指数": "risk_index",
        "risk_rank": "risk_rank",
        "风险排名": "risk_rank",
        "risk_level": "risk_level",
    }
    risk = risk.rename(columns={col: rename_candidates[col] for col in risk.columns if col in rename_candidates})
    keep = [col for col in ["road_id", "risk_index", "risk_rank", "risk_level"] if col in risk.columns]
    if not {"road_id", "risk_rank"}.issubset(set(keep)):
        return (
            pd.DataFrame(
                [
                    {
                        "status": "第一问结果格式不完整",
                        "message": "q1_risk_summary.csv 缺少 road_id 或 risk_rank，未用于薄弱路段识别。",
                    }
                ]
            ),
            False,
        )
    risk = risk[keep].copy()
    risk = risk.sort_values("risk_rank").reset_index(drop=True)
    return risk, True


def identify_weak_links(obstacle_df: pd.DataFrame, weak_roads_df: pd.DataFrame, q1_loaded: bool) -> pd.DataFrame:
    top = obstacle_df.head(5).copy()
    top["weak_link_type"] = "韧性障碍指标"
    top["diagnosis"] = top.apply(
        lambda row: (
            f"{row['indicator']} 归一化得分为 {row['normalized_score']:.3f}，"
            f"偏离理想水平 {row['deviation']:.3f}，障碍度排名第 {int(row['obstacle_rank'])}。"
        ),
        axis=1,
    )
    weak_links = top[
        [
            "weak_link_type",
            "obstacle_rank",
            "layer",
            "indicator",
            "normalized_score",
            "deviation",
            "obstacle_degree",
            "diagnosis",
        ]
    ].copy()

    if q1_loaded and "road_id" in weak_roads_df.columns:
        roads = ", ".join(weak_roads_df.head(3)["road_id"].astype(str).tolist())
        weak_links.loc[len(weak_links)] = {
            "weak_link_type": "路段运行风险辅助信息",
            "obstacle_rank": np.nan,
            "layer": "路网运行",
            "indicator": "第一问高风险路段",
            "normalized_score": np.nan,
            "deviation": np.nan,
            "obstacle_degree": np.nan,
            "diagnosis": f"结合第一问路段运行风险结果，{roads} 可作为重点关注薄弱路段。",
        }
    return weak_links


def save_outputs(
    output_dir: Path,
    normalized_df: pd.DataFrame,
    score_df: pd.DataFrame,
    obstacle_df: pd.DataFrame,
    weak_links_df: pd.DataFrame,
    weak_roads_df: pd.DataFrame,
    weight_sensitivity_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(output_dir / "q2_indicator_normalized.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(output_dir / "q2_resilience_score.csv", index=False, encoding="utf-8-sig")
    obstacle_df.to_csv(output_dir / "q2_obstacle_degree.csv", index=False, encoding="utf-8-sig")
    weak_links_df.to_csv(output_dir / "q2_weak_links.csv", index=False, encoding="utf-8-sig")
    weak_roads_df.to_csv(output_dir / "q2_weak_roads.csv", index=False, encoding="utf-8-sig")
    weight_sensitivity_df.to_csv(output_dir / "q2_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    diagnosis_df.to_csv(output_dir / "q2_diagnosis_table.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_dir / "q2_results.xlsx", engine="openpyxl") as writer:
        normalized_df.to_excel(writer, sheet_name="indicator_normalized", index=False)
        score_df.to_excel(writer, sheet_name="resilience_score", index=False)
        obstacle_df.to_excel(writer, sheet_name="obstacle_degree", index=False)
        weak_links_df.to_excel(writer, sheet_name="weak_links", index=False)
        weak_roads_df.to_excel(writer, sheet_name="weak_roads", index=False)
        weight_sensitivity_df.to_excel(writer, sheet_name="weight_sensitivity", index=False)
        diagnosis_df.to_excel(writer, sheet_name="diagnosis_table", index=False)


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


def plot_results(
    output_dir: Path,
    normalized_df: pd.DataFrame,
    score_df: pd.DataFrame,
    obstacle_df: pd.DataFrame,
    weight_sensitivity_df: pd.DataFrame,
) -> str:
    plt, use_chinese, reason = _setup_matplotlib()
    if plt is None:
        return f"图片未生成：{reason}"

    def text(cn: str, en: str) -> str:
        return cn if use_chinese else en

    layer_scores = score_df[score_df["score_type"] == "层级韧性"].copy()

    labels = layer_scores["layer"].tolist()
    values = layer_scores["score"].astype(float).tolist()
    if labels:
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]
        fig = plt.figure(figsize=(7, 7), dpi=160)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles_closed, values_closed, linewidth=2, color="#4C72B0")
        ax.fill(angles_closed, values_closed, color="#4C72B0", alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels if use_chinese else ["Resistance", "Recovery", "Adaptation"][: len(labels)])
        ax.set_ylim(0, 1)
        ax.set_title(text("城市内涝韧性三层能力雷达图", "Urban Waterlogging Resilience Radar"))
        fig.tight_layout()
        fig.savefig(output_dir / "resilience_radar.png")
        plt.close(fig)

    top_obstacle = obstacle_df.head(8).iloc[::-1]
    plt.figure(figsize=(9, 5.5), dpi=160)
    y_labels = top_obstacle["indicator"].tolist() if use_chinese else [f"I{int(r)}" for r in top_obstacle["obstacle_rank"]]
    plt.barh(y_labels, top_obstacle["obstacle_degree"], color="#DD8452")
    plt.xlabel(text("障碍度", "Obstacle degree"))
    plt.title(text("韧性障碍度排名", "Obstacle Degree Ranking"))
    plt.tight_layout()
    plt.savefig(output_dir / "obstacle_degree_bar.png")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=160)
    x_labels = layer_scores["layer"].tolist() if use_chinese else ["Resistance", "Recovery", "Adaptation"][: len(layer_scores)]
    plt.bar(x_labels, layer_scores["score"], color=["#4C72B0", "#55A868", "#C44E52"][: len(layer_scores)])
    plt.ylim(0, 1)
    plt.ylabel(text("层级韧性得分", "Layer score"))
    plt.title(text("抵御-恢复-适应能力得分", "Resistance-Recovery-Adaptation Scores"))
    plt.tight_layout()
    plt.savefig(output_dir / "layer_score_bar.png")
    plt.close()

    sensitivity_plot = (
        weight_sensitivity_df.groupby("indicator", as_index=False)["abs_score_delta"]
        .max()
        .sort_values("abs_score_delta", ascending=True)
    )
    plt.figure(figsize=(9, 5.5), dpi=160)
    y_labels = (
        sensitivity_plot["indicator"].tolist()
        if use_chinese
        else [f"I{i + 1}" for i in range(len(sensitivity_plot))]
    )
    plt.barh(y_labels, sensitivity_plot["abs_score_delta"], color="#8172B3")
    plt.xlabel(text("综合韧性得分最大变化幅度", "Max absolute score delta"))
    plt.title(text("指标权重上下浮动10%的敏感性分析", "Weight Sensitivity Analysis (+/-10%)"))
    plt.tight_layout()
    plt.savefig(output_dir / "weight_sensitivity_bar.png")
    plt.close()

    return "图片已生成" if use_chinese else "图片已生成；未检测到中文字体，图题使用英文。"


def write_model_notes(
    output_dir: Path,
    score_df: pd.DataFrame,
    obstacle_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    weak_roads_df: pd.DataFrame,
    q1_loaded: bool,
    plot_status: str,
) -> None:
    overall = score_df.loc[score_df["score_type"] == "综合韧性", "score"].iloc[0]
    overall_level = score_df.loc[score_df["score_type"] == "综合韧性", "resilience_level"].iloc[0]
    layer_df = score_df[score_df["score_type"] == "层级韧性"].copy()
    layer_text = "；".join(
        f"{row['layer']}={row['score']:.4f}（{row['resilience_level']}）"
        for _, row in layer_df.iterrows()
    )
    top5 = obstacle_df.head(5)
    top5_text = "；".join(
        f"第{int(row['obstacle_rank'])}位 {row['indicator']}（障碍度 {row['obstacle_degree']:.4f}）"
        for _, row in top5.iterrows()
    )
    weak_causes = "、".join(top5["indicator"].astype(str).tolist())
    diagnosis_text = "；".join(
        f"{row['indicator']}：{row['governance_direction']}" for _, row in diagnosis_df.iterrows()
    )

    if q1_loaded and "road_id" in weak_roads_df.columns:
        top_roads = weak_roads_df.head(3)["road_id"].astype(str).tolist()
        road_note = f"结合第一问路段运行风险结果，{ '、'.join(top_roads) }可作为重点关注薄弱路段。"
        if set(["L5", "L2", "L8"]).issubset(set(top_roads)):
            road_note = "结合第一问路段运行风险结果，L5、L2、L8可作为重点关注薄弱路段。"
        q4_note = "第四问改造方案可将系统短板指标与 L5、L2、L8 等高风险路段叠加，形成“指标优先级 + 空间优先级”的治理排序。"
    else:
        road_note = "未读取第一问结果，第二问仅完成系统层面韧性评价。"
        q4_note = "第四问改造方案可优先依据障碍度较高的系统指标确定治理方向，再结合后续路段风险结果补充空间落点。"

    notes = f"""第二问模型说明与论文可用结论

一、模型路线
第二问采用“题给权重 + 指标归一化 + 综合评价 + 障碍度分析”的模型路线。主计算仅依赖附表5，不强依赖第一问具体积水演化算法。

二、综合韧性评价结果
综合韧性得分 R = {overall:.4f}，韧性等级为“{overall_level}”。该得分由所有指标按题给权重归一化后加权得到，得分越接近 1 表明现状越接近理想韧性状态。

三、三层韧性得分
三层能力得分为：{layer_text}。这些分值反映抵御能力、恢复能力和适应能力在层内归一化权重下的相对表现。

四、障碍度分析
障碍度排名前五的指标为：{top5_text}。薄弱环节主要由 {weak_causes} 等指标造成，说明后续韧性提升应优先围绕这些短板展开。

五、治理方向诊断
障碍度前五指标对应治理方向为：{diagnosis_text}。

六、薄弱路段辅助识别
{road_note}

七、对第四问改造方案的支撑
第二问结果将服务于第四问改造方案设计：障碍度较高的系统指标用于确定治理优先方向，第一问高风险路段则用于辅助确定空间治理重点。{q4_note}

八、输出状态
{plot_status}
"""
    (output_dir / "q2_model_notes.txt").write_text(notes, encoding="utf-8")


def print_summary(input_file: Path, output_dir: Path, score_df: pd.DataFrame, obstacle_df: pd.DataFrame, q1_loaded: bool) -> None:
    overall = score_df.loc[score_df["score_type"] == "综合韧性", "score"].iloc[0]
    print(f"输入文件路径: {input_file}")
    print(f"输出目录: {output_dir}")
    overall_level = score_df.loc[score_df["score_type"] == "综合韧性", "resilience_level"].iloc[0]
    print(f"综合韧性得分: {overall:.4f}（{overall_level}）")
    print("\n三层韧性得分:")
    for _, row in score_df[score_df["score_type"] == "层级韧性"].iterrows():
        print(f"  {row['layer']}: {row['score']:.4f}（{row['resilience_level']}）")
    print("\n障碍度排名前五:")
    for _, row in obstacle_df.head(5).iterrows():
        print(f"  第{int(row['obstacle_rank'])}名 {row['indicator']}: {row['obstacle_degree']:.4f}")
    print("\n第一问路段风险辅助信息:", "已读取" if q1_loaded else "未读取")


def validate_outputs(output_dir: Path) -> None:
    report_path = output_dir / "q2_validation_report.txt"
    lines = ["Q2 validation report", f"Output directory: {output_dir}", ""]
    warnings: list[str] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "WARNING"
        message = f"[{status}] {name}"
        if detail:
            message += f" - {detail}"
        lines.append(message)
        if not passed:
            warnings.append(message)

    required_files = [
        "q2_indicator_normalized.csv",
        "q2_resilience_score.csv",
        "q2_obstacle_degree.csv",
        "q2_weak_links.csv",
        "q2_weak_roads.csv",
        "q2_weight_sensitivity.csv",
        "q2_diagnosis_table.csv",
        "q2_results.xlsx",
        "resilience_radar.png",
        "obstacle_degree_bar.png",
        "layer_score_bar.png",
        "weight_sensitivity_bar.png",
        "q2_model_notes.txt",
    ]
    missing = [name for name in required_files if not (output_dir / name).exists()]
    add_check("主要输出文件均已生成", not missing, f"missing={missing}" if missing else "all present")

    try:
        normalized = pd.read_csv(output_dir / "q2_indicator_normalized.csv")
        scores = pd.read_csv(output_dir / "q2_resilience_score.csv")
        obstacle = pd.read_csv(output_dir / "q2_obstacle_degree.csv")
    except Exception as exc:
        add_check("可读取核心 CSV 输出", False, str(exc))
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for warning in warnings:
            print(warning)
        print(f"Validation report saved to: {report_path}")
        return

    lowland = normalized.loc[normalized["indicator"].astype(str) == "低洼路段占比"]
    lowland_score = float(lowland.iloc[0]["normalized_score"]) if not lowland.empty else float("nan")
    add_check(
        "低洼路段占比 normalized_score 为 0.76",
        np.isclose(lowland_score, 0.76, atol=1e-6),
        f"actual={lowland_score:.6f}" if np.isfinite(lowland_score) else "missing",
    )

    def score_value(score_type: str, layer: str) -> float:
        row = scores.loc[(scores["score_type"] == score_type) & (scores["layer"] == layer)]
        return float(row.iloc[0]["score"]) if not row.empty else float("nan")

    checks = [
        ("综合韧性得分约 0.5713", score_value("综合韧性", "总体"), 0.5713),
        ("抵御能力约 0.7166", score_value("层级韧性", "抵御能力"), 0.7166),
        ("恢复能力约 0.4910", score_value("层级韧性", "恢复能力"), 0.4910),
        ("适应能力约 0.4958", score_value("层级韧性", "适应能力"), 0.4958),
    ]
    for name, actual, expected in checks:
        add_check(
            name,
            np.isclose(actual, expected, atol=5e-4),
            f"actual={actual:.6f}" if np.isfinite(actual) else "missing",
        )

    top_obstacle = str(obstacle.iloc[0]["indicator"]) if not obstacle.empty else ""
    add_check(
        "障碍度第一为积水消退半衰期",
        top_obstacle == "积水消退半衰期",
        f"actual={top_obstacle}",
    )

    if (output_dir / "q2_results.xlsx").exists():
        try:
            sheets = pd.ExcelFile(output_dir / "q2_results.xlsx").sheet_names
            expected_sheets = {
                "indicator_normalized",
                "resilience_score",
                "obstacle_degree",
                "weak_links",
                "weak_roads",
                "weight_sensitivity",
                "diagnosis_table",
            }
            add_check(
                "q2_results.xlsx 包含增强输出 sheet",
                expected_sheets.issubset(set(sheets)),
                f"sheets={sheets}",
            )
        except Exception as exc:
            add_check("q2_results.xlsx 可打开", False, str(exc))

    lines.append("")
    lines.append(f"Total warnings: {len(warnings)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for warning in warnings:
        print(warning)
    print(f"Validation report saved to: {report_path}")


def main() -> None:
    configure_console()
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "results" / "q2"
    input_file = find_input_file()

    indicator_df = load_indicator_data(input_file)
    normalized_df = normalize_indicators(indicator_df)
    score_df = compute_resilience_scores(normalized_df)
    obstacle_df = compute_obstacle_degree(normalized_df)
    weight_sensitivity_df = compute_weight_sensitivity(normalized_df)
    diagnosis_df = build_diagnosis_table(obstacle_df)
    weak_roads_df, q1_loaded = load_q1_road_risk_optional(project_dir)
    weak_links_df = identify_weak_links(obstacle_df, weak_roads_df, q1_loaded)

    save_outputs(
        output_dir,
        normalized_df,
        score_df,
        obstacle_df,
        weak_links_df,
        weak_roads_df,
        weight_sensitivity_df,
        diagnosis_df,
    )
    plot_status = plot_results(output_dir, normalized_df, score_df, obstacle_df, weight_sensitivity_df)
    write_model_notes(output_dir, score_df, obstacle_df, diagnosis_df, weak_roads_df, q1_loaded, plot_status)
    print_summary(input_file, output_dir, score_df, obstacle_df, q1_loaded)
    print(f"\n{plot_status}")
    print("第二问结果文件已保存。")
    validate_outputs(output_dir)


if __name__ == "__main__":
    main()

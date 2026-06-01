from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl


DATA_FILE = Path("traffic_tables.xlsx")
N = 60


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_series(sheet_index: int) -> np.ndarray:
    wb = openpyxl.load_workbook(DATA_FILE, data_only=True, read_only=True)
    ws = wb.worksheets[sheet_index]
    values = []
    for row in ws.iter_rows(min_row=2, max_row=61, min_col=3, max_col=3, values_only=True):
        values.append(float(row[0]))
    if len(values) != N:
        raise ValueError(f"{ws.title} should contain {N} observations, got {len(values)}")
    return np.array(values, dtype=float)


def shifted_time_grid() -> np.ndarray:
    # Branches 1 and 2 need 3 minutes to arrive at A3. The table interval is 2 minutes.
    return np.arange(N, dtype=float) - 1.5


def clock(t: float) -> str:
    total = 7 * 60 + int(round(2 * t))
    return f"{total // 60:02d}:{total % 60:02d}"


def metrics(y: np.ndarray, fit: np.ndarray) -> dict[str, float]:
    err = fit - y
    denom = np.maximum(np.abs(y), 1e-9)
    return {
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAE": float(np.mean(np.abs(err))),
        "MAXAE": float(np.max(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / denom) * 100),
    }


def q1_shape_q3(x: np.ndarray, bp: tuple[float, float, float, float, float]) -> np.ndarray:
    a, b, c, d, e = bp
    p_part = np.zeros_like(x, dtype=float)
    s_part = np.zeros_like(x, dtype=float)

    m = (a < x) & (x <= b)
    p_part[m] = (x[m] - a) / (b - a)

    m = (b < x) & (x <= c)
    p_part[m] = (c - x[m]) / (c - b)
    s_part[m] = (x[m] - b) / (c - b)

    m = (c < x) & (x <= d)
    s_part[m] = 1.0

    m = (d < x) & (x <= e)
    s_part[m] = (e - x[m]) / (e - d)
    return np.column_stack([p_part, s_part])


def q1_shape_q4(x: np.ndarray, bp: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = bp
    out = np.zeros_like(x, dtype=float)
    m = (a < x) & (x <= b)
    out[m] = (x[m] - a) / (b - a)
    m = (b < x) & (x <= c)
    out[m] = 1.0
    m = (c < x) & (x <= d)
    out[m] = (d - x[m]) / (d - c)
    return out.reshape(-1, 1)


def q2_shape(x: np.ndarray, up_end: float, flat_end: float) -> np.ndarray:
    inc = np.minimum(np.maximum(x + 1.0, 0.0), up_end + 1.0)
    dec = np.maximum(x - flat_end, 0.0)
    return np.column_stack([np.ones_like(x), inc, dec])


def green_intervals(first_green_minute: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start = first_green_minute / 2.0
    while start < N:
        intervals.append((start, min(N - 1.0, start + 5.5)))
        start += 9.0
    return intervals


def q3_signal_design(first_green_minute: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    t = np.arange(N, dtype=float)
    cols = []
    intervals = green_intervals(first_green_minute)
    for start, end in intervals:
        active = (t >= start) & (t < end)
        const = np.where(active, 1.0, 0.0)
        slope = np.where(active, t - start, 0.0)
        cols.extend([const, slope])
    return np.column_stack(cols), intervals


def solve_weighted(A: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coef
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return coef


def huber_weights(resid: np.ndarray) -> np.ndarray:
    med = np.median(resid)
    scale = 1.4826 * np.median(np.abs(resid - med)) + 1e-6
    c = 1.345 * scale
    return np.minimum(1.0, c / np.maximum(np.abs(resid), 1e-9))


@dataclass
class FitResult:
    score: float
    bp: tuple[float, ...]
    coef: np.ndarray
    fit: np.ndarray
    detail: dict[str, object]


def best_q3(y: np.ndarray) -> FitResult:
    x = shifted_time_grid()
    q2 = q2_shape(x, up_end=36.0, flat_end=48.0)
    sig, intervals = q3_signal_design(first_green_minute=8.0)
    best: FitResult | None = None

    for a in (-1.0, 0.0, 2.0, 4.0):
        for b in range(int(a + 8), 25, 2):
            for c in range(b + 6, 43, 2):
                for d in range(c + 4, 53, 2):
                    for e in (55.0, 59.0, 61.0):
                        if not (a < b < c < d < e):
                            continue
                        q1 = q1_shape_q3(x, (a, b, c, d, e))
                        A = np.column_stack([q1, q2, sig])
                        coef = solve_weighted(A, y)
                        fit = A @ coef
                        q1_peak, q1_stable = coef[0], coef[1]
                        q2_base, q2_rise, q2_fall = coef[2], coef[3], coef[4]
                        penalty = 0.0
                        if q1_peak < q1_stable or q1_stable < 0:
                            penalty += 1000.0
                        if q2_base < 0 or q2_rise < 0 or q2_fall > 0:
                            penalty += 1000.0
                        sig_values = sig @ coef[5:]
                        penalty += 25.0 * float(np.mean(np.minimum(sig_values, 0.0) ** 2))
                        penalty += 5.0 * float(np.mean(np.minimum(fit, 0.0) ** 2))
                        score = metrics(y, fit)["RMSE"] + penalty
                        if best is None or score < best.score:
                            best = FitResult(score, (a, b, c, d, e), coef, fit, {"intervals": intervals})
    assert best is not None
    return best


def best_q4(y: np.ndarray) -> FitResult:
    x = shifted_time_grid()
    best: FitResult | None = None

    for first_green_minute in np.arange(0.0, 18.0, 1.0):
        sig, intervals = q3_signal_design(first_green_minute)
        q2 = q2_shape(x, up_end=18.0, flat_end=36.0)
        for a in (-1.0, 0.0, 2.0, 4.0, 6.0):
            for b in range(int(a + 6), 29, 2):
                for c in range(b + 6, 47, 2):
                    for d in (53.0, 57.0, 61.0):
                        if not (a < b < c < d):
                            continue
                        q1 = q1_shape_q4(x, (a, b, c, d))
                        A = np.column_stack([q1, q2, sig])
                        w = np.ones(N)
                        coef = solve_weighted(A, y, w)
                        for _ in range(8):
                            fit = A @ coef
                            w = huber_weights(y - fit)
                            coef = solve_weighted(A, y, w)
                        fit = A @ coef
                        q1_level = coef[0]
                        q2_base, q2_rise, q2_fall = coef[1], coef[2], coef[3]
                        sig_values = sig @ coef[4:]
                        penalty = 0.0
                        if q1_level < 0:
                            penalty += 1000.0
                        if q2_base < 0 or q2_rise < 0 or q2_fall > 0:
                            penalty += 1000.0
                        penalty += 25.0 * float(np.mean(np.minimum(sig_values, 0.0) ** 2))
                        robust = math.sqrt(float(np.average((fit - y) ** 2, weights=w)))
                        score = robust + penalty
                        if best is None or score < best.score:
                            best = FitResult(
                                score,
                                (a, b, c, d),
                                coef,
                                fit,
                                {"intervals": intervals, "first_green_minute": first_green_minute, "weights": w},
                            )
    assert best is not None
    return best


def eval_q1_q3(t: float, bp: tuple[float, ...], coef: np.ndarray) -> float:
    return float((q1_shape_q3(np.array([t]), bp) @ coef[:2]).item())


def eval_q1_q4(t: float, bp: tuple[float, ...], coef: np.ndarray) -> float:
    return float((q1_shape_q4(np.array([t]), bp) @ coef[:1]).item())


def eval_q2(t: float, coef3: np.ndarray, up_end: float, flat_end: float) -> float:
    return float((q2_shape(np.array([t]), up_end, flat_end) @ coef3).item())


def eval_signal(t: float, intervals: list[tuple[float, float]], coef: np.ndarray) -> float:
    for i, (start, end) in enumerate(intervals):
        if start <= t < end:
            return float(coef[2 * i] + coef[2 * i + 1] * (t - start))
    return 0.0


def print_metrics(label: str, y: np.ndarray, fit: np.ndarray) -> None:
    m = metrics(y, fit)
    print(f"{label}误差: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, 最大绝对误差={m['MAXAE']:.4f}, MAPE={m['MAPE']:.2f}%")


def print_q3(result: FitResult, y: np.ndarray) -> None:
    bp = result.bp
    c = result.coef
    intervals = result.detail["intervals"]
    assert isinstance(intervals, list)
    print("\n================ 问题3 ================")
    print(f"支路1断点 t={bp}，参数 peak={c[0]:.4f}, stable={c[1]:.4f}")
    print("支路1: 0 -> 线性增长 -> 线性减少 -> 稳定 -> 线性减少至0")
    print(f"支路2: q2(t) = {c[2]:.4f} + {c[3]:.4f}*min(max(t+1,0),37) + ({c[4]:.4f})*max(t-48,0)")
    print("支路3绿灯区间及函数:")
    for i, (start, end) in enumerate(intervals):
        a, b = c[5 + 2 * i], c[5 + 2 * i + 1]
        print(f"  [{clock(start)}, {clock(end)}): q3(t) = {a:.4f} + {b:.4f}(t-{start:.1f}); 红灯时 q3(t)=0")
    print_metrics("问题3", y, result.fit)
    for t in (15.0, 45.0):
        x = t - 1.5
        q1 = max(0.0, eval_q1_q3(x, bp, c))
        q2 = max(0.0, eval_q2(x, c[2:5], 36.0, 48.0))
        q3 = max(0.0, eval_signal(t, intervals, c[5:]))
        print(f"{clock(t)}: 支路1={q1:.4f}, 支路2={q2:.4f}, 支路3={q3:.4f}, 合计={q1 + q2 + q3:.4f}")


def print_q4(result: FitResult, y: np.ndarray) -> None:
    bp = result.bp
    c = result.coef
    intervals = result.detail["intervals"]
    first_green_minute = result.detail["first_green_minute"]
    assert isinstance(intervals, list)
    print("\n================ 问题4 ================")
    print(f"估计首个绿灯开始时刻: 7:{int(first_green_minute):02d}，支路1断点 t={bp}，稳定流量={c[0]:.4f}")
    print("支路1: 0 -> 线性增长 -> 稳定 -> 线性减少至0")
    print(f"支路2: q2(t) = {c[1]:.4f} + {c[2]:.4f}*min(max(t+1,0),19) + ({c[3]:.4f})*max(t-36,0)")
    print("支路3绿灯区间及函数:")
    for i, (start, end) in enumerate(intervals):
        a, b = c[4 + 2 * i], c[4 + 2 * i + 1]
        print(f"  [{clock(start)}, {clock(end)}): q3(t) = {a:.4f} + {b:.4f}(t-{start:.1f}); 红灯时 q3(t)=0")
    print_metrics("问题4(对含噪观测)", y, result.fit)
    w = result.detail.get("weights")
    if isinstance(w, np.ndarray):
        low_weight = np.where(w < 0.5)[0]
        print("稳健拟合识别的较大扰动采样点:", ", ".join(f"t={i}" for i in low_weight[:12]) or "无")
    for t in (15.0, 45.0):
        x = t - 1.5
        q1 = max(0.0, eval_q1_q4(x, bp, c))
        q2 = max(0.0, eval_q2(x, c[1:4], 18.0, 36.0))
        q3 = max(0.0, eval_signal(t, intervals, c[4:]))
        print(f"{clock(t)}: 支路1={q1:.4f}, 支路2={q2:.4f}, 支路3={q3:.4f}, 合计={q1 + q2 + q3:.4f}")


def main() -> None:
    configure_console()
    q3_y = load_series(2)
    q4_y = load_series(3)
    q3 = best_q3(q3_y)
    q4 = best_q4(q4_y)
    print_q3(q3, q3_y)
    print_q4(q4, q4_y)


if __name__ == "__main__":
    main()

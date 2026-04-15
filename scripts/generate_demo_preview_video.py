from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "nonzero_demo_tables_v4"
OUT_VIDEO = ROOT / "docs" / "nexus_demo_preview.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 30


def fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def make_title_frame(progress: float) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        blend = y / max(1, HEIGHT - 1)
        img[y, :, 0] = int(22 + 30 * blend)
        img[y, :, 1] = int(52 + 35 * blend)
        img[y, :, 2] = int(72 + 60 * blend)

    alpha = min(1.0, max(0.0, progress))
    title = "NEXUS-ATMS"
    subtitle = "AI-Based Urban Traffic Management Demo Preview"
    cv2.putText(img, title, (140, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (245, 245, 245), 5, cv2.LINE_AA)
    cv2.putText(img, subtitle, (145, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (220, 235, 245), 2, cv2.LINE_AA)

    bar_w = int(740 * alpha)
    cv2.rectangle(img, (140, 430), (880, 460), (70, 90, 110), -1)
    cv2.rectangle(img, (140, 430), (140 + bar_w, 460), (100, 200, 140), -1)
    cv2.putText(img, "Loading project highlights...", (140, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 240, 245), 2, cv2.LINE_AA)
    return img


def make_waiting_chart_frame(wide_df: pd.DataFrame, frame_idx: int, total_frames: int) -> np.ndarray:
    frac = min(1.0, (frame_idx + 1) / max(1, total_frames))
    max_tick = max(1, int(np.ceil(frac * len(wide_df))))
    sub = wide_df.iloc[:max_tick].copy()

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#f6f9fc")
    ax.set_facecolor("#ffffff")

    x = sub["Tick"].astype(int)
    for col, color in [("J0", "#0b84a5"), ("J1", "#f6c85f"), ("J2", "#6f4e7c"), ("J3", "#9dd866")]:
        ax.plot(x, sub[col].astype(float), label=col, linewidth=2.2, color=color)

    ax.plot(x, sub["AvgWait"].astype(float), label="AvgWait", linewidth=3.2, color="#e45756")

    ax.set_title("Time-Series Waiting Time Across Junction Groups", fontsize=21, weight="bold", pad=14)
    ax.set_xlabel("Tick", fontsize=12)
    ax.set_ylabel("Waiting Time (s)", fontsize=12)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=10)

    if not sub.empty:
        latest = float(sub["AvgWait"].iloc[-1])
        mean_val = float(sub["AvgWait"].mean())
        note = f"Current AvgWait: {latest:.2f}s   |   Segment Mean: {mean_val:.2f}s"
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=11, color="#1f3556")

    frame = fig_to_bgr(fig)
    plt.close(fig)
    return frame


def make_corridor_frame(before: float, after: float, frame_idx: int, total_frames: int) -> np.ndarray:
    t = min(1.0, (frame_idx + 1) / max(1, total_frames))
    cur_after = before - (before - after) * t

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")

    labels = ["Before Corridor", "After Corridor"]
    vals = [before, cur_after]
    colors = ["#ef476f", "#06d6a0"]
    bars = ax.bar(labels, vals, color=colors, width=0.56)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}s", ha="center", va="bottom", fontsize=16, weight="bold")

    gain_pct = 0.0
    if before > 0:
        gain_pct = ((before - cur_after) / before) * 100.0

    ax.set_ylim(0, max(before, after) * 1.7)
    ax.set_ylabel("Delay (seconds)", fontsize=12)
    ax.set_title("Green Corridor Impact on Emergency Vehicle Delay", fontsize=21, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.24)
    ax.text(0.03, 0.93, f"Delay improvement: {gain_pct:.1f}%", transform=ax.transAxes, fontsize=14, color="#264653")
    ax.text(0.03, 0.86, "Event Type: Ambulance priority corridor activation", transform=ax.transAxes, fontsize=12, color="#3a3a3a")

    frame = fig_to_bgr(fig)
    plt.close(fig)
    return frame


def make_outro_frame() -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(img, "Submission Preview Complete", (220, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (240, 240, 240), 4, cv2.LINE_AA)
    cv2.putText(img, "NEXUS-ATMS | RL + Green Corridor + Analytics", (260, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 225, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Next: replace placeholder timeline with your narration audio", (230, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (210, 210, 210), 2, cv2.LINE_AA)
    return img


def main() -> int:
    wide_path = DATA_DIR / "junction_waiting_times_nonzero_wide.csv"
    corridor_path = DATA_DIR / "green_corridor_events_nonzero.csv"

    if not wide_path.exists() or not corridor_path.exists():
        raise FileNotFoundError("Required CSV files not found in results/nonzero_demo_tables_v4")

    wide_df = pd.read_csv(wide_path)
    corridor_df = pd.read_csv(corridor_path)

    before = float(corridor_df["DelayBefore_s"].astype(float).mean()) if not corridor_df.empty else 13.4
    after = float(corridor_df["DelayAfter_s"].astype(float).mean()) if not corridor_df.empty else 7.37

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer. Try installing OpenCV codecs.")

    title_frames = 4 * FPS
    wait_frames = 12 * FPS
    corridor_frames = 8 * FPS
    outro_frames = 4 * FPS

    for i in range(title_frames):
        writer.write(make_title_frame(i / max(1, title_frames - 1)))

    for i in range(wait_frames):
        writer.write(make_waiting_chart_frame(wide_df, i, wait_frames))

    for i in range(corridor_frames):
        writer.write(make_corridor_frame(before, after, i, corridor_frames))

    outro = make_outro_frame()
    for _ in range(outro_frames):
        writer.write(outro)

    writer.release()
    print(f"Created demo preview video: {OUT_VIDEO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

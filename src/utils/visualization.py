"""
Visualization utilities for Smart Traffic Management System
Generates professional charts for training curves, comparisons, and reports.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------
COLOUR_BASELINE = "#E74C3C"
COLOUR_RL = "#2ECC71"
COLOUR_ACCENT = "#3498DB"


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ------------------------------------------------------------------
# Learning Curve
# ------------------------------------------------------------------
def plot_learning_curve(
    rewards: List[float],
    window: int = 20,
    title: str = "Training Learning Curve",
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plot reward over episodes with a smoothed overlay.

    Args:
        rewards: List of per-episode rewards.
        window: Smoothing window size.
        title: Chart title.
        save_path: Where to save the PNG (or None to skip).

    Returns:
        Path to saved image, or None.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(rewards) + 1)

    ax.plot(episodes, rewards, alpha=0.3, color=COLOUR_ACCENT, linewidth=0.8, label="Raw")

    # Moving average
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax.plot(
        episodes[window - 1 :],
        smoothed,
        color=COLOUR_ACCENT,
        linewidth=2,
        label=f"Smoothed (w={window})",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


# ------------------------------------------------------------------
# Baseline vs RL Comparison Bar Chart
# ------------------------------------------------------------------
def plot_comparison_bar(
    baseline: Dict,
    rl_agent: Dict,
    metrics: Optional[List[str]] = None,
    title: str = "Fixed-Timing vs RL Agent",
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Create a grouped bar chart comparing baseline and RL metrics.

    Args:
        baseline: Dict of baseline metric values.
        rl_agent: Dict of RL metric values.
        metrics: Keys to compare (defaults to common keys).
        title: Chart title.
        save_path: Where to save PNG.

    Returns:
        Path to saved image, or None.
    """
    if not HAS_MATPLOTLIB:
        return None

    if metrics is None:
        metrics = ["avg_waiting_time", "avg_queue_length", "throughput"]

    labels = [m.replace("_", " ").title() for m in metrics]
    baseline_vals = [baseline.get(m, 0) for m in metrics]
    rl_vals = [rl_agent.get(m, 0) for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Fixed-Timing", color=COLOUR_BASELINE)
    bars2 = ax.bar(x + width / 2, rl_vals, width, label="RL Agent", color=COLOUR_RL)

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


# ------------------------------------------------------------------
# Queue Length Over Time
# ------------------------------------------------------------------
def plot_queue_over_time(
    baseline_queues: List[float],
    rl_queues: List[float],
    title: str = "Queue Length Over Time",
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Plot queue lengths for baseline and RL over simulation steps."""
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    steps_b = np.arange(len(baseline_queues))
    steps_r = np.arange(len(rl_queues))

    ax.plot(steps_b, baseline_queues, color=COLOUR_BASELINE, alpha=0.7, label="Fixed-Timing")
    ax.plot(steps_r, rl_queues, color=COLOUR_RL, alpha=0.7, label="RL Agent")

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Queue Length (vehicles)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None


# ------------------------------------------------------------------
# Multi-Panel Report Figure
# ------------------------------------------------------------------
def generate_report_figure(
    rewards: List[float],
    baseline: Dict,
    rl_agent: Dict,
    save_path: str = "results/report.png",
) -> Optional[str]:
    """
    Generate a multi-panel summary figure suitable for reports.

    Panels:
        1. Learning curve
        2. Comparison bar chart
        3. Improvement percentages
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Learning curve ---
    ax = axes[0]
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color=COLOUR_ACCENT, linewidth=0.8)
    window = min(20, max(len(rewards) // 5, 1))
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1 :], smoothed, color=COLOUR_ACCENT, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress")

    # --- Panel 2: Comparison bars ---
    ax = axes[1]
    metrics = ["avg_waiting_time", "avg_queue_length", "throughput"]
    labels = ["Wait Time (s)", "Queue Len", "Throughput"]
    baseline_vals = [baseline.get(m, 0) for m in metrics]
    rl_vals = [rl_agent.get(m, 0) for m in metrics]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color=COLOUR_BASELINE)
    ax.bar(x + width / 2, rl_vals, width, label="RL Agent", color=COLOUR_RL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Performance Comparison")

    # --- Panel 3: Improvement percentages ---
    ax = axes[2]
    improvements = []
    imp_labels = []
    for m, lab in zip(metrics, labels):
        bv = baseline.get(m, 1)
        rv = rl_agent.get(m, 0)
        if bv != 0:
            pct = (rv - bv) / abs(bv) * 100
        else:
            pct = 0
        improvements.append(pct)
        imp_labels.append(lab)

    colours = [COLOUR_RL if v < 0 else COLOUR_BASELINE for v in improvements]
    # For throughput, positive improvement is good
    if len(improvements) > 2:
        colours[2] = COLOUR_RL if improvements[2] > 0 else COLOUR_BASELINE

    ax.barh(imp_labels, improvements, color=colours)
    ax.set_xlabel("Change (%)")
    ax.set_title("RL Improvement")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Smart Traffic Management — RL Performance Report", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

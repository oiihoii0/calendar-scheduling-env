"""
Gantt-chart visualization for CalendarSchedulingEnv schedules.

Produces a colour-coded timeline showing event placement, room assignments,
and conflicts — ideal for hackathon demos and README screenshots.

Usage:
    from scheduling_env.visualize import render_gantt
    render_gantt(env)                   # display interactively
    render_gantt(env, save_path="schedule.png")  # save to file
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Curated colour palette — vibrant but harmonious
_PALETTE = [
    "#6366F1",  # indigo
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#F43F5E",  # rose
    "#F97316",  # orange
    "#EAB308",  # yellow
    "#22C55E",  # green
    "#14B8A6",  # teal
    "#06B6D4",  # cyan
    "#3B82F6",  # blue
    "#A855F7",  # purple
    "#E11D48",  # crimson
    "#0EA5E9",  # sky
    "#10B981",  # emerald
    "#F59E0B",  # amber
]

_CONFLICT_COLOR = "#EF4444"
_BG_COLOR = "#0F172A"
_GRID_COLOR = "#1E293B"
_TEXT_COLOR = "#F8FAFC"
_SUBTEXT_COLOR = "#94A3B8"


def render_gantt(
    env: Any,
    *,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 7),
    show: bool = True,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Render the current schedule as a Gantt chart.

    Parameters
    ----------
    env : CalendarEnv
        Environment instance with a populated schedule.
    save_path : str, optional
        If provided, save the figure to this path (e.g. "schedule.png").
    figsize : tuple
        Figure dimensions in inches.
    show : bool
        Whether to call plt.show().
    title : str, optional
        Custom chart title.

    Returns
    -------
    matplotlib.figure.Figure or None if matplotlib is not installed.
    """
    if not HAS_MPL:
        print(
            "[visualize] matplotlib not installed. "
            "Install with: pip install matplotlib"
        )
        return None

    schedule = env.schedule
    if not schedule:
        print("[visualize] No events scheduled yet — nothing to render.")
        return None

    state = env.get_state()
    rooms = env.rooms
    room_to_y = {room: i for i, room in enumerate(rooms)}

    # ---- figure setup -----------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # ---- detect conflicts for highlighting ---------------------------
    conflict_pairs = set()
    grade_result = env.grader.grade(schedule)
    for a, b in grade_result.conflicts:
        conflict_pairs.add(a)
        conflict_pairs.add(b)

    # ---- draw events -------------------------------------------------
    for eid, slot in schedule.items():
        event = env.grader._event_map.get(eid)
        if event is None:
            continue

        start = slot["start_hour"]
        duration = event.duration_hours
        room = slot["room"]
        y = room_to_y.get(room, 0)

        is_conflict = eid in conflict_pairs
        color = _CONFLICT_COLOR if is_conflict else _PALETTE[eid % len(_PALETTE)]

        # Draw rounded rectangle
        rect = FancyBboxPatch(
            (start, y - 0.35),
            duration,
            0.7,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="white" if is_conflict else color,
            linewidth=2 if is_conflict else 1,
            alpha=0.9,
        )
        ax.add_patch(rect)

        # Event label
        label = event.title
        if len(label) > 18:
            label = label[:16] + "…"
        fontsize = 8 if duration < 0.75 else 9
        ax.text(
            start + duration / 2,
            y + 0.05,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
        )
        # Priority badge
        ax.text(
            start + duration / 2,
            y - 0.18,
            f"P{event.priority}",
            ha="center",
            va="center",
            fontsize=7,
            color=(1.0, 1.0, 1.0, 0.7),
        )

    # ---- axes formatting ---------------------------------------------
    ax.set_xlim(8.75, 17.25)
    ax.set_ylim(-0.6, len(rooms) - 0.4)
    ax.set_yticks(range(len(rooms)))
    ax.set_yticklabels(rooms, fontsize=11, fontweight="bold", color=_TEXT_COLOR)

    # X-axis: hours
    hours = [h for h in range(9, 18)]
    ax.set_xticks(hours)
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in hours],
        fontsize=9,
        color=_SUBTEXT_COLOR,
    )
    ax.tick_params(axis="both", colors=_SUBTEXT_COLOR, length=0)

    # Grid
    for h in hours:
        ax.axvline(h, color=_GRID_COLOR, linewidth=0.5, zorder=0)
    # Half-hour subtle grid
    for h in [h + 0.5 for h in range(9, 17)]:
        ax.axvline(h, color=_GRID_COLOR, linewidth=0.3, linestyle=":", zorder=0)

    for i in range(len(rooms)):
        ax.axhline(i - 0.5, color=_GRID_COLOR, linewidth=0.5, zorder=0)
    ax.axhline(len(rooms) - 0.5, color=_GRID_COLOR, linewidth=0.5, zorder=0)

    # Lunch zone highlight
    ax.axvspan(12, 13, alpha=0.08, color="#FBBF24", zorder=0)
    ax.text(
        12.5,
        len(rooms) - 0.55,
        "LUNCH",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#FBBF24",
        alpha=0.6,
    )

    # ---- title & stats -----------------------------------------------
    chart_title = title or f"Schedule: {state['task_name']}  ({state['difficulty']})"
    ax.set_title(
        chart_title,
        fontsize=15,
        fontweight="bold",
        color=_TEXT_COLOR,
        pad=16,
    )

    # Stats bar at bottom
    stats_text = (
        f"Events: {state['events_scheduled']}/{state['events_total']}   │   "
        f"Conflicts: {state['total_conflicts']}   │   "
        f"Violations: {state['constraint_violations']}   │   "
        f"Utilisation: {state['utilization_score']:.0%}   │   "
        f"Reward: {state['episode_reward']:.1f}"
    )
    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        color=_SUBTEXT_COLOR,
        fontstyle="italic",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_PALETTE[0], label="Scheduled"),
        mpatches.Patch(facecolor=_CONFLICT_COLOR, label="Conflict"),
        mpatches.Patch(facecolor="#FBBF24", alpha=0.3, label="Lunch window"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        facecolor=_GRID_COLOR,
        edgecolor=_GRID_COLOR,
        labelcolor=_TEXT_COLOR,
    )

    ax.invert_yaxis()
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOR)
        print(f"[visualize] Saved to {save_path}")

    if show:
        plt.show()

    return fig


def render_comparison(
    results: List[Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Any]:
    """Render a bar-chart comparing agent performance across tasks.

    Parameters
    ----------
    results : list of dict
        Output from baseline.run_episode for multiple runs.
    """
    if not HAS_MPL:
        print("[visualize] matplotlib not installed.")
        return None

    import numpy as np

    tasks = sorted(set(r["task"] for r in results))
    agents = sorted(set(r["agent"] for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=_BG_COLOR)

    metrics = [
        ("total_reward", "Total Reward"),
        ("conflicts", "Conflicts"),
        ("quality", "Quality Score"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        ax.set_facecolor(_BG_COLOR)
        x = np.arange(len(tasks))
        width = 0.25

        for i, agent in enumerate(agents):
            vals = [
                next(r[metric] for r in results if r["agent"] == agent and r["task"] == t)
                for t in tasks
            ]
            color = _PALETTE[i * 3]
            bars = ax.bar(x + i * width, vals, width, label=agent, color=color, alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [t.replace("_scheduling", "") for t in tasks],
            fontsize=9,
            color=_SUBTEXT_COLOR,
        )
        ax.set_title(label, fontsize=12, fontweight="bold", color=_TEXT_COLOR)
        ax.tick_params(colors=_SUBTEXT_COLOR)
        ax.spines[:].set_color(_GRID_COLOR)
        ax.legend(fontsize=8, facecolor=_GRID_COLOR, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    plt.suptitle(
        "🤖  Agent Performance Comparison",
        fontsize=14,
        fontweight="bold",
        color=_TEXT_COLOR,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOR)
        print(f"[visualize] Saved to {save_path}")
    if show:
        plt.show()

    return fig

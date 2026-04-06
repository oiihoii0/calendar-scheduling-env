"""
PPO Training Script for CalendarSchedulingEnv.

Trains a PPO agent (via Stable-Baselines3) on all three difficulty levels,
saves the trained models, and produces a learning-curve chart comparing
PPO against the heuristic baseline.

Usage:
    python train.py                        # train on all 3 tasks
    python train.py --task simple_scheduling --timesteps 50000
    python train.py --task complex_scheduling --timesteps 200000 --plot
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scheduling_env.env import CalendarEnv
from scheduling_env.baseline import HeuristicAgent, run_episode


# ======================================================================
# Config
# ======================================================================

TASK_CONFIGS = {
    "simple_scheduling": {
        "timesteps":  50_000,
        "n_steps":    512,
        "batch_size": 64,
        "n_epochs":   10,
        "lr":         3e-4,
    },
    "constrained_scheduling": {
        "timesteps":  100_000,
        "n_steps":    1024,
        "batch_size": 128,
        "n_epochs":   10,
        "lr":         2e-4,
    },
    "complex_scheduling": {
        "timesteps":  200_000,
        "n_steps":    2048,
        "batch_size": 256,
        "n_epochs":   10,
        "lr":         1e-4,
    },
}

MODELS_DIR = "trained_models"
CHARTS_DIR = "charts"


# ======================================================================
# Reward-tracking callback
# ======================================================================

class RewardLoggerCallback(BaseCallback):
    """Records mean episode reward every `log_interval` rollouts."""

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.timestep_log: List[int] = []
        self._ep_rewards_buf: List[float] = []
        self._rollout_count = 0

    def _on_step(self) -> bool:
        # SB3 Monitor wrapper stores episode info in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards_buf.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._ep_rewards_buf:
            mean_r = statistics.mean(self._ep_rewards_buf)
            self.episode_rewards.append(mean_r)
            self.timestep_log.append(self.num_timesteps)
            self._ep_rewards_buf = []

            if self._rollout_count % self.log_interval == 0:
                print(
                    f"  Step {self.num_timesteps:>8,d} | "
                    f"mean_reward = {mean_r:+.2f}"
                )


# ======================================================================
# Training
# ======================================================================

def make_env(task_name: str, seed: int = 0):
    """Factory for a Monitor-wrapped CalendarEnv."""
    def _init():
        env = CalendarEnv(task_name=task_name)
        env = Monitor(env)
        return env
    return _init


def train_ppo(
    task_name: str,
    timesteps: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a PPO agent on *task_name* and save the model.

    Returns a dict with training metadata and the reward curve.
    """
    cfg = TASK_CONFIGS[task_name]
    total_steps = timesteps or cfg["timesteps"]

    print(f"\n{'=' * 60}")
    print(f"  Training PPO on: {task_name}")
    print(f"  Timesteps: {total_steps:,}")
    print(f"{'=' * 60}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Vectorised environment (single env — no parallelism needed)
    vec_env = DummyVecEnv([make_env(task_name, seed)])

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["lr"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    callback = RewardLoggerCallback(log_interval=5, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback)
    elapsed = time.time() - t0

    # Save model
    model_path = os.path.join(MODELS_DIR, f"ppo_{task_name}")
    model.save(model_path)
    print(f"\n  Model saved to {model_path}.zip")
    print(f"  Training time: {elapsed:.1f}s")

    return {
        "task": task_name,
        "timesteps": total_steps,
        "elapsed": elapsed,
        "model_path": model_path,
        "reward_curve": callback.episode_rewards,
        "timestep_log": callback.timestep_log,
        "final_reward": callback.episode_rewards[-1] if callback.episode_rewards else 0.0,
    }


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_ppo(model_path: str, task_name: str, n_episodes: int = 10) -> Dict[str, Any]:
    """Load a saved model and evaluate it."""
    model = PPO.load(model_path)
    env = CalendarEnv(task_name=task_name)
    rewards = []
    conflicts_list = []
    scheduled_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        state = env.get_state()
        conflicts_list.append(state["total_conflicts"])
        scheduled_list.append(state["events_scheduled"])

    return {
        "agent": "PPO",
        "task": task_name,
        "reward_mean": round(statistics.mean(rewards), 2),
        "reward_std": round(statistics.stdev(rewards), 2) if len(rewards) > 1 else 0.0,
        "conflicts_mean": round(statistics.mean(conflicts_list), 1),
        "scheduled_mean": round(statistics.mean(scheduled_list), 1),
        "n_episodes": n_episodes,
    }


# ======================================================================
# Plotting
# ======================================================================

_BG = "#0F172A"
_GRID = "#1E293B"
_TEXT = "#F8FAFC"
_MUTED = "#94A3B8"
_COLORS = ["#6366F1", "#22C55E", "#F97316"]


def plot_learning_curves(
    training_results: List[Dict[str, Any]],
    save_path: str = "charts/learning_curves.png",
) -> None:
    """Plot reward-vs-timesteps for each trained task."""
    fig, axes = plt.subplots(1, len(training_results), figsize=(7 * len(training_results), 5),
                             facecolor=_BG)

    if len(training_results) == 1:
        axes = [axes]

    for ax, result, color in zip(axes, training_results, _COLORS):
        ax.set_facecolor(_BG)
        xs = result["timestep_log"]
        ys = result["reward_curve"]

        if not xs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", color=_TEXT)
            continue

        # Smooth with rolling average
        window = max(1, len(ys) // 10)
        ys_smooth = np.convolve(ys, np.ones(window) / window, mode="valid")
        xs_smooth = xs[window - 1:]

        ax.plot(xs, ys, color=color, alpha=0.2, linewidth=1)
        ax.plot(xs_smooth, ys_smooth, color=color, linewidth=2.5,
                label=f"PPO (smoothed)")

        ax.set_title(
            result["task"].replace("_", " ").title(),
            fontsize=13, fontweight="bold", color=_TEXT, pad=10
        )
        ax.set_xlabel("Timesteps", fontsize=10, color=_MUTED)
        ax.set_ylabel("Mean Episode Reward", fontsize=10, color=_MUTED)
        ax.tick_params(colors=_MUTED)
        ax.spines[:].set_color(_GRID)
        ax.set_facecolor(_BG)
        ax.legend(fontsize=9, facecolor=_GRID, edgecolor=_GRID, labelcolor=_TEXT)

        # Annotate final reward
        if ys:
            ax.annotate(
                f"Final: {ys[-1]:+.1f}",
                xy=(xs[-1], ys[-1]),
                xytext=(-60, 15),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )

    plt.suptitle("PPO Learning Curves — CalendarSchedulingEnv",
                 fontsize=14, fontweight="bold", color=_TEXT, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"  Learning curves saved to {save_path}")
    plt.close(fig)


def plot_comparison(
    ppo_results: List[Dict[str, Any]],
    save_path: str = "charts/ppo_vs_heuristic.png",
) -> None:
    """Bar chart: PPO reward vs heuristic reward per task."""
    tasks = [r["task"] for r in ppo_results]

    # Get heuristic rewards
    heuristic_rewards = []
    for task in tasks:
        results = [run_episode(HeuristicAgent(), task, seed=s, verbose=False)
                   for s in range(10)]
        heuristic_rewards.append(statistics.mean(r["total_reward"] for r in results))

    ppo_rewards = [r["reward_mean"] for r in ppo_results]
    ppo_stds = [r["reward_std"] for r in ppo_results]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG)
    ax.set_facecolor(_BG)

    bars1 = ax.bar(x - width / 2, heuristic_rewards, width,
                   label="Heuristic", color="#94A3B8", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ppo_rewards, width,
                   yerr=ppo_stds, capsize=5,
                   label="PPO (trained)", color="#6366F1", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_scheduling", "").replace("_", " ").title() for t in tasks],
        fontsize=11, color=_TEXT
    )
    ax.set_ylabel("Mean Episode Reward", fontsize=11, color=_MUTED)
    ax.set_title("PPO vs Heuristic Baseline",
                 fontsize=14, fontweight="bold", color=_TEXT, pad=12)
    ax.tick_params(colors=_MUTED)
    ax.spines[:].set_color(_GRID)
    ax.legend(fontsize=10, facecolor=_GRID, edgecolor=_GRID, labelcolor=_TEXT)
    ax.axhline(0, color=_MUTED, linewidth=0.5, linestyle="--")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                ha="center", fontsize=8, color=_TEXT)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                ha="center", fontsize=8, color=_TEXT)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"  Comparison chart saved to {save_path}")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CalendarSchedulingEnv")
    parser.add_argument("--task", type=str, default="all",
                        help="Task name or 'all' (default: all)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override default timestep count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes for post-training evaluation")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    training_results = []
    eval_results = []

    for task in tasks:
        # Train
        result = train_ppo(task, timesteps=args.timesteps, seed=args.seed)
        training_results.append(result)

        # Evaluate
        print(f"\n  Evaluating PPO on {task} ({args.eval_episodes} episodes)...")
        eval_r = evaluate_ppo(result["model_path"], task, args.eval_episodes)
        eval_results.append(eval_r)
        print(f"  PPO reward: {eval_r['reward_mean']:+.2f} +/- {eval_r['reward_std']:.2f} "
              f"| conflicts: {eval_r['conflicts_mean']:.1f}")

    # Summary table
    print(f"\n{'=' * 65}")
    print(f"  {'Task':<28} | {'PPO Reward':>12} | {'Conflicts':>9}")
    print(f"  {'-' * 60}")
    for r in eval_results:
        print(f"  {r['task']:<28} | "
              f"{r['reward_mean']:>+6.2f} +/- {r['reward_std']:<4.2f} | "
              f"{r['conflicts_mean']:>9.1f}")
    print(f"{'=' * 65}")

    # Charts
    if not args.no_plot:
        print("\nGenerating charts...")
        plot_learning_curves(training_results)
        plot_comparison(eval_results)
        print("Done! Charts saved to charts/")


if __name__ == "__main__":
    main()

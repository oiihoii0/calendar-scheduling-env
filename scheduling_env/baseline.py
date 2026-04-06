"""
Baseline agent for CalendarSchedulingEnv.

Implements three strategies:
    1. RandomAgent       – picks a random valid action each step.
    2. GreedyAgent       – schedules highest-priority event first in the
                           earliest available slot.
    3. HeuristicAgent    – greedy priority ordering + constraint-aware
                           slot selection.

Each agent exposes a simple ``act(obs, info, env) -> int`` interface.
The ``run_episode`` helper drives a full episode and returns results.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scheduling_env.env import CalendarEnv


# ======================================================================
# Agent base
# ======================================================================

class BaseAgent:
    """Base class for scheduling agents."""

    name: str = "base"

    def act(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        env: CalendarEnv,
    ) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ======================================================================
# 1. Random agent
# ======================================================================

class RandomAgent(BaseAgent):
    """Selects a uniformly random *valid* action each step."""

    name = "random"

    def act(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        env: CalendarEnv,
    ) -> int:
        valid = env.get_valid_actions()
        if valid:
            return random.choice(valid)
        return env.action_space.sample()


# ======================================================================
# 2. Greedy agent
# ======================================================================

class GreedyAgent(BaseAgent):
    """Schedules the highest-priority unscheduled event in the earliest
    available slot."""

    name = "greedy"

    def act(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        env: CalendarEnv,
    ) -> int:
        # Sort unscheduled events by descending priority, then ascending id
        unscheduled = [
            ev for ev in env.all_events if ev.id not in env.scheduled_ids
        ]
        if not unscheduled:
            return env.action_space.sample()

        unscheduled.sort(key=lambda e: (-e.priority, e.id))
        target = unscheduled[0]
        target_idx = env.all_events.index(target)

        # Find earliest valid slot
        best_action = self._find_earliest_slot(target, target_idx, env)
        if best_action is not None:
            return best_action

        # Fallback: any valid action
        valid = env.get_valid_actions()
        if valid:
            return valid[0]
        return env.action_space.sample()

    @staticmethod
    def _find_earliest_slot(
        event: Any, event_idx: int, env: CalendarEnv
    ) -> Optional[int]:
        slots_needed = math.ceil(event.duration_minutes / env._SLOT_DURATION)
        for t in range(env._TIME_SLOTS_PER_DAY):
            start_hour = env._DAY_START + t * (env._SLOT_DURATION / 60.0)
            if (
                start_hour < event.earliest_start_hour
                or start_hour > event.latest_start_hour
            ):
                continue
            for r in range(env.num_rooms):
                # Check room requirement
                if (
                    event.room_required
                    and env.rooms[r] != event.room_required
                ):
                    continue
                can_fit = all(
                    t + dt < env._TIME_SLOTS_PER_DAY
                    and env.slot_grid[t + dt, r] > 0.5
                    for dt in range(slots_needed)
                )
                if can_fit:
                    return (
                        event_idx * env.num_slots
                        + t * env.num_rooms
                        + r
                    )
        return None


# ======================================================================
# 3. Heuristic agent
# ======================================================================

class HeuristicAgent(BaseAgent):
    """Priority-ordered scheduling with constraint awareness.

    Improvements over GreedyAgent:
    - Respects room requirements strictly.
    - Avoids attendee conflicts by checking the current schedule.
    - Prefers slots that leave the most flexibility for remaining events.
    """

    name = "heuristic"

    def act(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        env: CalendarEnv,
    ) -> int:
        unscheduled = [
            ev for ev in env.all_events if ev.id not in env.scheduled_ids
        ]
        if not unscheduled:
            return env.action_space.sample()

        unscheduled.sort(key=lambda e: (-e.priority, e.earliest_start_hour))

        for target in unscheduled:
            target_idx = env.all_events.index(target)
            action = self._find_best_slot(target, target_idx, env)
            if action is not None:
                return action

        # Fallback
        valid = env.get_valid_actions()
        if valid:
            return valid[0]
        return env.action_space.sample()

    def _find_best_slot(
        self, event: Any, event_idx: int, env: CalendarEnv
    ) -> Optional[int]:
        """Find the best conflict-free slot for *event*."""
        slots_needed = math.ceil(event.duration_minutes / env._SLOT_DURATION)
        candidates: List[Tuple[float, int]] = []  # (score, action)

        for t in range(env._TIME_SLOTS_PER_DAY):
            start_hour = env._DAY_START + t * (env._SLOT_DURATION / 60.0)
            end_hour = start_hour + event.duration_hours

            if (
                start_hour < event.earliest_start_hour
                or start_hour > event.latest_start_hour
            ):
                continue
            if end_hour > env._DAY_END:
                continue

            for r in range(env.num_rooms):
                room = env.rooms[r]
                # Room requirement
                if event.room_required and room != event.room_required:
                    continue
                # Slot availability
                can_fit = all(
                    t + dt < env._TIME_SLOTS_PER_DAY
                    and env.slot_grid[t + dt, r] > 0.5
                    for dt in range(slots_needed)
                )
                if not can_fit:
                    continue

                # Check attendee conflicts
                has_conflict = False
                for eid, sched_slot in env.schedule.items():
                    other = env.grader._event_map.get(eid)
                    if other is None:
                        continue
                    other_start = sched_slot["start_hour"]
                    other_end = other_start + other.duration_hours
                    if start_hour < other_end and other_start < end_hour:
                        shared = set(event.attendees) & set(other.attendees)
                        if shared or sched_slot["room"] == room:
                            has_conflict = True
                            break
                if has_conflict:
                    continue

                # Score: prefer earlier + more remaining flexibility
                flexibility = sum(
                    env.slot_grid[tt, r]
                    for tt in range(env._TIME_SLOTS_PER_DAY)
                ) - slots_needed
                score = flexibility - t * 0.1

                action = (
                    event_idx * env.num_slots
                    + t * env.num_rooms
                    + r
                )
                candidates.append((score, action))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][1]
        return None


# ======================================================================
# Episode runner
# ======================================================================

def run_episode(
    agent: BaseAgent,
    task_name: str = "simple_scheduling",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single episode with the given agent and return results.

    Parameters
    ----------
    agent : BaseAgent
        Scheduling agent.
    task_name : str
        One of "simple_scheduling", "constrained_scheduling",
        "complex_scheduling".
    seed : int
        RNG seed.
    verbose : bool
        Print step-by-step and final schedule.

    Returns
    -------
    dict with keys: total_reward, steps, events_scheduled, events_total,
                    conflicts, violations, utilization, quality.
    """
    env = CalendarEnv(task_name=task_name)
    obs, info = env.reset(seed=seed)

    if verbose:
        print(f"\n{'#' * 60}")
        print(f" Agent: {agent.name}  |  Task: {task_name}")
        print(f"{'#' * 60}")

    total_reward = 0.0
    done = False

    while not done:
        action = agent.act(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if verbose and reward != 0:
            decoded = info.get("action_decoded", {})
            ev_idx = decoded.get("event_idx", "?")
            ev_name = (
                env.all_events[ev_idx].title
                if isinstance(ev_idx, int) and ev_idx < len(env.all_events)
                else "?"
            )
            print(
                f"  Step {info['step']:>3d} | "
                f"Placed {ev_name:<22s} at {decoded.get('start_hour', '?'):>5} "
                f"in {decoded.get('room', '?'):<16s} | "
                f"reward={reward:+.2f}"
            )

    if verbose:
        env.render()

    state = env.get_state()
    return {
        "agent": agent.name,
        "task": task_name,
        "total_reward": round(total_reward, 4),
        "steps": state["step"],
        "events_scheduled": state["events_scheduled"],
        "events_total": state["events_total"],
        "conflicts": state["total_conflicts"],
        "violations": state["constraint_violations"],
        "utilization": state["utilization_score"],
        "quality": state["quality_score"],
    }


# ======================================================================
# CLI entry-point
# ======================================================================

def main() -> None:
    """Run all three agents on all three tasks and print a summary table."""
    tasks = [
        "simple_scheduling",
        "constrained_scheduling",
        "complex_scheduling",
    ]
    agents = [RandomAgent(), GreedyAgent(), HeuristicAgent()]
    results: List[Dict[str, Any]] = []

    for task in tasks:
        for agent in agents:
            res = run_episode(agent, task_name=task, seed=42, verbose=True)
            results.append(res)

    # Summary table
    print(f"\n{'=' * 90}")
    print(f" {'Agent':<12s} | {'Task':<25s} | {'Reward':>8s} | "
          f"{'Sched':>5s} | {'Confl':>5s} | {'Util':>6s} | {'Qual':>6s}")
    print(f"{'-' * 90}")
    for r in results:
        print(
            f" {r['agent']:<12s} | {r['task']:<25s} | "
            f"{r['total_reward']:>8.2f} | "
            f"{r['events_scheduled']:>5d} | "
            f"{r['conflicts']:>5d} | "
            f"{r['utilization']:>6.1%} | "
            f"{r['quality']:>6.2f}"
        )
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()

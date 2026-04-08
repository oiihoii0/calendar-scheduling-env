"""
CalendarEnv – Gymnasium-compatible scheduling environment.

The agent's objective is to assign each event a start time and room,
maximising priority-weighted utilisation while satisfying all constraints.

Action encoding
---------------
    action (int) = event_index * num_slots + slot_index

    - event_index : selects which unscheduled event to place (0 .. N-1)
    - slot_index  : selects a discretised (time, room) pair

Observation space
-----------------
    Dict with:
        scheduled_events : Box(0, 1, shape=(max_events, 4))
            Per-event features: [normalised_start, norm_duration, norm_priority, scheduled?]
        available_slots  : Box(0, 1, shape=(num_time_steps * num_rooms, 2))
            Per-slot features: [normalised_time, is_available]
        conflicts        : Discrete(max_conflicts + 1)
        time_remaining   : Box(0, 1, shape=(1,))
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium
import numpy as np

from scheduling_env.grader import ScheduleGrader
from scheduling_env.models import CalendarConstraint, Event, ScheduleState, TimeSlot


class CalendarEnv(gymnasium.Env):
    """A Gymnasium environment for meeting-scheduling optimisation."""

    metadata = {"render_modes": []}

    # Time discretisation: 30-minute slots from 09:00 to 17:00 → 16 slots
    _TIME_SLOTS_PER_DAY = 16
    _SLOT_DURATION = 30  # minutes
    _DAY_START = 9.0
    _DAY_END = 17.0

    def __init__(
        self,
        task_name: str = "simple_scheduling",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.task_name = task_name
        self.render_mode = render_mode

        # ---- load task data -------------------------------------------------
        self._task_data = self._load_task(task_name)
        self.all_events: List[Event] = self._task_data["events"]
        self.constraints: List[CalendarConstraint] = self._task_data["constraints"]
        self.rooms: List[str] = self._task_data["rooms"]
        self.max_steps: int = self._task_data["max_steps"]
        self.task_difficulty: str = self._task_data["difficulty"]
        self.task_description: str = self._task_data["description"]

        # ---- derived dimensions ---------------------------------------------
        self.num_events = len(self.all_events)
        self.num_rooms = len(self.rooms)
        self.num_slots = self._TIME_SLOTS_PER_DAY * self.num_rooms  # per-room slots
        self.max_events = 15  # padded upper bound for observation

        # ---- spaces ---------------------------------------------------------
        self.action_space = gymnasium.spaces.Discrete(
            self.num_events * self.num_slots
        )
        self.observation_space = gymnasium.spaces.Dict(
            {
                "scheduled_events": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_events, 4),
                    dtype=np.float32,
                ),
                "available_slots": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._TIME_SLOTS_PER_DAY * self.num_rooms, 2),
                    dtype=np.float32,
                ),
                "conflicts": gymnasium.spaces.Discrete(
                    self.num_events * (self.num_events - 1) // 2 + 1
                ),
                "time_remaining": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        # ---- episode state --------------------------------------------------
        self.current_step: int = 0
        self.schedule: Dict[int, Dict[str, Any]] = {}
        self.scheduled_ids: Set[int] = set()
        self.slot_grid: np.ndarray = np.ones(
            (self._TIME_SLOTS_PER_DAY, self.num_rooms), dtype=np.float32
        )
        self.episode_reward: float = 0.0
        self.current_conflicts: int = 0

        # ---- grader ---------------------------------------------------------
        self.grader = ScheduleGrader(
            all_events=self.all_events,
            constraints=self.constraints,
            rooms=self.rooms,
        )

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        self.current_step = 0
        self.schedule = {}
        self.scheduled_ids = set()
        self.slot_grid = np.ones(
            (self._TIME_SLOTS_PER_DAY, self.num_rooms), dtype=np.float32
        )
        self.episode_reward = 0.0
        self.current_conflicts = 0

        obs = self._get_observation()
        info = {
            "step": 0,
            "task": self.task_name,
            "difficulty": self.task_difficulty,
            "num_events": self.num_events,
            "description": self.task_description,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one scheduling action.

        Parameters
        ----------
        action : int
            Encoded as  event_index * num_slots + slot_index.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # decode action
        event_idx = action // self.num_slots
        slot_idx = action % self.num_slots
        time_idx = slot_idx // self.num_rooms
        room_idx = slot_idx % self.num_rooms

        start_hour = self._DAY_START + time_idx * (self._SLOT_DURATION / 60.0)
        room = self.rooms[room_idx]

        reward = 0.0
        info: Dict[str, Any] = {
            "step": self.current_step,
            "action_decoded": {
                "event_idx": event_idx,
                "start_hour": start_hour,
                "room": room,
            },
        }

        # ---- validate event index -------------------------------------------
        if event_idx >= self.num_events:
            reward = -1.0
            info["error"] = "invalid_event_index"
            return (
                self._get_observation(),
                reward,
                False,
                self._is_truncated(),
                info,
            )

        event = self.all_events[event_idx]

        # ---- already scheduled? ---------------------------------------------
        if event.id in self.scheduled_ids:
            reward = -0.5
            info["error"] = "already_scheduled"
            return (
                self._get_observation(),
                reward,
                False,
                self._is_truncated(),
                info,
            )

        # ---- grade the placement --------------------------------------------
        step_reward, step_info = self.grader.grade_step(
            event=event,
            start_hour=start_hour,
            room=room,
            current_schedule=self.schedule,
        )
        reward = step_reward
        info.update(step_info)

        # ---- commit to schedule (even if conflicting – agent must learn) ----
        if step_info.get("valid", True):
            self.schedule[event.id] = {"start_hour": start_hour, "room": room}
            self.scheduled_ids.add(event.id)
            self._mark_slots(event, time_idx, room_idx)

        self.current_conflicts = len(
            self.grader.grade(self.schedule).conflicts
        )
        self.episode_reward += reward

        # ---- termination ----------------------------------------------------
        terminated = len(self.scheduled_ids) == self.num_events
        truncated = self._is_truncated()

        # bonus for finishing all events conflict-free
        if terminated and self.current_conflicts == 0:
            bonus = 5.0 + self.num_events * 0.5
            reward += bonus
            self.episode_reward += bonus
            info["completion_bonus"] = bonus

        obs = self._get_observation()
        info["episode_reward"] = self.episode_reward
        info["total_conflicts"] = self.current_conflicts
        info["events_scheduled"] = len(self.scheduled_ids)

        return obs, reward, terminated, truncated, info

    # ==================================================================
    # State / observation helpers
    # ==================================================================

    def _get_observation(self) -> Dict[str, Any]:
        """Build the observation dict for the current state."""
        # scheduled_events: (max_events, 4)
        sched = np.zeros((self.max_events, 4), dtype=np.float32)
        for i, event in enumerate(self.all_events):
            if i >= self.max_events:
                break
            sched[i, 0] = (event.earliest_start_hour - self._DAY_START) / (
                self._DAY_END - self._DAY_START
            )
            sched[i, 1] = event.duration_minutes / 90.0
            sched[i, 2] = event.priority / 5.0
            sched[i, 3] = 1.0 if event.id in self.scheduled_ids else 0.0

        # available_slots: (num_time_slots * num_rooms, 2)
        slots = np.zeros(
            (self._TIME_SLOTS_PER_DAY * self.num_rooms, 2), dtype=np.float32
        )
        for t in range(self._TIME_SLOTS_PER_DAY):
            for r in range(self.num_rooms):
                idx = t * self.num_rooms + r
                slots[idx, 0] = t / self._TIME_SLOTS_PER_DAY
                slots[idx, 1] = self.slot_grid[t, r]

        # conflicts
        conflicts = min(
            self.current_conflicts,
            self.num_events * (self.num_events - 1) // 2,
        )

        # time_remaining
        remaining = np.array(
            [1.0 - self.current_step / self.max_steps], dtype=np.float32
        )

        return {
            "scheduled_events": sched,
            "available_slots": slots,
            "conflicts": conflicts,
            "time_remaining": remaining,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the full environment state.

        Useful for the OpenEnv spec's ``state`` accessor.
        """
        state = self.grader.build_schedule_state(self.schedule)
        return {
            "task_name": self.task_name,
            "difficulty": self.task_difficulty,
            "step": self.current_step,
            "max_steps": self.max_steps,
            "schedule": {
                eid: {
                    "event_title": self.grader._event_map[eid].title,
                    "start_hour": s["start_hour"],
                    "room": s["room"],
                }
                for eid, s in self.schedule.items()
            },
            "events_scheduled": len(self.scheduled_ids),
            "events_total": self.num_events,
            "total_conflicts": state.total_conflicts,
            "constraint_violations": state.constraint_violations,
            "utilization_score": round(state.utilization_score, 4),
            "quality_score": round(state.quality_score, 4),
            "episode_reward": round(self.episode_reward, 4),
        }

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _load_task(task_name: str) -> Dict[str, Any]:
        """Import and call the task factory function."""
        from scheduling_env import tasks

        # Handle Gymnasium ID mapping
        mapping = {
            "CalendarSchedulingEasy-v0": "simple_scheduling",
            "CalendarSchedulingMedium-v0": "constrained_scheduling",
            "CalendarSchedulingHard-v0": "complex_scheduling",
            "CalendarScheduling-v0": "simple_scheduling",
        }
        internal_name = mapping.get(task_name, task_name)

        factory_name = f"create_{internal_name}_task"
        factory = getattr(tasks, factory_name, None)
        if factory is None:
            available = [
                n.replace("create_", "").replace("_task", "")
                for n in dir(tasks)
                if n.startswith("create_") and n.endswith("_task")
            ]
            raise ValueError(
                f"Unknown task {task_name!r}. Available: {available}"
            )
        return factory()

    def _mark_slots(self, event: Event, time_idx: int, room_idx: int) -> None:
        """Mark time-room slots as occupied."""
        slots_needed = math.ceil(event.duration_minutes / self._SLOT_DURATION)
        for dt in range(slots_needed):
            t = time_idx + dt
            if 0 <= t < self._TIME_SLOTS_PER_DAY:
                self.slot_grid[t, room_idx] = 0.0

    def _is_truncated(self) -> bool:
        return self.current_step >= self.max_steps

    # ==================================================================
    # Nice-to-haves
    # ==================================================================

    def get_valid_actions(self) -> List[int]:
        """Return a list of action indices that correspond to unscheduled
        events placed in currently-available slots."""
        valid: List[int] = []
        for ev_idx, event in enumerate(self.all_events):
            if event.id in self.scheduled_ids:
                continue
            slots_needed = math.ceil(
                event.duration_minutes / self._SLOT_DURATION
            )
            for t in range(self._TIME_SLOTS_PER_DAY):
                start_hour = self._DAY_START + t * (self._SLOT_DURATION / 60.0)
                if (
                    start_hour < event.earliest_start_hour
                    or start_hour > event.latest_start_hour
                ):
                    continue
                # check room availability
                for r in range(self.num_rooms):
                    can_fit = all(
                        self.slot_grid[t + dt, r] > 0.5
                        for dt in range(slots_needed)
                        if t + dt < self._TIME_SLOTS_PER_DAY
                    )
                    if can_fit:
                        action = ev_idx * self.num_slots + t * self.num_rooms + r
                        valid.append(action)
        return valid

    def render(self) -> None:
        """Pretty-print the current schedule to stdout."""
        print(f"\n{'=' * 60}")
        print(f" Schedule — {self.task_name} ({self.task_difficulty})")
        print(f" Step {self.current_step}/{self.max_steps}")
        print(f"{'=' * 60}")

        if not self.schedule:
            print("  (no events scheduled yet)")
        else:
            for eid in sorted(self.schedule):
                slot = self.schedule[eid]
                ev = self.grader._event_map[eid]
                sh = slot["start_hour"]
                eh = sh + ev.duration_hours
                h_s, m_s = int(sh), int((sh % 1) * 60)
                h_e, m_e = int(eh), int((eh % 1) * 60)
                print(
                    f"  [{h_s:02d}:{m_s:02d}–{h_e:02d}:{m_e:02d}] "
                    f"{ev.title:<22s} | room={slot['room']:<15s} "
                    f"| pri={ev.priority}"
                )

        state = self.get_state()
        print(f"\n  Conflicts: {state['total_conflicts']}  |  "
              f"Violations: {state['constraint_violations']}  |  "
              f"Utilisation: {state['utilization_score']:.1%}  |  "
              f"Reward: {state['episode_reward']:.2f}")
        print(f"{'=' * 60}\n")

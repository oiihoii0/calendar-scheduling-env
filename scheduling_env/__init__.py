"""
CalendarSchedulingEnv - An OpenEnv environment for AI-driven calendar scheduling.

This package provides a Gymnasium-compatible environment where an AI agent
learns to optimally schedule meetings and events on a calendar, respecting
constraints such as time windows, room availability, attendee conflicts,
and priority levels.

Difficulty Levels:
    - easy (simple_scheduling): 3-5 events, no room constraints, wide time windows.
    - medium (constrained_scheduling): 6-8 events, room constraints, attendee conflicts.
    - hard (complex_scheduling): 10-15 events, tight windows, travel time, lunch breaks.

Gym Registration:
    import gymnasium as gym
    import scheduling_env  # auto-registers envs

    env = gym.make("CalendarScheduling-v0")
    env = gym.make("CalendarSchedulingEasy-v0")
    env = gym.make("CalendarSchedulingMedium-v0")
    env = gym.make("CalendarSchedulingHard-v0")
"""

__version__ = "1.0.0"
__author__ = "CalendarSchedulingEnv Team"

from scheduling_env.env import CalendarEnv
from scheduling_env.models import (
    CalendarConstraint,
    Event,
    ScheduleState,
    TimeSlot,
)
from scheduling_env.tasks import (
    create_complex_scheduling_task,
    create_constrained_scheduling_task,
    create_simple_scheduling_task,
)
from scheduling_env.grader import ScheduleGrader

# Auto-register Gymnasium environments on import
from scheduling_env import registration as _registration  # noqa: F401

__all__ = [
    "CalendarEnv",
    "Event",
    "TimeSlot",
    "CalendarConstraint",
    "ScheduleState",
    "ScheduleGrader",
    "create_simple_scheduling_task",
    "create_constrained_scheduling_task",
    "create_complex_scheduling_task",
]

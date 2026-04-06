"""
Gymnasium environment registration for CalendarSchedulingEnv.

After installing the package, users can create environments via:

    import gymnasium as gym
    env = gym.make("CalendarScheduling-v0")                  # easy
    env = gym.make("CalendarScheduling-v0", task_name="constrained_scheduling")  # medium
    env = gym.make("CalendarScheduling-v0", task_name="complex_scheduling")      # hard

Or use the difficulty-specific aliases:

    env = gym.make("CalendarSchedulingEasy-v0")
    env = gym.make("CalendarSchedulingMedium-v0")
    env = gym.make("CalendarSchedulingHard-v0")
"""

import gymnasium


def register_envs() -> None:
    """Register all CalendarSchedulingEnv variants with Gymnasium."""

    # Generic entry — users pass task_name kwarg
    gymnasium.register(
        id="CalendarScheduling-v0",
        entry_point="scheduling_env.env:CalendarEnv",
        kwargs={"task_name": "simple_scheduling"},
    )

    # Difficulty-specific aliases
    gymnasium.register(
        id="CalendarSchedulingEasy-v0",
        entry_point="scheduling_env.env:CalendarEnv",
        kwargs={"task_name": "simple_scheduling"},
    )

    gymnasium.register(
        id="CalendarSchedulingMedium-v0",
        entry_point="scheduling_env.env:CalendarEnv",
        kwargs={"task_name": "constrained_scheduling"},
    )

    gymnasium.register(
        id="CalendarSchedulingHard-v0",
        entry_point="scheduling_env.env:CalendarEnv",
        kwargs={"task_name": "complex_scheduling"},
    )


# Auto-register when the module is imported
register_envs()

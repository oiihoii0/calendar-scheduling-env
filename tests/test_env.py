"""
Test suite for CalendarSchedulingEnv.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest
import numpy as np


# =====================================================================
# Models
# =====================================================================

class TestEvent:
    def test_valid_event(self):
        from scheduling_env.models import Event

        ev = Event(
            id=0, title="Sync", duration_minutes=30,
            earliest_start_hour=9, latest_start_hour=16,
            priority=3, attendees=["Alice"],
        )
        assert ev.duration_hours == 0.5
        assert ev.room_required is None
        assert "Sync" in repr(ev)

    def test_invalid_duration(self):
        from scheduling_env.models import Event

        with pytest.raises(ValueError, match="duration_minutes"):
            Event(id=0, title="X", duration_minutes=45,
                  earliest_start_hour=9, latest_start_hour=16,
                  priority=3, attendees=[])

    def test_invalid_priority(self):
        from scheduling_env.models import Event

        with pytest.raises(ValueError, match="priority"):
            Event(id=0, title="X", duration_minutes=30,
                  earliest_start_hour=9, latest_start_hour=16,
                  priority=6, attendees=[])

    def test_invalid_time_window(self):
        from scheduling_env.models import Event

        with pytest.raises(ValueError, match="earliest_start_hour"):
            Event(id=0, title="X", duration_minutes=30,
                  earliest_start_hour=15, latest_start_hour=10,
                  priority=3, attendees=[])


class TestTimeSlot:
    def test_overlap(self):
        from scheduling_env.models import TimeSlot

        a = TimeSlot(start_hour=9.0, duration_minutes=60, room="A")
        b = TimeSlot(start_hour=9.5, duration_minutes=60, room="A")
        c = TimeSlot(start_hour=10.0, duration_minutes=60, room="A")

        assert a.overlaps(b)
        assert not a.overlaps(c)  # a ends at 10.0, c starts at 10.0


class TestCalendarConstraint:
    def test_valid_types(self):
        from scheduling_env.models import CalendarConstraint

        c = CalendarConstraint(constraint_type="lunch_break", severity="soft")
        assert c.severity == "soft"

    def test_invalid_type(self):
        from scheduling_env.models import CalendarConstraint

        with pytest.raises(ValueError, match="constraint_type"):
            CalendarConstraint(constraint_type="made_up")

    def test_invalid_severity(self):
        from scheduling_env.models import CalendarConstraint

        with pytest.raises(ValueError, match="severity"):
            CalendarConstraint(constraint_type="no_overlap", severity="medium")


class TestScheduleState:
    def test_quality_score(self):
        from scheduling_env.models import ScheduleState

        state = ScheduleState(
            utilization_score=0.8,
            total_conflicts=0,
            constraint_violations=0,
        )
        assert state.quality_score == 0.8
        assert state.is_conflict_free

    def test_quality_penalised(self):
        from scheduling_env.models import ScheduleState

        state = ScheduleState(
            utilization_score=0.8,
            total_conflicts=2,
            constraint_violations=1,
        )
        assert state.quality_score < 0.8
        assert not state.is_conflict_free


# =====================================================================
# Tasks
# =====================================================================

class TestTasks:
    def test_simple_task(self):
        from scheduling_env.tasks import create_simple_scheduling_task

        task = create_simple_scheduling_task()
        assert len(task["events"]) == 5
        assert task["difficulty"] == "easy"
        assert task["max_steps"] == 50

    def test_constrained_task(self):
        from scheduling_env.tasks import create_constrained_scheduling_task

        task = create_constrained_scheduling_task()
        assert len(task["events"]) == 8
        assert task["difficulty"] == "medium"
        assert any(c.constraint_type == "lunch_break" for c in task["constraints"])

    def test_complex_task(self):
        from scheduling_env.tasks import create_complex_scheduling_task

        task = create_complex_scheduling_task()
        assert len(task["events"]) == 12
        assert task["difficulty"] == "hard"
        assert len(task["rooms"]) == 5

    def test_all_events_have_unique_ids(self):
        from scheduling_env.tasks import (
            create_simple_scheduling_task,
            create_constrained_scheduling_task,
            create_complex_scheduling_task,
        )
        for factory in [
            create_simple_scheduling_task,
            create_constrained_scheduling_task,
            create_complex_scheduling_task,
        ]:
            task = factory()
            ids = [e.id for e in task["events"]]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {factory.__name__}"


# =====================================================================
# Environment
# =====================================================================

class TestCalendarEnv:
    def test_reset(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        obs, info = env.reset(seed=42)

        assert "scheduled_events" in obs
        assert "available_slots" in obs
        assert "conflicts" in obs
        assert "time_remaining" in obs
        assert info["step"] == 0
        assert info["difficulty"] == "easy"

    def test_observation_shapes(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        obs, _ = env.reset(seed=0)

        assert obs["scheduled_events"].shape == (15, 4)
        assert obs["time_remaining"].shape == (1,)
        assert 0.0 <= obs["time_remaining"][0] <= 1.0

    def test_step_valid(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        env.reset(seed=0)

        valid = env.get_valid_actions()
        assert len(valid) > 0, "Should have valid actions at start"

        obs, reward, term, trunc, info = env.step(valid[0])
        assert isinstance(reward, float)
        assert isinstance(term, bool)

    def test_step_invalid_event(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        env.reset(seed=0)

        # Action that maps to event_idx >= num_events
        invalid = env.num_events * env.num_slots + 1
        if invalid < env.action_space.n:
            _, reward, _, _, info = env.step(invalid)
        else:
            # Force an invalid event index via large action
            _, reward, _, _, info = env.step(env.action_space.n - 1)

    def test_full_episode(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        obs, info = env.reset(seed=42)

        steps = 0
        done = False
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done, "Episode should terminate within max_steps"

    def test_get_state(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        env.reset(seed=0)

        valid = env.get_valid_actions()
        env.step(valid[0])

        state = env.get_state()
        assert "task_name" in state
        assert "schedule" in state
        assert "events_scheduled" in state
        assert state["events_scheduled"] == 1

    def test_get_valid_actions_decreases(self):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name="simple_scheduling")
        env.reset(seed=0)

        before = len(env.get_valid_actions())
        valid = env.get_valid_actions()
        env.step(valid[0])
        after = len(env.get_valid_actions())

        assert after < before, "Valid actions should decrease after scheduling"

    @pytest.mark.parametrize(
        "task",
        ["simple_scheduling", "constrained_scheduling", "complex_scheduling"],
    )
    def test_all_tasks_load(self, task):
        from scheduling_env import CalendarEnv

        env = CalendarEnv(task_name=task)
        obs, info = env.reset(seed=0)
        assert info["task"] == task

    def test_invalid_task_raises(self):
        from scheduling_env import CalendarEnv

        with pytest.raises(ValueError, match="Unknown task"):
            CalendarEnv(task_name="nonexistent_task")


# =====================================================================
# Grader
# =====================================================================

class TestScheduleGrader:
    def test_empty_schedule(self):
        from scheduling_env.grader import ScheduleGrader
        from scheduling_env.tasks import create_simple_scheduling_task

        task = create_simple_scheduling_task()
        grader = ScheduleGrader(task["events"], task["constraints"], task["rooms"])
        result = grader.grade({})

        assert result.events_scheduled == 0
        assert result.total_reward == 0.0

    def test_conflict_detection(self):
        from scheduling_env.grader import ScheduleGrader
        from scheduling_env.tasks import create_simple_scheduling_task

        task = create_simple_scheduling_task()
        grader = ScheduleGrader(task["events"], task["constraints"], task["rooms"])

        # Two events in same room at same time
        schedule = {
            0: {"start_hour": 9.0, "room": "Main Room"},
            1: {"start_hour": 9.0, "room": "Main Room"},
        }
        result = grader.grade(schedule)
        assert len(result.conflicts) > 0

    def test_no_conflict(self):
        from scheduling_env.grader import ScheduleGrader
        from scheduling_env.tasks import create_simple_scheduling_task

        task = create_simple_scheduling_task()
        grader = ScheduleGrader(task["events"], task["constraints"], task["rooms"])

        schedule = {
            0: {"start_hour": 9.0, "room": "Main Room"},
            1: {"start_hour": 10.0, "room": "Main Room"},
        }
        result = grader.grade(schedule)
        assert len(result.conflicts) == 0

    def test_grade_step(self):
        from scheduling_env.grader import ScheduleGrader
        from scheduling_env.tasks import create_simple_scheduling_task

        task = create_simple_scheduling_task()
        grader = ScheduleGrader(task["events"], task["constraints"], task["rooms"])

        reward, info = grader.grade_step(
            event=task["events"][0],
            start_hour=9.0,
            room="Main Room",
            current_schedule={},
        )
        assert reward > 0
        assert info["valid"]


# =====================================================================
# Baseline agents
# =====================================================================

class TestBaselineAgents:
    @pytest.mark.parametrize("agent_cls", ["RandomAgent", "GreedyAgent", "HeuristicAgent"])
    def test_agent_runs_episode(self, agent_cls):
        from scheduling_env import baseline

        agent = getattr(baseline, agent_cls)()
        result = baseline.run_episode(
            agent=agent,
            task_name="simple_scheduling",
            seed=0,
            verbose=False,
        )
        assert result["events_scheduled"] > 0
        assert "total_reward" in result

    def test_heuristic_beats_random(self):
        """Heuristic should outperform random on medium+ tasks over multiple seeds."""
        from scheduling_env.baseline import HeuristicAgent, RandomAgent, run_episode

        heuristic_rewards = []
        random_rewards = []
        for seed in range(5):
            h = run_episode(HeuristicAgent(), "constrained_scheduling", seed, False)
            r = run_episode(RandomAgent(), "constrained_scheduling", seed, False)
            heuristic_rewards.append(h["total_reward"])
            random_rewards.append(r["total_reward"])

        avg_h = sum(heuristic_rewards) / len(heuristic_rewards)
        avg_r = sum(random_rewards) / len(random_rewards)
        assert avg_h >= avg_r, (
            f"Heuristic ({avg_h:.1f}) should beat Random ({avg_r:.1f})"
        )


# =====================================================================
# Gym registration
# =====================================================================

class TestGymRegistration:
    def test_make_default(self):
        import gymnasium as gym
        import scheduling_env.registration  # triggers register

        env = gym.make("CalendarScheduling-v0")
        obs, info = env.reset(seed=0)
        assert "scheduled_events" in obs

    def test_make_easy(self):
        import gymnasium as gym
        import scheduling_env.registration

        env = gym.make("CalendarSchedulingEasy-v0")
        _, info = env.reset()
        assert info["difficulty"] == "easy"

    def test_make_medium(self):
        import gymnasium as gym
        import scheduling_env.registration

        env = gym.make("CalendarSchedulingMedium-v0")
        _, info = env.reset()
        assert info["difficulty"] == "medium"

    def test_make_hard(self):
        import gymnasium as gym
        import scheduling_env.registration

        env = gym.make("CalendarSchedulingHard-v0")
        _, info = env.reset()
        assert info["difficulty"] == "hard"

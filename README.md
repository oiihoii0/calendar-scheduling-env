---
title: Calendar Scheduling Env
emoji: 🗓️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# CalendarSchedulingEnv 📅


> An **OpenEnv**-compatible Gymnasium environment where AI agents learn to optimally schedule meetings and events on a calendar.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Overview

CalendarSchedulingEnv models the real-world challenge of scheduling meetings within an 8-hour work day (09:00–17:00). An agent must assign each event a **start time** and **room**, maximising priority-weighted utilisation while respecting hard and soft constraints.

### Key features

| Feature | Description |
|---------|-------------|
| **3 Difficulty Levels** | Easy → Medium → Hard with progressive constraint complexity |
| **Rich Constraint System** | No-overlap, lunch breaks, travel time, attendee availability |
| **Incremental Grading** | Per-step rewards for immediate feedback + episode-end bonuses |
| **Baseline Agents** | Random, Greedy, and Heuristic baselines included |
| **Full Gymnasium API** | `reset()` / `step()` / `render()` — plug into any RL framework |
| **OpenEnv `get_state()`** | Serialisable state snapshots for evaluation pipelines |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-team/calendar-scheduling-env.git
cd calendar-scheduling-env

# Install dependencies
pip install -r requirements.txt
```

### Docker

```bash
docker compose up --build
```

---

## ⚡ Quick Start

```python
from scheduling_env import CalendarEnv

# Create environment (easy / medium / hard)
env = CalendarEnv(task_name="simple_scheduling")     # easy
# env = CalendarEnv(task_name="constrained_scheduling") # medium
# env = CalendarEnv(task_name="complex_scheduling")     # hard

obs, info = env.reset(seed=42)
print(f"Task: {info['description']}")

done = False
while not done:
    action = env.action_space.sample()  # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()
print(env.get_state())
```

---

## 🎯 Task Difficulties

### Easy: `simple_scheduling`
- **5 events**, wide time windows, single room
- No room constraints, no attendee conflicts
- Goal: place all events without overlaps

### Medium: `constrained_scheduling`
- **8 events**, 3 rooms, room requirements
- Lunch break constraint (12:00–13:00)
- Attendee unavailability windows
- Goal: respect all hard constraints, minimise soft violations

### Hard: `complex_scheduling`
- **12 events**, 5 rooms across 2 buildings
- Travel time between buildings (15 min buffer)
- Hard lunch break, CEO availability limits
- Tight time windows with heavy contention
- Goal: maximise priority-weighted scheduling with zero hard violations

---

## 🧠 Action & Observation Spaces

### Action Space

Actions are encoded as a single integer:

```
action = event_index × num_slots + slot_index
slot_index = time_index × num_rooms + room_index
```

- `event_index`: which unscheduled event to place
- `time_index`: 30-minute slot (0 = 09:00, 1 = 09:30, …, 15 = 16:30)
- `room_index`: which room to assign

### Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `scheduled_events` | `(15, 4)` | Per-event: [norm_start, norm_duration, norm_priority, is_scheduled] |
| `available_slots` | `(N, 2)` | Per-slot: [norm_time, is_available] |
| `conflicts` | `Discrete` | Current number of conflicts |
| `time_remaining` | `(1,)` | Fraction of max steps remaining |

---

## 🏆 Reward Structure

| Component | Value | Description |
|-----------|-------|-------------|
| Valid placement | **+1.0** | Event placed in a valid slot |
| Priority bonus | **+0.2 × priority** | Higher-priority events earn more |
| Completion bonus | **+5.0 + 0.5 × N** | All events placed conflict-free |
| Room conflict | **−2.0** | Two events in same room at same time |
| Attendee conflict | **−2.0** | Shared attendees at same time |
| Hard violation | **−3.0** | Hard constraint broken |
| Soft violation | **−1.0** | Soft constraint broken |
| Invalid event | **−1.0** | Event index out of range |
| Already scheduled | **−0.5** | Attempting to re-schedule an event |

---

## 🤖 Baseline Agents

Three agents are included for benchmarking:

```bash
python -m scheduling_env.baseline
```

| Agent | Strategy |
|-------|----------|
| **RandomAgent** | Picks a random valid action |
| **GreedyAgent** | Highest priority first, earliest slot |
| **HeuristicAgent** | Priority ordering + constraint-aware + flexibility scoring |

### Running baselines

```python
from scheduling_env.baseline import HeuristicAgent, run_episode

result = run_episode(
    agent=HeuristicAgent(),
    task_name="complex_scheduling",
    seed=42,
    verbose=True,
)
print(result)
```

---

## 📊 Environment State (OpenEnv Spec)

The `get_state()` method returns a fully serialisable snapshot:

```python
state = env.get_state()
# {
#   "task_name": "constrained_scheduling",
#   "difficulty": "medium",
#   "step": 15,
#   "max_steps": 80,
#   "schedule": {
#     0: {"event_title": "Sprint Planning", "start_hour": 9.0, "room": "Conference A"},
#     ...
#   },
#   "events_scheduled": 6,
#   "events_total": 8,
#   "total_conflicts": 0,
#   "constraint_violations": 1,
#   "utilization_score": 0.6875,
#   "quality_score": 0.5875,
#   "episode_reward": 12.4
# }
```

---

## 🧪 Grading API

Use `ScheduleGrader` directly for custom evaluation:

```python
from scheduling_env import ScheduleGrader
from scheduling_env.tasks import create_constrained_scheduling_task

task = create_constrained_scheduling_task()
grader = ScheduleGrader(
    all_events=task["events"],
    constraints=task["constraints"],
    rooms=task["rooms"],
)

# Grade a complete schedule
schedule = {
    0: {"start_hour": 9.0, "room": "Conference A"},
    1: {"start_hour": 10.0, "room": "Conference B"},
}
result = grader.grade(schedule)
print(result)  # GradeResult(reward=..., scheduled=2/8, conflicts=0, violations=0)
```

---

## 🗂 Project Structure

```
scheduling_env/
├── __init__.py          # Package exports, version
├── env.py               # CalendarEnv (Gymnasium environment)
├── models.py            # Event, TimeSlot, CalendarConstraint, ScheduleState
├── tasks.py             # 3 task factories (easy/medium/hard)
├── grader.py            # ScheduleGrader + GradeResult
├── baseline.py          # RandomAgent, GreedyAgent, HeuristicAgent
requirements.txt         # Python dependencies
Dockerfile               # Container image
docker-compose.yml       # Local dev orchestration
README.md                # This file
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

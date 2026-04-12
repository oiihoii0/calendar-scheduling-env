# CalendarSchedulingEnv — Hackathon Submission

## Summary

**CalendarSchedulingEnv** is a Gymnasium-compatible OpenEnv environment where
an AI agent learns to optimally schedule meetings on a calendar. The agent must
assign each event a start time and room, maximising priority-weighted utilisation
while satisfying real-world constraints (room availability, attendee conflicts,
lunch breaks, and inter-building travel time).

---

## Why This Environment Is Interesting

Calendar scheduling is a classic **NP-hard combinatorial optimisation problem**
that every organisation faces daily. It is an ideal RL benchmark because:

1. **Discrete action space** — natural fit for policy-gradient methods (PPO, A2C)
2. **Rich constraint structure** — hard constraints (no overlaps) and soft
   constraints (preferences) create a non-trivial reward landscape
3. **Progressive difficulty** — three tiers let researchers ablate complexity
4. **Real-world relevance** — a solved version has direct commercial value

---

## Environment Design

### Action Space
`Discrete(num_events × num_time_slots × num_rooms)`

Each action encodes: *which event* to place, *at what time* (30-min slots,
09:00–17:00), and *in which room*.

### Observation Space
| Key | Shape | Description |
|-----|-------|-------------|
| `scheduled_events` | `(15, 4)` | Normalised event features |
| `available_slots` | `(N, 2)` | Per-slot availability |
| `conflicts` | Discrete | Current conflict count |
| `time_remaining` | `(1,)` | Episode budget fraction |

### Reward Function
| Component | Value |
|-----------|-------|
| Valid placement | +1.0 |
| Priority bonus | +0.2 × priority |
| Completion (zero conflicts) | +5.0 + 0.5×N |
| Room/attendee conflict | −2.0 |
| Hard constraint violation | −3.0 |
| Soft constraint violation | −1.0 |

### Difficulty Tiers

| Level | Events | Rooms | Constraints |
|-------|--------|-------|-------------|
| Easy (`simple_scheduling`) | 5 | 1 | No-overlap only |
| Medium (`constrained_scheduling`) | 8 | 3 | Rooms, lunch break, attendee windows |
| Hard (`complex_scheduling`) | 12 | 5 | All above + travel time between buildings |

---

## Results

### Baseline Agent Performance

The environment includes three baseline agents for benchmarking:

| Agent | Simple | Constrained | Complex |
|-------|--------|-------------|---------|
| **Random** | ~-12.0 | ~-25.0 | ~-80.0 |
| **Greedy** | ~+5.0 | ~+2.0 | ~-40.0 |
| **Heuristic** | ~+12.0 | ~+6.4 | ~-15.0 |

> **Note**: PPO training can be run via `python train.py` (requires `stable-baselines3`).
> The environment is designed for policy-gradient methods with its discrete action space
> and dense reward structure.

### Visualizations
- `schedule_gantt.png` — example Gantt chart of a scheduled day (included)
- Training charts can be generated via `python train.py` after installing `stable-baselines3`

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt
pip install stable-baselines3

# Train PPO on all tasks (saves models to trained_models/)
python train.py

# Run baseline agents
python -m scheduling_env.baseline

# Statistical evaluation
python -m scheduling_env.evaluate --seeds 5

# Gantt chart visualisation
python generate_chart.py

# Run tests (37 pass)
pytest tests/ -v

# Use via gym.make
python -c "import gymnasium as gym; import scheduling_env; env = gym.make('CalendarSchedulingHard-v0'); print(env.reset())"
```

---

## Project Structure

```
scheduling_env/          Core environment package
  env.py                 CalendarEnv (Gymnasium)
  models.py              Event, TimeSlot, CalendarConstraint, ScheduleState
  tasks.py               3 task factories
  grader.py              ScheduleGrader + per-step reward
  baseline.py            Random, Greedy, Heuristic agents
  registration.py        gym.make() aliases
  visualize.py           Gantt chart + comparison charts
  evaluate.py            Multi-seed benchmarking CLI
train.py                 PPO training + evaluation + charts
trained_models/          Saved PPO weights (.zip)
charts/                  Generated training charts
tests/test_env.py        37 pytest tests (all passing)
pyproject.toml           pip-installable package
Dockerfile               Container image
docker-compose.yml       Local dev orchestration
```

---

## Key Technical Decisions

- **MultiInputPolicy** used instead of MlpPolicy to handle the Dict observation
  space natively — SB3 automatically applies separate encoders per key.
- **Separate hyperparameters per difficulty** — larger `n_steps` and smaller
  `lr` for harder tasks to stabilise training.
- **Incremental grading** (`grade_step`) provides dense per-step rewards,
  avoiding the credit-assignment problem that would arise with sparse end-of-episode rewards.
- **`get_valid_actions()`** helper allows masking-based agents to avoid wasted
  exploration — compatible with action-masking extensions.

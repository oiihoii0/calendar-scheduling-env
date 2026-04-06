"""
FastAPI HTTP server for CalendarSchedulingEnv.

Exposes the OpenEnv-required endpoints:
  GET  /health          Health check — returns 200
  POST /reset           Reset environment, return first observation
  POST /step            Take one action, return obs/reward/done/info
  GET  /state           Return full serialisable state snapshot
  GET  /tasks           List all available tasks

The server manages multiple independent sessions via session_id.
Each session has its own CalendarEnv instance.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from scheduling_env.env import CalendarEnv

app = FastAPI(
    title="CalendarSchedulingEnv",
    description="OpenEnv-compatible calendar scheduling environment",
    version="1.0.0",
)

# In-memory session store: session_id -> CalendarEnv
_sessions: Dict[str, CalendarEnv] = {}

TASKS = [
    "simple_scheduling",
    "constrained_scheduling",
    "complex_scheduling",
]

# Max possible scores per task for normalisation
_MAX_SCORES = {
    "simple_scheduling":       17.5,   # 5 × 2.0 + 5.0 + 2.5
    "constrained_scheduling":  35.0,   # 8 × 2.0 + 5.0 + 4.0
    "complex_scheduling":      56.0,   # 12 × 2.0 + 5.0 + 6.0
}


# ── Request / response models ─────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "simple_scheduling"
    seed: int = 42
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    # Structured action — preferred for LLM agents
    event_title: Optional[str] = None   # e.g. "Team Sync"
    start_hour: Optional[float] = None  # e.g. 9.0
    room: Optional[str] = None          # e.g. "Main Room"
    # Raw integer action — alternative
    action: Optional[int] = None
    session_id: str = "default"


# ── Helpers ───────────────────────────────────────────────────────────

def _serialize_obs(env: CalendarEnv) -> Dict[str, Any]:
    """Human-readable observation for LLM consumption."""
    unscheduled = [
        {
            "id": ev.id,
            "title": ev.title,
            "duration_minutes": ev.duration_minutes,
            "earliest_start_hour": ev.earliest_start_hour,
            "latest_start_hour": ev.latest_start_hour,
            "priority": ev.priority,
            "attendees": ev.attendees,
            "room_required": ev.room_required,
        }
        for ev in env.all_events
        if ev.id not in env.scheduled_ids
    ]
    scheduled = [
        {
            "title": env.grader._event_map[eid].title,
            "start_hour": s["start_hour"],
            "end_hour": round(s["start_hour"] + env.grader._event_map[eid].duration_hours, 2),
            "room": s["room"],
        }
        for eid, s in env.schedule.items()
    ]
    return {
        "unscheduled_events": unscheduled,
        "scheduled_events": scheduled,
        "available_rooms": env.rooms,
        "conflicts": env.current_conflicts,
        "step": env.current_step,
        "max_steps": env.max_steps,
        "time_remaining": round(1.0 - env.current_step / env.max_steps, 4),
        "events_scheduled": len(env.scheduled_ids),
        "events_total": env.num_events,
    }


def _resolve_action(env: CalendarEnv, req: StepRequest) -> int:
    """Convert structured LLM action to integer action."""
    if req.action is not None:
        return req.action

    # Match by title
    event = None
    for ev in env.all_events:
        if ev.title.lower() == (req.event_title or "").lower():
            event = ev
            break

    if event is None:
        # Try partial match
        for ev in env.all_events:
            if (req.event_title or "").lower() in ev.title.lower():
                event = ev
                break

    if event is None:
        # Default: first unscheduled event
        for ev in env.all_events:
            if ev.id not in env.scheduled_ids:
                event = ev
                break

    if event is None:
        return 0

    event_idx = env.all_events.index(event)
    start_hour = req.start_hour if req.start_hour is not None else event.earliest_start_hour
    room = req.room if req.room else (event.room_required or env.rooms[0])

    # Clamp to valid range
    start_hour = max(env._DAY_START, min(env._DAY_END - event.duration_hours, start_hour))
    time_idx = int(round((start_hour - env._DAY_START) / (env._SLOT_DURATION / 60.0)))
    time_idx = max(0, min(env._TIME_SLOTS_PER_DAY - 1, time_idx))

    room_idx = env.rooms.index(room) if room in env.rooms else 0
    return event_idx * env.num_slots + time_idx * env.num_rooms + room_idx


def _normalize_score(reward: float, task_name: str) -> float:
    """Normalize cumulative reward to [0.0, 1.0]."""
    max_score = _MAX_SCORES.get(task_name, 20.0)
    return round(min(max(reward / max_score, 0.0), 1.0), 4)


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "simple_scheduling",
                "difficulty": "easy",
                "description": "Schedule 5 meetings in an 8-hour day. No room constraints.",
                "max_steps": 50,
            },
            {
                "name": "constrained_scheduling",
                "difficulty": "medium",
                "description": "Schedule 8 meetings with room requirements and lunch break.",
                "max_steps": 80,
            },
            {
                "name": "complex_scheduling",
                "difficulty": "hard",
                "description": "Schedule 12 meetings across 5 rooms with travel time constraints.",
                "max_steps": 120,
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    session_id = req.session_id or str(uuid4())
    if req.task_name not in TASKS:
        raise HTTPException(400, f"Unknown task. Must be one of: {TASKS}")

    env = CalendarEnv(task_name=req.task_name)
    obs_raw, info = env.reset(seed=req.seed)
    _sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": _serialize_obs(env),
        "info": info,
        "task": req.task_name,
        "difficulty": env.task_difficulty,
    }


@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")

    action = _resolve_action(env, req)
    obs_raw, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    score = _normalize_score(env.episode_reward, env.task_name)

    return {
        "observation": _serialize_obs(env),
        "reward": round(reward, 4),
        "done": done,
        "score": score,
        "info": {
            **info,
            "action_used": action,
            "normalized_score": score,
        },
    }


@app.get("/state")
def state(session_id: str = "default"):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    raw_state = env.get_state()
    raw_state["normalized_score"] = _normalize_score(
        raw_state["episode_reward"], env.task_name
    )
    return raw_state


@app.delete("/session/{session_id}")
def close_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"closed": session_id}

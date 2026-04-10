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
"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from scheduling_env.env import CalendarEnv

app = FastAPI(
    title="CalendarSchedulingEnv",
    description="OpenEnv-compatible calendar scheduling environment",
    version="1.0.0",
)

# ── In-memory session store ──────────────────────────────────────────
_sessions: Dict[str, CalendarEnv] = {}

TASKS = [
    "CalendarSchedulingEasy-v0",
    "CalendarSchedulingMedium-v0",
    "CalendarSchedulingHard-v0",
]

_MAX_SCORES = {
    "CalendarSchedulingEasy-v0":   17.5,
    "CalendarSchedulingMedium-v0": 35.0,
    "CalendarSchedulingHard-v0":   56.0,
}

# ── Dashboard UI ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CalendarSchedulingEnv — OpenEnv</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg-dark: #0a0a0f;
    --text-main: #e2e8f0;
    --primary: #7c3aed;
    --secondary: #2563eb;
    --accent: #a78bfa;
    --glass: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Inter',sans-serif;background:var(--bg-dark);color:var(--text-main);min-height:100vh;overflow-x:hidden}
  
  .bg{
    position:fixed;inset:0;
    background:
      radial-gradient(ellipse at 20% 20%, #1a0533 0%, transparent 50%),
      radial-gradient(ellipse at 80% 80%, #0d1f3c 0%, transparent 50%),
      var(--bg-dark);
    z-index:-1;
  }

  .container{max-width:1100px;margin:0 auto;padding:48px 24px}

  /* Hero */
  .hero{text-align:center;margin-bottom:64px}
  .badge{display:inline-flex;align-items:center;gap:8px;background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.3);border-radius:999px;padding:6px 16px;font-size:13px;color:var(--accent);margin-bottom:24px;letter-spacing:0.5px}
  .badge::before{content:'';width:8px;height:8px;border-radius:50%;background:var(--accent);animation:pulse-dot 2s infinite}
  @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(1.3)}}
  
  h1{font-size:clamp(2rem,5vw,3.5rem);font-weight:800;background:linear-gradient(135deg,#fff 0%,var(--accent) 50%,#60a5fa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;margin-bottom:16px}
  .subtitle{font-size:1.1rem;color:#94a3b8;max-width:600px;margin:0 auto 32px;line-height:1.6}
  
  .hero-actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
  .btn{padding:12px 28px;border-radius:12px;font-size:14px;font-weight:600;text-decoration:none;transition:all 0.3s;cursor:pointer;border:none}
  .btn-primary{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;box-shadow:0 4px 20px rgba(124,58,237,0.3)}
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(124,58,237,0.4)}
  .btn-ghost{background:var(--glass);color:#cbd5e1;border:1px solid var(--glass-border); backdrop-filter: blur(10px); }
  .btn-ghost:hover{background:rgba(255,255,255,0.1);color:#fff}

  /* Stats bar */
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:64px}
  .stat{background:var(--glass);border:1px solid var(--glass-border);border-radius:16px;padding:24px;text-align:center; backdrop-filter: blur(5px); }
  .stat-num{font-size:2rem;font-weight:800;background:linear-gradient(135deg,var(--accent),#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-label{font-size:13px;color:#64748b;margin-top:4px}

  /* Tasks */
  .section-title{font-size:1.5rem;font-weight:700;color:#f1f5f9;margin-bottom:24px}
  .tasks{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:64px}
  .task-card{background:var(--glass);border:1px solid var(--glass-border);border-radius:20px;padding:28px;transition:all 0.35s;position:relative;overflow:hidden; backdrop-filter: blur(5px); }
  .task-card:hover{transform:translateY(-6px);border-color:rgba(139,92,246,0.5);background:rgba(139,92,246,0.06)}
  
  .difficulty{display:inline-block;padding:4px 12px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
  .easy .difficulty{background:rgba(16,185,129,0.15);color:#10b981}
  .medium .difficulty{background:rgba(245,158,11,0.15);color:#f59e0b}
  .hard .difficulty{background:rgba(239,68,68,0.15);color:#ef4444}
  
  .task-name{font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:8px}
  .task-desc{font-size:14px;color:#94a3b8;line-height:1.6;margin-bottom:16px}
  .task-meta{display:flex;gap:16px}
  .task-meta span{font-size:12px;color:#475569;display:flex;align-items:center;gap:4px}

  /* Endpoints */
  .endpoints{margin-bottom:64px}
  .endpoint{display:flex;align-items:center;gap:16px;padding:16px 20px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;margin-bottom:8px;transition:all 0.2s}
  .endpoint:hover{background:rgba(255,255,255,0.05);border-color:rgba(255,255,255,0.1); transform: translateX(5px); }
  .method{padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700;min-width:52px;text-align:center; font-family: monospace; }
  .get{background:rgba(59,130,246,0.2);color:#60a5fa}
  .post{background:rgba(16,185,129,0.2);color:#10b981}
  .delete{background:rgba(239,68,68,0.2);color:#ef4444}
  .path{font-family: monospace;font-size:14px;color:#e2e8f0;flex:1}
  .ep-desc{font-size:13px;color:#475569}

  /* Footer */
  .footer{text-align:center;padding-top:32px;border-top:1px solid rgba(255,255,255,0.05);color:#334155;font-size:13px}
  .footer a{color:var(--accent);text-decoration:none}
</style>
</head>
<body>
<div class="bg"></div>
<div class="container">

  <div class="hero">
    <div class="badge">🚀 Live on HuggingFace Spaces</div>
    <h1>Calendar Scheduling<br/>Environment</h1>
    <p class="subtitle">An OpenEnv-compatible AI training environment where agents learn to optimally schedule meetings — respecting rooms, attendees, priorities and real-world constraints.</p>
    <div class="hero-actions">
      <a href="/docs" class="btn btn-primary">⚡ Try the API</a>
      <a href="/tasks" class="btn btn-ghost">📋 View Tasks</a>
      <a href="/health" class="btn btn-ghost">💚 Health Check</a>
    </div>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-num">3</div><div class="stat-label">Difficulty Levels</div></div>
    <div class="stat"><div class="stat-num">12</div><div class="stat-label">Max Events (Hard)</div></div>
    <div class="stat"><div class="stat-num">5</div><div class="stat-label">Rooms Available</div></div>
    <div class="stat"><div class="stat-num">0–1</div><div class="stat-label">Normalised Score</div></div>
  </div>

  <div class="section-title">🎯 Challenge Tasks</div>
  <div class="tasks">
    <div class="task-card easy">
      <span class="difficulty">Easy</span>
      <div class="task-name">Simple Scheduling</div>
      <div class="task-desc">Schedule 5 meetings in an 8-hour work day. Wide time windows, no room constraints — perfect for learning the basics.</div>
      <div class="task-meta">
        <span>📅 5 events</span>
        <span>🏠 1 room</span>
        <span>⏱️ 50 max steps</span>
      </div>
    </div>
    <div class="task-card medium">
      <span class="difficulty">Medium</span>
      <div class="task-name">Constrained Scheduling</div>
      <div class="task-desc">Schedule 8 meetings across 3 rooms with specific room requirements, lunch break constraints and attendee conflicts.</div>
      <div class="task-meta">
        <span>📅 8 events</span>
        <span>🏠 3 rooms</span>
        <span>⏱️ 80 max steps</span>
      </div>
    </div>
    <div class="task-card hard">
      <span class="difficulty">Hard</span>
      <div class="task-name">Complex Scheduling</div>
      <div class="task-desc">Schedule 12 meetings across 5 rooms in 2 buildings with travel time constraints between floors and tight time windows.</div>
      <div class="task-meta">
        <span>📅 12 events</span>
        <span>🏠 5 rooms</span>
        <span>⏱️ 120 max steps</span>
      </div>
    </div>
  </div>

  <div class="section-title">🔌 API Endpoints</div>
  <div class="endpoints">
    <div class="endpoint"><span class="method get">GET</span><span class="path">/health</span><span class="ep-desc">Check server status</span></div>
    <div class="endpoint"><span class="method get">GET</span><span class="path">/tasks</span><span class="ep-desc">List all available tasks</span></div>
    <div class="endpoint"><span class="method post">POST</span><span class="path">/reset</span><span class="ep-desc">Start a new episode — returns initial observation</span></div>
    <div class="endpoint"><span class="method post">POST</span><span class="path">/step</span><span class="ep-desc">Schedule one meeting — returns reward & next observation</span></div>
    <div class="endpoint"><span class="method get">GET</span><span class="path">/state</span><span class="ep-desc">Get full calendar state snapshot</span></div>
    <div class="endpoint"><span class="method delete">DELETE</span><span class="path">/session/{id}</span><span class="ep-desc">Close a session</span></div>
  </div>

  <div class="footer">
    Built for the <strong>OpenEnv Hackathon</strong> &nbsp;·&nbsp; Powered by <a href="https://gymnasium.farama.org/">Gymnasium</a> &nbsp;·&nbsp; <a href="/docs">API Docs →</a>
  </div>
</div>
</body>
</html>"""

# ── OpenEnv Endpoints ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "CalendarSchedulingEasy-v0"
    seed: int = 42
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    event_title: Optional[str] = None
    start_hour: Optional[float] = None
    room: Optional[str] = None
    action: Optional[int] = None
    session_id: str = "default"

def _serialize_obs(env: CalendarEnv) -> Dict[str, Any]:
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
    if req.action is not None:
        return req.action
    event = next((ev for ev in env.all_events if ev.title.lower() == (req.event_title or "").lower()), None)
    if not event:
        event = next((ev for ev in env.all_events if (req.event_title or "").lower() in ev.title.lower()), None)
    if not event:
        event = next((ev for ev in env.all_events if ev.id not in env.scheduled_ids or True), None)
    
    event_idx = env.all_events.index(event)
    start_hour = req.start_hour if req.start_hour is not None else event.earliest_start_hour
    room = req.room if req.room else (event.room_required or env.rooms[0])
    
    start_hour = max(env._DAY_START, min(env._DAY_END - event.duration_hours, start_hour))
    time_idx = int(round((start_hour - env._DAY_START) / (env._SLOT_DURATION / 60.0)))
    time_idx = max(0, min(env._TIME_SLOTS_PER_DAY - 1, time_idx))
    room_idx = env.rooms.index(room) if room in env.rooms else 0
    return event_idx * env.num_slots + time_idx * env.num_rooms + room_idx

def _normalize_score(reward: float, task_name: str) -> float:
    max_score = _MAX_SCORES.get(task_name, 20.0)
    return round(min(max(reward / max_score, 0.0), 1.0), 4)

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "CalendarSchedulingEasy-v0",
                "env_id": "CalendarSchedulingEasy-v0",
                "name": "Simple Scheduling",
                "difficulty": "easy",
                "description": "Schedule 5 meetings in an 8-hour day. No room constraints.",
                "max_steps": 50,
                "grader": "scheduling_env.grader:grade_schedule",
                "has_grader": True,
            },
            {
                "id": "CalendarSchedulingMedium-v0",
                "env_id": "CalendarSchedulingMedium-v0",
                "name": "Constrained Scheduling",
                "difficulty": "medium",
                "description": "Schedule 8 meetings with room requirements and lunch break.",
                "max_steps": 80,
                "grader": "scheduling_env.grader:grade_schedule",
                "has_grader": True,
            },
            {
                "id": "CalendarSchedulingHard-v0",
                "env_id": "CalendarSchedulingHard-v0",
                "name": "Complex Scheduling",
                "difficulty": "hard",
                "description": "Schedule 12 meetings across 5 rooms with travel time constraints.",
                "max_steps": 120,
                "grader": "scheduling_env.grader:grade_schedule",
                "has_grader": True,
            },
        ]
    }

@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    session_id = req.session_id or str(uuid4())
    if req.task_name not in TASKS:
        raise HTTPException(400, f"Unknown task. Must be one of: {TASKS}")
    env = CalendarEnv(task_name=req.task_name)
    obs_raw, info = env.reset(seed=req.seed)
    _sessions[session_id] = env
    return {"session_id": session_id, "observation": _serialize_obs(env), "info": info, "task": req.task_name}

@app.post("/step")
def step(req: StepRequest = Body(default=StepRequest())):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found.")
    action = _resolve_action(env, req)
    obs_raw, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    score = _normalize_score(env.episode_reward, env.task_name)
    return {
        "observation": _serialize_obs(env),
        "reward": round(reward, 4),
        "done": done,
        "score": score,
        "info": {**info, "action_used": action, "normalized_score": score},
    }

@app.get("/state")
def state(session_id: str = "default"):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    raw_state = env.get_state()
    raw_state["normalized_score"] = _normalize_score(raw_state["episode_reward"], env.task_name)
    return raw_state

@app.delete("/session/{session_id}")
def close_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"closed": session_id}
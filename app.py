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
    <title>CalendarSchedulingEnv — Premium AI Training Environment</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
    <style>
        :root {
            --bg: #020617;
            --slate-900: #0f172a;
            --violet-500: #8b5cf6;
            --cyan-400: #22d3ee;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --glass: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glow: 0 0 20px rgba(139, 92, 246, 0.3);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Outfit', sans-serif; 
            background-color: var(--bg); 
            color: var(--text-primary); 
            min-height: 100vh;
            overflow-x: hidden;
            line-height: 1.5;
        }

        /* Animated Mesh Background */
        .mesh-bg {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: -1;
            background: 
                radial-gradient(circle at 0% 0%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(34, 211, 238, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 1) 0%, var(--bg) 100%);
            overflow: hidden;
        }
        .mesh-bg::after {
            content: "";
            position: absolute;
            inset: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            opacity: 0.05;
            pointer-events: none;
        }

        .container { max-width: 1200px; margin: 0 auto; padding: 4rem 2rem; position: relative; }

        /* Navigation */
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6rem;
            animation: slideDown 0.8s ease-out forwards;
        }
        .logo { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.05em; display: flex; align-items: center; gap: 10px;}
        .logo span { background: linear-gradient(135deg, var(--violet-500), var(--cyan-400)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .logo-icon { width: 32px; height: 32px; background: var(--violet-500); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: var(--glow); }

        /* Hero Section */
        .hero { text-align: center; margin-bottom: 8rem; }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 99rem;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--violet-500);
            margin-bottom: 2rem;
            animation: fadeIn 1s ease-out forwards;
        }
        .hero h1 {
            font-size: clamp(2.5rem, 8vw, 4.5rem);
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
            margin-bottom: 1.5rem;
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .hero h1 span { color: var(--text-secondary); }
        .hero p {
            font-size: 1.25rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto 3rem;
            animation: slideUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .cta-group { 
            display: flex; gap: 1rem; justify-content: center; 
            animation: slideUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .btn {
            padding: 12px 32px;
            border-radius: 12px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.3s;
            cursor: pointer;
            border: none;
            font-size: 1rem;
        }
        .btn-primary { background: var(--violet-500); color: white; box-shadow: var(--glow); }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(139, 92, 246, 0.5); }
        .btn-secondary { background: var(--glass); color: var(--text-primary); border: 1px solid var(--glass-border); backdrop-filter: blur(10px); }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.2); }

        /* Grid */
        .section-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 2.5rem; display: flex; align-items: center; gap: 12px; }
        .section-title::before { content: ""; width: 24px; height: 4px; background: var(--violet-500); border-radius: 2px; }

        .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-bottom: 6rem; }
        .card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }
        .card:hover {
            transform: translateY(-8px);
            border-color: rgba(139, 92, 246, 0.4);
            background: rgba(139, 92, 246, 0.04);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        .card::after {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(circle at top right, rgba(139, 92, 246, 0.1), transparent 70%);
            pointer-events: none;
        }

        .difficulty { 
            display: inline-block; padding: 4px 12px; border-radius: 8px; font-size: 0.75rem; 
            font-weight: 700; text-transform: uppercase; margin-bottom: 1.5rem; letter-spacing: 0.05em;
        }
        .easy .difficulty { background: rgba(34, 197, 94, 0.1); color: #22c55e; }
        .medium .difficulty { background: rgba(245, 158, 11, 0.1); color: #f59e0b; }
        .hard .difficulty { background: rgba(239, 68, 68, 0.1); color: #ef4444; }

        .card h3 { font-size: 1.25rem; margin-bottom: 0.75rem; }
        .card p { font-size: 0.9375rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem; }
        .card-stats { display: flex; gap: 1rem; }
        .card-stats span { font-size: 0.8125rem; color: var(--text-secondary); display: flex; align-items: center; gap: 6px; }

        /* Endpoints */
        .endpoint-list { display: flex; flex-direction: column; gap: 0.75rem; }
        .endpoint {
            display: flex; align-items: center; gap: 1rem; padding: 1rem 1.5rem;
            background: rgba(255, 255, 255, 0.02); border: 1px solid var(--glass-border);
            border-radius: 16px; transition: all 0.2s;
        }
        .endpoint:hover { background: rgba(255, 255, 255, 0.05); transform: translateX(8px); }
        .method {
            padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;
            min-width: 60px; text-align:center; font-family: monospace;
        }
        .get { background: rgba(56, 189, 248, 0.15); color: #38bdf8; }
        .post { background: rgba(34, 211, 238, 0.15); color: #22d3ee; }
        .delete { background: rgba(248, 113, 113, 0.15); color: #f87171; }
        .path { font-family: monospace; font-size: 0.9375rem; color: var(--text-primary); flex: 1; }
        .desc { font-size: 0.875rem; color: var(--text-secondary); }

        /* Animations */
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { 
            from { opacity: 0; transform: translateY(30px); } 
            to { opacity: 1; transform: translateY(0); } 
        }
        @keyframes slideDown { 
            from { opacity: 0; transform: translateY(-20px); } 
            to { opacity: 1; transform: translateY(0); } 
        }

        @media (max-width: 768px) {
            .container { padding: 2rem 1.5rem; }
            .hero h1 { font-size: 2.5rem; }
            .cta-group { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="mesh-bg"></div>
    <div class="container fade-in">
        <nav>
            <div class="logo">
                <div class="logo-icon">📅</div>
                Calendar<span>Scheduling</span>Env
            </div>
            <div class="nav-links">
                <a href="/docs" class="btn btn-secondary" style="padding: 8px 20px; font-size: 0.875rem;">Documentation</a>
            </div>
        </nav>

        <section class="hero">
            <div class="badge">
                <span style="font-size: 1.2rem;">✨</span> Powered by Gymnasium & OpenEnv
            </div>
            <h1>The Intelligent<br/><span>Scheduling API</span></h1>
            <p>A high-fidelity AI environment where agents learn to solve complex scheduling conflicts, manage room resources, and optimize attendee priorities in real-time.</p>
            <div class="cta-group">
                <a href="/docs" class="btn btn-primary">Start Training</a>
                <a href="/health" class="btn btn-secondary">System Status</a>
            </div>
        </section>

        <h2 class="section-title">Challenge Environments</h2>
        <div class="feature-grid">
            <div class="card easy">
                <span class="difficulty">Beginner</span>
                <h3>Simple Scheduling</h3>
                <p>Perfect for initial agent warm-up. 5 events, 8 hours, and plenty of room. Focuses on basic sequential scheduling.</p>
                <div class="card-stats">
                    <span>🗓️ 5 Events</span>
                    <span>🏢 1 Room</span>
                </div>
            </div>
            <div class="card medium">
                <span class="difficulty">Intermediate</span>
                <h3>Constrained Flow</h3>
                <p>The complexity rises. 8 events with room-specific requirements and unavoidable lunch breaks. Requires conflict resolution.</p>
                <div class="card-stats">
                    <span>🗓️ 8 Events</span>
                    <span>🏢 3 Rooms</span>
                </div>
            </div>
            <div class="card hard">
                <span class="difficulty">Advanced</span>
                <h3>Global Resource Crisis</h3>
                <p>The ultimate test. 12 events across 5 rooms in multiple buildings. Minimal time windows and maximum priority conflicts.</p>
                <div class="card-stats">
                    <span>🗓️ 12 Events</span>
                    <span>🏢 5 Rooms</span>
                </div>
            </div>
        </div>

        <h2 class="section-title">Protocol Endpoints</h2>
        <div class="endpoint-list">
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="path">/reset</span>
                <span class="desc">Initialize a new training session</span>
            </div>
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="path">/step</span>
                <span class="desc">Execute a scheduling action</span>
            </div>
            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="path">/state</span>
                <span class="desc">Query current environment snapshot</span>
            </div>
            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="path">/tasks</span>
                <span class="desc">Retrieve available environment IDs</span>
            </div>
        </div>

        <footer style="margin-top: 8rem; text-align: center; color: var(--text-secondary); font-size: 0.875rem; opacity: 0.6;">
            &copy; 2026 OpenEnv hackathon. Built for performance and precision.
        </footer>
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
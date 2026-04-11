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
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalendarSchedulingEnv | OpenEnv Server</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --dark-bg: #0f0f1a;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #a0a0b0;
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --glow-purple: 0 0 40px rgba(139, 92, 246, 0.3);
            --glow-blue: 0 0 40px rgba(59, 130, 246, 0.3);
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--dark-bg);
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
            position: relative;
        }
        
        /* Animated Background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .bg-animation::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(99, 102, 241, 0.1) 0%, transparent 40%);
            animation: float 20s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            33% { transform: translate(30px, -30px) rotate(1deg); }
            66% { transform: translate(-20px, 20px) rotate(-1deg); }
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 24px;
        }
        
        /* Hero Section */
        .hero {
            text-align: center;
            padding: 60px 0 40px;
            position: relative;
        }
        
        .hero-icon {
            width: 100px;
            height: 100px;
            background: var(--primary-gradient);
            border-radius: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 24px;
            font-size: 48px;
            box-shadow: var(--glow-purple);
            animation: pulse 3s ease-in-out infinite;
            position: relative;
        }
        
        .hero-icon::after {
            content: '';
            position: absolute;
            inset: -4px;
            border-radius: 32px;
            background: var(--primary-gradient);
            opacity: 0.4;
            filter: blur(20px);
            z-index: -1;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .hero h1 {
            font-size: 3.5rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }
        
        .hero .subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto 32px;
            line-height: 1.6;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 50px;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--accent-green);
        }
        
        .status-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: blink 2s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        
        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-4px);
            border-color: rgba(139, 92, 246, 0.4);
            box-shadow: var(--glow-purple);
        }
        
        .stat-card i {
            font-size: 2rem;
            margin-bottom: 12px;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-card .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        
        .stat-card .stat-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        /* Section Styles */
        .section {
            margin: 60px 0;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .section-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .section-header i {
            font-size: 1.25rem;
            color: var(--accent-purple);
        }
        
        /* Endpoints Grid */
        .endpoints-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }
        
        .endpoint-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 16px;
            transition: all 0.3s ease;
            cursor: pointer;
            backdrop-filter: blur(10px);
            text-decoration: none;
            color: inherit;
        }
        
        .endpoint-card:hover {
            transform: translateY(-2px);
            border-color: rgba(59, 130, 246, 0.5);
            box-shadow: var(--glow-blue);
        }
        
        .method-badge {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .method-badge.get {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
        }
        
        .method-badge.post {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
        }
        
        .endpoint-info {
            flex: 1;
        }
        
        .endpoint-path {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: var(--text-primary);
            font-weight: 500;
            margin-bottom: 4px;
        }
        
        .endpoint-desc {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .endpoint-card i.arrow {
            color: var(--text-secondary);
            opacity: 0;
            transition: all 0.3s ease;
        }
        
        .endpoint-card:hover i.arrow {
            opacity: 1;
            transform: translateX(4px);
        }
        
        /* Tasks Section */
        .tasks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .task-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .task-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
        }
        
        .task-card.easy::before { background: var(--success-gradient); }
        .task-card.medium::before { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
        .task-card.hard::before { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); }
        
        .task-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .task-card.easy:hover { box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15); }
        .task-card.medium:hover { box-shadow: 0 20px 40px rgba(250, 112, 154, 0.15); }
        .task-card.hard:hover { box-shadow: 0 20px 40px rgba(255, 8, 68, 0.15); }
        
        .task-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        
        .task-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        
        .task-card.easy .task-icon { background: rgba(16, 185, 129, 0.1); }
        .task-card.medium .task-icon { background: rgba(250, 112, 154, 0.1); }
        .task-card.hard .task-icon { background: rgba(255, 8, 68, 0.1); }
        
        .task-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .task-difficulty {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            margin-left: auto;
        }
        
        .task-card.easy .task-difficulty { background: rgba(16, 185, 129, 0.15); color: #34d399; }
        .task-card.medium .task-difficulty { background: rgba(250, 112, 154, 0.15); color: #fb7185; }
        .task-card.hard .task-difficulty { background: rgba(255, 8, 68, 0.15); color: #fb7185; }
        
        .task-desc {
            color: var(--text-secondary);
            font-size: 0.9375rem;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .task-meta {
            display: flex;
            gap: 20px;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .task-meta span {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .task-meta i {
            color: var(--accent-purple);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 40px 0;
            border-top: 1px solid var(--border-color);
            margin-top: 60px;
        }
        
        .footer p {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        .footer .highlight {
            color: var(--accent-purple);
            font-weight: 500;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 { font-size: 2.5rem; }
            .hero .subtitle { font-size: 1rem; }
            .endpoints-grid, .tasks-grid { grid-template-columns: 1fr; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-purple);
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    
    <div class="container">
        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-icon">🗓️</div>
            <h1>CalendarSchedulingEnv</h1>
            <p class="subtitle">A modern OpenEnv-compatible Gymnasium environment where AI agents learn to optimally schedule meetings and events on a calendar.</p>
            <div class="status-badge">
                <span>Server Running</span>
            </div>
        </section>
        
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <i class="fas fa-calendar-alt"></i>
                <div class="stat-value">3</div>
                <div class="stat-label">Difficulty Levels</div>
            </div>
            <div class="stat-card">
                <i class="fas fa-tasks"></i>
                <div class="stat-value">25</div>
                <div class="stat-label">Total Events</div>
            </div>
            <div class="stat-card">
                <i class="fas fa-door-open"></i>
                <div class="stat-value">9</div>
                <div class="stat-label">Meeting Rooms</div>
            </div>
            <div class="stat-card">
                <i class="fas fa-check-circle"></i>
                <div class="stat-value">37</div>
                <div class="stat-label">Tests Passing</div>
            </div>
        </div>
        
        <!-- API Endpoints -->
        <section class="section">
            <div class="section-header">
                <i class="fas fa-plug"></i>
                <h2>API Endpoints</h2>
            </div>
            <div class="endpoints-grid">
                <a href="/health" class="endpoint-card">
                    <span class="method-badge get">GET</span>
                    <div class="endpoint-info">
                        <div class="endpoint-path">/health</div>
                        <div class="endpoint-desc">Check server health status</div>
                    </div>
                    <i class="fas fa-chevron-right arrow"></i>
                </a>
                <a href="/tasks" class="endpoint-card">
                    <span class="method-badge get">GET</span>
                    <div class="endpoint-info">
                        <div class="endpoint-path">/tasks</div>
                        <div class="endpoint-desc">List all available tasks</div>
                    </div>
                    <i class="fas fa-chevron-right arrow"></i>
                </a>
                <a href="#" onclick="alert('Use POST method to /reset'); return false;" class="endpoint-card">
                    <span class="method-badge post">POST</span>
                    <div class="endpoint-info">
                        <div class="endpoint-path">/reset</div>
                        <div class="endpoint-desc">Reset environment with task & seed</div>
                    </div>
                    <i class="fas fa-chevron-right arrow"></i>
                </a>
                <a href="#" onclick="alert('Use POST method to /step'); return false;" class="endpoint-card">
                    <span class="method-badge post">POST</span>
                    <div class="endpoint-info">
                        <div class="endpoint-path">/step</div>
                        <div class="endpoint-desc">Take scheduling action</div>
                    </div>
                    <i class="fas fa-chevron-right arrow"></i>
                </a>
                <a href="/state?session_id=default" class="endpoint-card">
                    <span class="method-badge get">GET</span>
                    <div class="endpoint-info">
                        <div class="endpoint-path">/state</div>
                        <div class="endpoint-desc">Get full environment state</div>
                    </div>
                    <i class="fas fa-chevron-right arrow"></i>
                </a>
            </div>
        </section>
        
        <!-- Tasks -->
        <section class="section">
            <div class="section-header">
                <i class="fas fa-layer-group"></i>
                <h2>Scheduling Tasks</h2>
            </div>
            <div class="tasks-grid">
                <div class="task-card easy">
                    <div class="task-header">
                        <div class="task-icon">🟢</div>
                        <div>
                            <div class="task-title">Simple Scheduling</div>
                        </div>
                        <span class="task-difficulty">Easy</span>
                    </div>
                    <p class="task-desc">Perfect for getting started. Schedule 5 meetings in an 8-hour work day with wide time windows and no room constraints.</p>
                    <div class="task-meta">
                        <span><i class="fas fa-calendar"></i> 5 Events</span>
                        <span><i class="fas fa-door-closed"></i> 1 Room</span>
                        <span><i class="fas fa-clock"></i> 50 Steps</span>
                    </div>
                </div>
                
                <div class="task-card medium">
                    <div class="task-header">
                        <div class="task-icon">🟡</div>
                        <div>
                            <div class="task-title">Constrained Scheduling</div>
                        </div>
                        <span class="task-difficulty">Medium</span>
                    </div>
                    <p class="task-desc">Step up the challenge. Schedule 8 meetings with room requirements, lunch break constraints, and attendee unavailability windows.</p>
                    <div class="task-meta">
                        <span><i class="fas fa-calendar"></i> 8 Events</span>
                        <span><i class="fas fa-door-closed"></i> 3 Rooms</span>
                        <span><i class="fas fa-clock"></i> 80 Steps</span>
                    </div>
                </div>
                
                <div class="task-card hard">
                    <div class="task-header">
                        <div class="task-icon">🔴</div>
                        <div>
                            <div class="task-title">Complex Scheduling</div>
                        </div>
                        <span class="task-difficulty">Hard</span>
                    </div>
                    <p class="task-desc">The ultimate test. Schedule 12 meetings across 5 rooms in 2 buildings with travel time constraints, hard lunch breaks, and CEO availability limits.</p>
                    <div class="task-meta">
                        <span><i class="fas fa-calendar"></i> 12 Events</span>
                        <span><i class="fas fa-door-closed"></i> 5 Rooms</span>
                        <span><i class="fas fa-clock"></i> 120 Steps</span>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer class="footer">
            <p>Built for <span class="highlight">Meta PyTorch Hackathon</span> • Powered by FastAPI & Gymnasium</p>
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
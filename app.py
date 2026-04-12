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
    return '''<!DOCTYPE html>

<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>CalendarSchedulingEnv | OpenEnv Server</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&amp;family=Space+Grotesk:wght@300;400;500;600;700&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<script id="tailwind-config">
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    "colors": {
                        "secondary": "#53ddfc",
                        "background": "#0e0e13",
                        "on-surface-variant": "#acaab1",
                        "surface-variant": "#25252d",
                        "outline": "#76747b",
                        "outline-variant": "#48474d",
                        "surface-bright": "#2c2b33",
                        "primary": "#6bff8f",
                        "primary-container": "#0abc56",
                        "tertiary": "#7de9ff",
                        "error": "#ff7351",
                        "surface-container": "#19191f",
                        "surface-container-highest": "#25252d",
                        "surface": "#0e0e13",
                        "on-surface": "#f9f5fd",
                        "surface-container-high": "#1f1f26",
                        "surface-container-low": "#131319",
                        "on-primary": "#005f28",
                    },
                    "borderRadius": {
                        "DEFAULT": "0.5rem",
                        "lg": "0.75rem",
                        "xl": "1rem",
                    },
                    "fontFamily": {
                        "headline": ["Inter"],
                        "body": ["Inter"],
                        "label": ["Space Grotesk"]
                    }
                },
            },
        }
    </script>
<style>
        body {
            background: linear-gradient(135deg, #0e0e13 0%, #1a1a24 50%, #0e0e13 100%);
        }
        .glass-panel {
            background: rgba(25, 25, 31, 0.6);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.02);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-panel:hover {
            background: rgba(35, 35, 45, 0.7);
            border-color: rgba(107, 255, 143, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(107, 255, 143, 0.1), 0 0 20px rgba(107, 255, 143, 0.1);
            transform: translateY(-2px);
        }
        .wireframe-grid {
            background-image: linear-gradient(rgba(107, 255, 143, 0.03) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(107, 255, 143, 0.03) 1px, transparent 1px);
            background-size: 30px 30px;
            animation: gridMove 20s linear infinite;
        }
        @keyframes gridMove {
            0% { transform: perspective(1000px) rotateY(-20deg) scale(1.2) translateX(0); }
            100% { transform: perspective(1000px) rotateY(-20deg) scale(1.2) translateX(30px); }
        }
        .glow-text {
            text-shadow: 0 0 20px rgba(107, 255, 143, 0.3);
            animation: pulseGlow 3s ease-in-out infinite;
            background: linear-gradient(135deg, #6bff8f 0%, #7de9ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        @keyframes pulseGlow {
            0%, 100% { filter: drop-shadow(0 0 20px rgba(107, 255, 143, 0.3)); }
            50% { filter: drop-shadow(0 0 30px rgba(107, 255, 143, 0.5)); }
        }
        .stat-card {
            opacity: 0;
            transform: translateY(30px);
            animation: slideUp 0.6s ease-out forwards;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-color, #6bff8f), transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .stat-card:hover::before {
            opacity: 1;
        }
        .stat-card:nth-child(1) { animation-delay: 0.1s; --accent-color: #6bff8f; }
        .stat-card:nth-child(2) { animation-delay: 0.2s; --accent-color: #53ddfc; }
        .stat-card:nth-child(3) { animation-delay: 0.3s; --accent-color: #7de9ff; }
        .stat-card:nth-child(4) { animation-delay: 0.4s; --accent-color: #6bff8f; }
        @keyframes slideUp {
            to { opacity: 1; transform: translateY(0); }
        }
        .task-card {
            opacity: 0;
            transform: translateX(-30px);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .task-card.visible {
            opacity: 1;
            transform: translateX(0);
        }
        .task-card:hover {
            transform: translateY(-4px) scale(1.02);
        }
        .task-card .wireframe-grid {
            transition: opacity 0.3s;
        }
        .task-card:hover .wireframe-grid {
            opacity: 0.6;
        }
        .endpoint-row {
            opacity: 0;
            transform: translateX(20px);
            transition: all 0.3s ease;
            position: relative;
        }
        .endpoint-row::after {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 0;
            height: 100%;
            background: linear-gradient(90deg, rgba(107, 255, 143, 0.1), transparent);
            transition: width 0.3s;
            pointer-events: none;
        }
        .endpoint-row:hover::after {
            width: 100%;
        }
        .endpoint-row.visible {
            opacity: 1;
            transform: translateX(0);
        }
        .typewriter {
            overflow: hidden;
            border-right: 2px solid #6bff8f;
            white-space: nowrap;
            animation: typing 2s steps(20, end), blink 0.75s step-end infinite;
            max-width: fit-content;
        }
        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }
        @keyframes blink {
            50% { border-color: transparent; }
        }
        .number-counter {
            font-variant-numeric: tabular-nums;
            background: linear-gradient(135deg, var(--counter-color, #6bff8f) 0%, rgba(255,255,255,0.8) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-card:nth-child(1) .number-counter { --counter-color: #6bff8f; }
        .stat-card:nth-child(2) .number-counter { --counter-color: #53ddfc; }
        .stat-card:nth-child(3) .number-counter { --counter-color: #7de9ff; }
        .stat-card:nth-child(4) .number-counter { --counter-color: #6bff8f; }
        .floating {
            animation: floating 6s ease-in-out infinite;
        }
        @keyframes floating {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        .live-pulse {
            animation: livePulse 2s ease-in-out infinite;
        }
        @keyframes livePulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(107, 255, 143, 0.4); }
            50% { box-shadow: 0 0 0 8px rgba(107, 255, 143, 0); }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0e0e13; }
        ::-webkit-scrollbar-thumb { background: #25252d; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #6bff8f; }
    </style>
</head>
<body class="bg-background text-on-surface font-body selection:bg-primary/30 min-h-screen">
<!-- Simple Top Header -->
<header class="flex justify-between items-center px-8 h-16 bg-background/80 backdrop-blur-xl border-b border-white/5 sticky top-0 z-50">
<div class="flex items-center gap-3">
<div class="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
<span class="material-symbols-outlined text-primary text-lg">calendar_month</span>
</div>
<h1 class="text-lg font-bold text-on-surface tracking-tight">CalendarSchedulingEnv</h1>
</div>
<div class="flex items-center gap-4">
<div class="flex items-center gap-2 px-3 py-1.5 bg-primary/10 border border-primary/20 rounded-full">
<span class="w-2 h-2 rounded-full bg-primary live-pulse"></span>
<span class="text-[10px] font-medium uppercase tracking-widest text-primary">Live</span>
</div>
</div>
</header>

<!-- Main Content -->
<main class="max-w-6xl mx-auto px-6 py-10 space-y-8">
<!-- Hero Section -->
<section class="relative p-8 rounded-2xl overflow-hidden glass-panel">
<div class="absolute top-0 right-0 w-96 h-full opacity-30 wireframe-grid"></div>
<div class="relative z-10 space-y-4">
<h2 class="text-3xl md:text-4xl font-bold tracking-tight text-on-surface mb-2">
                    AI Scheduling <span class="text-primary glow-text inline-block">Environment</span>
</h2>
<p class="max-w-xl text-on-surface-variant text-sm leading-relaxed">
                    A high-fidelity Gymnasium environment for multi-agent temporal resource allocation. 
                    Train RL agents to optimize meeting schedules across complex constraints.
                </p>
<div class="flex gap-3 pt-2">
<a href="/docs" class="bg-primary text-on-primary font-semibold text-xs uppercase tracking-wider px-5 py-2.5 rounded-lg hover:scale-[1.02] hover:shadow-[0_0_20px_rgba(107,255,143,0.3)] transition-all">
                        Get Started
                    </a>
<a href="/tasks" class="border border-white/10 text-on-surface font-semibold text-xs uppercase tracking-wider px-5 py-2.5 rounded-lg hover:bg-white/5 hover:border-primary/30 transition-all">
                        View Tasks
                    </a>
</div>
</div>
</section>

<!-- Stats Row -->
<section class="grid grid-cols-2 md:grid-cols-4 gap-4" id="stats-section">
<div class="glass-panel p-5 rounded-xl stat-card">
<p class="text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Tasks</p>
<div class="text-2xl font-bold text-primary number-counter" data-target="3">0</div>
<p class="text-[10px] text-on-surface-variant mt-1">Difficulty Levels</p>
</div>
<div class="glass-panel p-5 rounded-xl stat-card">
<p class="text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Events</p>
<div class="text-2xl font-bold text-secondary number-counter" data-target="25">0</div>
<p class="text-[10px] text-on-surface-variant mt-1">Total Capacity</p>
</div>
<div class="glass-panel p-5 rounded-xl stat-card">
<p class="text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Rooms</p>
<div class="text-2xl font-bold text-tertiary number-counter" data-target="9">0</div>
<p class="text-[10px] text-on-surface-variant mt-1">Meeting Spaces</p>
</div>
<div class="glass-panel p-5 rounded-xl stat-card">
<p class="text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Tests</p>
<div class="text-2xl font-bold text-primary number-counter" data-target="37">0</div>
<p class="text-[10px] text-on-surface-variant mt-1">Passing</p>
</div>
</section>

<!-- Task Cards -->
<section class="space-y-4">
<h3 class="text-xs uppercase tracking-widest text-on-surface-variant font-medium">Available Tasks</h3>
<div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="tasks-section">
<!-- Easy -->
<div class="glass-panel rounded-xl overflow-hidden hover:border-primary/30 transition-all group cursor-pointer task-card" style="transition-delay: 0.1s;">
<div class="h-32 bg-surface-container-lowest relative wireframe-grid flex items-center justify-center floating">
<span class="material-symbols-outlined text-primary text-5xl opacity-40 group-hover:opacity-80 transition-all group-hover:scale-110">event_available</span>
<div class="absolute top-3 left-3 px-2 py-1 bg-primary/20 rounded text-[9px] font-medium text-primary uppercase">Easy</div>
</div>
<div class="p-5 space-y-3">
<h4 class="font-semibold text-on-surface group-hover:text-primary transition-colors">Simple Scheduling</h4>
<p class="text-xs text-on-surface-variant">5 events, 1 room. Basic temporal constraints.</p>
<div class="flex items-center justify-between pt-2">
<span class="text-[10px] text-primary font-medium">50 Steps</span>
<span class="material-symbols-outlined text-on-surface-variant text-sm group-hover:text-primary group-hover:translate-x-1 transition-all">arrow_forward</span>
</div>
</div>
</div>
<!-- Medium -->
<div class="glass-panel rounded-xl overflow-hidden hover:border-secondary/30 transition-all group cursor-pointer task-card" style="transition-delay: 0.2s;">
<div class="h-32 bg-surface-container-lowest relative wireframe-grid flex items-center justify-center floating" style="animation-delay: 0.5s;">
<span class="material-symbols-outlined text-secondary text-5xl opacity-40 group-hover:opacity-80 transition-all group-hover:scale-110">event_note</span>
<div class="absolute top-3 left-3 px-2 py-1 bg-secondary/20 rounded text-[9px] font-medium text-secondary uppercase">Medium</div>
</div>
<div class="p-5 space-y-3">
<h4 class="font-semibold text-on-surface group-hover:text-secondary transition-colors">Constrained Scheduling</h4>
<p class="text-xs text-on-surface-variant">8 events, 3 rooms. Lunch breaks & priorities.</p>
<div class="flex items-center justify-between pt-2">
<span class="text-[10px] text-secondary font-medium">80 Steps</span>
<span class="material-symbols-outlined text-on-surface-variant text-sm group-hover:text-secondary group-hover:translate-x-1 transition-all">arrow_forward</span>
</div>
</div>
</div>
<!-- Hard -->
<div class="glass-panel rounded-xl overflow-hidden hover:border-error/30 transition-all group cursor-pointer task-card" style="transition-delay: 0.3s;">
<div class="h-32 bg-surface-container-lowest relative wireframe-grid flex items-center justify-center floating" style="animation-delay: 1s;">
<span class="material-symbols-outlined text-error text-5xl opacity-40 group-hover:opacity-80 transition-all group-hover:scale-110">event_busy</span>
<div class="absolute top-3 left-3 px-2 py-1 bg-error/20 rounded text-[9px] font-medium text-error uppercase">Hard</div>
</div>
<div class="p-5 space-y-3">
<h4 class="font-semibold text-on-surface group-hover:text-error transition-colors">Complex Scheduling</h4>
<p class="text-xs text-on-surface-variant">12 events, 5 rooms. Travel time & conflicts.</p>
<div class="flex items-center justify-between pt-2">
<span class="text-[10px] text-error font-medium">120 Steps</span>
<span class="material-symbols-outlined text-on-surface-variant text-sm group-hover:text-error group-hover:translate-x-1 transition-all">arrow_forward</span>
</div>
</div>
</div>
</div>
</section>

<!-- API Endpoints -->
<section class="glass-panel rounded-xl p-6" id="endpoints-section">
<h3 class="text-xs uppercase tracking-widest text-on-surface-variant font-medium mb-4">API Endpoints</h3>
<div class="space-y-2">
<a href="/health" class="flex items-center gap-4 p-3 rounded-lg hover:bg-white/5 transition-all group endpoint-row" style="transition-delay: 0.05s;">
<span class="px-2 py-1 bg-primary/20 text-primary text-[10px] font-mono rounded group-hover:bg-primary/30 transition-colors">GET</span>
<code class="text-sm font-mono text-on-surface">/health</code>
<span class="text-xs text-on-surface-variant flex-1 text-right group-hover:text-on-surface group-hover:translate-x-1 transition-all">Check status</span>
</a>
<a href="/tasks" class="flex items-center gap-4 p-3 rounded-lg hover:bg-white/5 transition-all group endpoint-row" style="transition-delay: 0.1s;">
<span class="px-2 py-1 bg-primary/20 text-primary text-[10px] font-mono rounded group-hover:bg-primary/30 transition-colors">GET</span>
<code class="text-sm font-mono text-on-surface">/tasks</code>
<span class="text-xs text-on-surface-variant flex-1 text-right group-hover:text-on-surface group-hover:translate-x-1 transition-all">List tasks</span>
</a>
<div class="flex items-center gap-4 p-3 rounded-lg opacity-60 endpoint-row" style="transition-delay: 0.15s;">
<span class="px-2 py-1 bg-error/20 text-error text-[10px] font-mono rounded">POST</span>
<code class="text-sm font-mono text-on-surface">/reset</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">Reset env</span>
</div>
<div class="flex items-center gap-4 p-3 rounded-lg opacity-60 endpoint-row" style="transition-delay: 0.2s;">
<span class="px-2 py-1 bg-secondary/20 text-secondary text-[10px] font-mono rounded">POST</span>
<code class="text-sm font-mono text-on-surface">/step</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">Execute action</span>
</div>
<a href="/state?session_id=default" class="flex items-center gap-4 p-3 rounded-lg hover:bg-white/5 transition-all group endpoint-row" style="transition-delay: 0.25s;">
<span class="px-2 py-1 bg-primary/20 text-primary text-[10px] font-mono rounded group-hover:bg-primary/30 transition-colors">GET</span>
<code class="text-sm font-mono text-on-surface">/state</code>
<span class="text-xs text-on-surface-variant flex-1 text-right group-hover:text-on-surface group-hover:translate-x-1 transition-all">Get state</span>
</a>
</div>
</section>
</main>

<!-- Simple Footer -->
<footer class="mt-16 py-8 border-t border-white/5 text-center">
<p class="text-[10px] text-on-surface-variant/50 uppercase tracking-widest">
                Meta PyTorch Hackathon • OpenEnv Compatible
            </p>
</footer>

<script>
        // Number counter animation
        function animateCounters() {
            const counters = document.querySelectorAll('.number-counter');
            counters.forEach(counter => {
                const target = parseInt(counter.getAttribute('data-target'));
                const duration = 1500;
                const step = target / (duration / 16);
                let current = 0;
                const updateCounter = () => {
                    current += step;
                    if (current < target) {
                        counter.textContent = Math.ceil(current);
                        requestAnimationFrame(updateCounter);
                    } else {
                        counter.textContent = target;
                    }
                };
                updateCounter();
            });
        }

        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    if (entry.target.id === 'stats-section') {
                        animateCounters();
                    }
                    entry.target.querySelectorAll('.task-card, .endpoint-row').forEach((el, i) => {
                        setTimeout(() => {
                            el.classList.add('visible');
                        }, i * 100);
                    });
                }
            });
        }, observerOptions);

        // Observe sections
        document.querySelectorAll('#stats-section, #tasks-section, #endpoints-section').forEach(section => {
            observer.observe(section);
        });

        // Add visible class to task cards on load
        setTimeout(() => {
            document.querySelectorAll('.task-card').forEach(card => {
                card.classList.add('visible');
            });
        }, 500);
    </script>
</body></html>'''

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
    normalized = reward / max_score
    # Clamp to strictly between 0 and 1 (not 0.0 or 1.0) for hackathon compliance
    return round(max(0.001, min(0.999, normalized)), 4)

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
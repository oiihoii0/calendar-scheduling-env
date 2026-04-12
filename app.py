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
<title>CalendarSchedulingEnv | Technical Dashboard</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&amp;family=Space+Grotesk:wght@300;400;500;600;700&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<script id="tailwind-config">
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    "colors": {
                        "tertiary-fixed-dim": "#00d1ec",
                        "secondary": "#53ddfc",
                        "on-secondary-fixed": "#003a45",
                        "on-primary-container": "#002c0f",
                        "inverse-on-surface": "#55545b",
                        "secondary-container": "#00687a",
                        "background": "#0e0e13",
                        "on-tertiary": "#005561",
                        "error-dim": "#d53d18",
                        "on-surface-variant": "#acaab1",
                        "surface-variant": "#25252d",
                        "on-secondary": "#004b58",
                        "outline": "#76747b",
                        "on-background": "#f9f5fd",
                        "on-primary-fixed": "#004a1d",
                        "outline-variant": "#48474d",
                        "surface-bright": "#2c2b33",
                        "secondary-fixed": "#65e1ff",
                        "on-error": "#450900",
                        "secondary-dim": "#40ceed",
                        "surface-container-lowest": "#000000",
                        "primary": "#6bff8f",
                        "primary-container": "#0abc56",
                        "tertiary": "#7de9ff",
                        "inverse-primary": "#006e2f",
                        "error": "#ff7351",
                        "surface-container": "#19191f",
                        "on-secondary-fixed-variant": "#005969",
                        "on-tertiary-fixed-variant": "#005561",
                        "surface-container-highest": "#25252d",
                        "on-tertiary-container": "#004b56",
                        "tertiary-dim": "#00d1ec",
                        "error-container": "#b92902",
                        "surface": "#0e0e13",
                        "on-surface": "#f9f5fd",
                        "primary-fixed": "#6bff8f",
                        "secondary-fixed-dim": "#48d4f3",
                        "on-error-container": "#ffd2c8",
                        "on-tertiary-fixed": "#00363e",
                        "on-primary-fixed-variant": "#006a2d",
                        "surface-tint": "#6bff8f",
                        "surface-container-high": "#1f1f26",
                        "tertiary-fixed": "#00e0fd",
                        "surface-dim": "#0e0e13",
                        "primary-dim": "#5bf083",
                        "primary-fixed-dim": "#5bf083",
                        "inverse-surface": "#fcf8ff",
                        "surface-container-low": "#131319",
                        "on-primary": "#005f28",
                        "on-secondary-container": "#ecfaff",
                        "tertiary-container": "#00e0fd"
                    },
                    "borderRadius": {
                        "DEFAULT": "0.125rem",
                        "lg": "0.25rem",
                        "xl": "0.5rem",
                        "full": "0.75rem"
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
        .glass-panel {
            background: rgba(25, 25, 31, 0.7);
            backdrop-filter: blur(12px);
        }
        .neon-glow {
            box-shadow: 0 0 12px rgba(107, 255, 143, 0.3);
        }
        .wireframe-grid {
            background-image: linear-gradient(rgba(107, 255, 143, 0.05) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(107, 255, 143, 0.05) 1px, transparent 1px);
            background-size: 20px 20px;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0e0e13; }
        ::-webkit-scrollbar-thumb { background: #25252d; border-radius: 10px; }
    </style>
</head>
<body class="bg-background text-on-surface font-body selection:bg-primary/30">
<!-- Main Wrapper: Two Column Layout -->
<div class="flex min-h-screen">
<!-- SideNavBar: Authority Source JSON -->
<aside class="hidden md:flex flex-col h-screen w-64 bg-[#131319] docked left-0 fixed z-50 py-8 space-y-2">
<div class="px-6 mb-10 flex items-center gap-3">
<div class="w-8 h-8 rounded bg-primary/20 flex items-center justify-center">
<span class="material-symbols-outlined text-primary text-xl" style="font-variation-settings: 'FILL' 1;">architecture</span>
</div>
<div>
<h2 class="font-label uppercase text-xs tracking-widest text-[#6bff8f]">Architect Admin</h2>
<p class="text-[10px] text-[#f9f5fd]/40 font-mono">System Active</p>
</div>
</div>
<nav class="flex-1 px-3 space-y-1">
<div class="flex items-center gap-4 px-4 py-3 bg-gradient-to-r from-[#6bff8f]/10 to-transparent text-[#6bff8f] border-l-4 border-[#6bff8f] cursor-pointer transition-colors duration-200">
<span class="material-symbols-outlined text-xl">dashboard</span>
<span class="font-label uppercase text-xs tracking-widest">Overview</span>
</div>
<div class="flex items-center gap-4 px-4 py-3 text-[#f9f5fd]/40 hover:text-[#f9f5fd]/80 hover:bg-[#19191f] transition-all cursor-pointer">
<span class="material-symbols-outlined text-xl">domain</span>
<span class="font-label uppercase text-xs tracking-widest">My Apartments</span>
</div>
<div class="flex items-center gap-4 px-4 py-3 text-[#f9f5fd]/40 hover:text-[#f9f5fd]/80 hover:bg-[#19191f] transition-all cursor-pointer">
<span class="material-symbols-outlined text-xl">analytics</span>
<span class="font-label uppercase text-xs tracking-widest">Reporting</span>
</div>
<div class="flex items-center gap-4 px-4 py-3 text-[#f9f5fd]/40 hover:text-[#f9f5fd]/80 hover:bg-[#19191f] transition-all cursor-pointer">
<span class="material-symbols-outlined text-xl">settings</span>
<span class="font-label uppercase text-xs tracking-widest">Settings</span>
</div>
</nav>
<div class="px-6 pt-6 border-t border-outline-variant/10">
<div class="flex items-center gap-3">
<img alt="User Profile" class="w-8 h-8 rounded-full border border-primary/20" data-alt="close-up profile avatar icon with neon green border and minimalist digital human silhouette" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAneZvtAL8KwaeZpakgnxbKqkjWepUw0p8xIuoZqGOHh0DUeaDCP7HeUcnSUePuu5P6lQJtMiIYrSJxL15_ndHj2E3bCw_ldUmOjN0fzQ2hHjoGPadrkOKxbwzx3-XCM_xKFaH6OcBpd65c_9ZHXN0RsX4WQBcW7eUOOPdp8hH35BVXkuAKcwprV5PbUisDmAdn4yBJ4WvVTimm22fWVnrqcI2XHU3fyNQTI5jz8qD69nedzqi0mQICNxY3FC2r1LhIM0sYKxQF77Jk"/>
<div class="overflow-hidden">
<p class="text-[10px] font-bold text-on-surface truncate">DEV_OPERATOR_01</p>
<p class="text-[9px] text-primary/60">ROOT ACCESS</p>
</div>
</div>
</div>
</aside>
<!-- Main Content Canvas -->
<main class="flex-1 md:ml-64 bg-background min-h-screen pb-24">
<!-- TopAppBar: Content & Identity JSON -->
<header class="flex justify-between items-center w-full px-8 h-16 bg-[#0e0e13]/80 backdrop-blur-xl docked full-width top-0 z-40 sticky">
<div class="flex items-center gap-8">
<h1 class="text-lg font-['Space_Grotesk'] font-bold uppercase tracking-widest text-[#6bff8f]">CalendarSchedulingEnv</h1>
<nav class="hidden lg:flex items-center gap-6 font-['Inter'] font-medium text-sm tracking-tight">
<a class="text-[#f9f5fd]/60 hover:text-[#f9f5fd] transition-colors" href="#">Docs</a>
<a class="text-[#6bff8f] border-b-2 border-[#6bff8f] pb-1" href="#">Tasks</a>
<a class="text-[#f9f5fd]/60 hover:text-[#f9f5fd] transition-colors" href="#">Health</a>
</nav>
</div>
<div class="flex items-center gap-4">
<div class="flex items-center gap-2 px-3 py-1 bg-primary/10 border border-primary/20 rounded-full">
<span class="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#6bff8f]"></span>
<span class="text-[10px] font-label uppercase tracking-widest text-primary">Live on Spaces</span>
</div>
</div>
</header>
<div class="max-w-7xl mx-auto px-8 pt-8 space-y-8">
<!-- 1. Hero Header -->
<section class="relative p-10 rounded-xl overflow-hidden border border-outline-variant/10 bg-surface-container-low">
<div class="absolute top-0 right-0 w-1/3 h-full opacity-20 pointer-events-none wireframe-grid" style="transform: perspective(1000px) rotateY(-30deg) scale(1.5);"></div>
<div class="relative z-10 space-y-6">
<div class="space-y-1">
<span class="text-primary font-label text-[10px] tracking-[0.3em] uppercase">Environment Dashboard</span>
<h2 class="text-4xl font-headline font-extrabold tracking-tight text-on-surface">Calendar Scheduling <span class="text-primary">Environment</span></h2>
</div>
<p class="max-w-2xl text-on-surface-variant leading-relaxed">A high-fidelity Markov Decision Process simulator for multi-room, multi-agent temporal resource allocation. Optimize scheduling efficiency across complex architectural constraints.</p>
<div class="flex gap-4">
<button class="bg-gradient-to-br from-primary to-primary-container text-on-primary font-label font-bold text-xs uppercase tracking-widest px-6 py-3 rounded hover:scale-[1.02] active:scale-95 transition-all">Try API Interface</button>
<button class="border border-outline-variant/30 text-on-surface font-label font-bold text-xs uppercase tracking-widest px-6 py-3 rounded hover:bg-white/5 transition-all">View Task Library</button>
</div>
</div>
</section>
<!-- 2. Global Stats -->
<section class="grid grid-cols-1 md:grid-cols-4 gap-4">
<div class="bg-surface-container-low p-6 rounded-lg border-l-2 border-primary/40">
<p class="text-on-surface-variant font-label text-[10px] uppercase tracking-wider mb-1">Difficulty Levels</p>
<div class="flex items-baseline gap-2">
<span class="text-3xl font-mono text-primary font-bold">03</span>
<span class="text-[10px] text-on-surface-variant font-mono">TIERS</span>
</div>
</div>
<div class="bg-surface-container-low p-6 rounded-lg border-l-2 border-secondary/40">
<p class="text-on-surface-variant font-label text-[10px] uppercase tracking-wider mb-1">Max Events</p>
<div class="flex items-baseline gap-2">
<span class="text-3xl font-mono text-secondary font-bold">12</span>
<span class="text-[10px] text-on-surface-variant font-mono">CAPACITY</span>
</div>
</div>
<div class="bg-surface-container-low p-6 rounded-lg border-l-2 border-primary/40">
<p class="text-on-surface-variant font-label text-[10px] uppercase tracking-wider mb-1">Active Rooms</p>
<div class="flex items-baseline gap-2">
<span class="text-3xl font-mono text-primary font-bold">05</span>
<span class="text-[10px] text-on-surface-variant font-mono">AVAILABLE</span>
</div>
</div>
<div class="bg-surface-container-low p-6 rounded-lg border-l-2 border-tertiary-fixed-dim/40">
<p class="text-on-surface-variant font-label text-[10px] uppercase tracking-wider mb-1">System Score</p>
<div class="flex items-baseline gap-2">
<span class="text-3xl font-mono text-tertiary-fixed-dim font-bold">0.84</span>
<span class="text-[10px] text-on-surface-variant font-mono">DELTA</span>
</div>
</div>
</section>
<!-- 3. Task Grid -->
<section class="space-y-4">
<div class="flex justify-between items-end">
<h3 class="font-label text-xs uppercase tracking-widest text-on-surface-variant">Active Task Library</h3>
<span class="text-[10px] font-mono text-primary/60">MODULAR_GRIDS_LOADED</span>
</div>
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
<!-- Easy -->
<div class="glass-panel rounded-xl border border-outline-variant/10 group hover:border-primary/30 transition-all overflow-hidden flex flex-col">
<div class="h-40 bg-surface-container-lowest relative wireframe-grid">
<div class="absolute inset-0 flex items-center justify-center opacity-40">
<span class="material-symbols-outlined text-primary text-6xl" style="font-variation-settings: 'wght' 100;">view_in_ar</span>
</div>
<div class="absolute bottom-2 left-4 px-2 py-1 bg-primary/20 rounded text-[9px] font-label text-primary uppercase">Tier 01</div>
</div>
<div class="p-6 space-y-4 flex-1">
<div>
<h4 class="text-on-surface font-bold text-lg">Simple Scheduling</h4>
<p class="text-xs text-on-surface-variant mt-1">Single-room allocation with no overlap constraints.</p>
</div>
<div class="flex justify-between items-center pt-4 mt-auto border-t border-outline-variant/10">
<span class="text-[10px] font-label uppercase text-primary">Status: Stable</span>
<span class="material-symbols-outlined text-on-surface-variant text-sm">arrow_forward</span>
</div>
</div>
</div>
<!-- Medium -->
<div class="glass-panel rounded-xl border border-outline-variant/10 group hover:border-secondary/30 transition-all overflow-hidden flex flex-col">
<div class="h-40 bg-surface-container-lowest relative wireframe-grid">
<div class="absolute inset-0 flex items-center justify-center opacity-40">
<span class="material-symbols-outlined text-secondary text-6xl" style="font-variation-settings: 'wght' 100;">dashboard_customize</span>
</div>
<div class="absolute bottom-2 left-4 px-2 py-1 bg-secondary/20 rounded text-[9px] font-label text-secondary uppercase">Tier 02</div>
</div>
<div class="p-6 space-y-4 flex-1">
<div>
<h4 class="text-on-surface font-bold text-lg">Constrained Scheduling</h4>
<p class="text-xs text-on-surface-variant mt-1">Multi-room dependency graphs and priority tiers.</p>
</div>
<div class="flex justify-between items-center pt-4 mt-auto border-t border-outline-variant/10">
<span class="text-[10px] font-label uppercase text-secondary">Status: Active</span>
<span class="material-symbols-outlined text-on-surface-variant text-sm">arrow_forward</span>
</div>
</div>
</div>
<!-- Hard -->
<div class="glass-panel rounded-xl border border-outline-variant/10 group hover:border-error/30 transition-all overflow-hidden flex flex-col">
<div class="h-40 bg-surface-container-lowest relative wireframe-grid">
<div class="absolute inset-0 flex items-center justify-center opacity-40">
<span class="material-symbols-outlined text-error text-6xl" style="font-variation-settings: 'wght' 100;">filter_center_focus</span>
</div>
<div class="absolute bottom-2 left-4 px-2 py-1 bg-error/20 rounded text-[9px] font-label text-error uppercase">Tier 03</div>
</div>
<div class="p-6 space-y-4 flex-1">
<div>
<h4 class="text-on-surface font-bold text-lg">Complex Scheduling</h4>
<p class="text-xs text-on-surface-variant mt-1">Dynamic event resizing and conflict resolution.</p>
</div>
<div class="flex justify-between items-center pt-4 mt-auto border-t border-outline-variant/10">
<span class="text-[10px] font-label uppercase text-error">Status: Critical</span>
<span class="material-symbols-outlined text-on-surface-variant text-sm">arrow_forward</span>
</div>
</div>
</div>
</div>
</section>
<!-- 4. API Control Center & Reward Logic Grid -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
<!-- API Control Center -->
<section class="space-y-4">
<h3 class="font-label text-xs uppercase tracking-widest text-on-surface-variant">API Control Center</h3>
<div class="bg-surface-container-low rounded-xl border border-outline-variant/10 overflow-hidden">
<div class="divide-y divide-outline-variant/10">
<div class="p-4 flex items-center gap-4 hover:bg-surface-container-highest/30 transition-all">
<span class="px-2 py-1 bg-primary/20 text-primary text-[9px] font-mono rounded">GET</span>
<code class="text-sm font-mono text-on-surface">/health</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">System status &amp; latency</span>
</div>
<div class="p-4 flex items-center gap-4 hover:bg-surface-container-highest/30 transition-all">
<span class="px-2 py-1 bg-primary/20 text-primary text-[9px] font-mono rounded">GET</span>
<code class="text-sm font-mono text-on-surface">/tasks</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">List available MDP tasks</span>
</div>
<div class="p-4 flex items-center gap-4 hover:bg-surface-container-highest/30 transition-all">
<span class="px-2 py-1 bg-error/20 text-error text-[9px] font-mono rounded">POST</span>
<code class="text-sm font-mono text-on-surface">/reset</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">Flush current environment state</span>
</div>
<div class="p-4 flex items-center gap-4 hover:bg-surface-container-highest/30 transition-all">
<span class="px-2 py-1 bg-secondary/20 text-secondary text-[9px] font-mono rounded">POST</span>
<code class="text-sm font-mono text-on-surface">/step</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">Execute agent action sequence</span>
</div>
<div class="p-4 flex items-center gap-4 hover:bg-surface-container-highest/30 transition-all">
<span class="px-2 py-1 bg-primary/20 text-primary text-[9px] font-mono rounded">GET</span>
<code class="text-sm font-mono text-on-surface">/state</code>
<span class="text-xs text-on-surface-variant flex-1 text-right">Current tensor observation</span>
</div>
</div>
</div>
</section>
<!-- Reward Logic -->
<section class="space-y-4">
<h3 class="font-label text-xs uppercase tracking-widest text-on-surface-variant">Reward Objective Logic</h3>
<div class="grid grid-cols-2 gap-4">
<div class="p-5 bg-surface-container-low rounded-xl border border-outline-variant/10 space-y-2">
<div class="flex justify-between items-center">
<span class="text-xs font-bold text-on-surface">Valid Action</span>
<span class="text-primary font-mono text-xs">+1.0</span>
</div>
<p class="text-[10px] text-on-surface-variant leading-relaxed">Assigned when an agent successfully schedules an event with no temporal conflicts.</p>
</div>
<div class="p-5 bg-surface-container-low rounded-xl border border-outline-variant/10 space-y-2">
<div class="flex justify-between items-center">
<span class="text-xs font-bold text-on-surface">Invalid Action</span>
<span class="text-error font-mono text-xs">-0.5</span>
</div>
<p class="text-[10px] text-on-surface-variant leading-relaxed">Penalty for attempting to place an event in an occupied slot or invalid room.</p>
</div>
<div class="p-5 bg-surface-container-low rounded-xl border border-outline-variant/10 space-y-2">
<div class="flex justify-between items-center">
<span class="text-xs font-bold text-on-surface">Completion</span>
<span class="text-primary font-mono text-xs">+5.0</span>
</div>
<p class="text-[10px] text-on-surface-variant leading-relaxed">Bonus for successfully scheduling all events within the max step limit.</p>
</div>
<div class="p-5 bg-surface-container-low rounded-xl border border-outline-variant/10 space-y-2">
<div class="flex justify-between items-center">
<span class="text-xs font-bold text-on-surface">Step Decay</span>
<span class="text-secondary font-mono text-xs">-0.01</span>
</div>
<p class="text-[10px] text-on-surface-variant leading-relaxed">Negative pressure per environment step to encourage rapid convergence.</p>
</div>
</div>
</section>
</div>
</div>
</main>
<!-- Footer: JSON Authority -->
<footer class="fixed bottom-0 left-0 md:left-64 right-0 flex justify-between items-center px-12 py-6 bg-[#0e0e13] border-t border-[#48474d]/15 z-50">
<p class="font-['Inter'] text-[10px] text-[#f9f5fd]/30 uppercase tracking-widest">© 2024 Obsidian Architect Hackathon</p>
<div class="flex gap-8">
<a class="font-['Inter'] text-[10px] text-[#f9f5fd]/30 hover:text-[#6bff8f] hover:underline transition-colors" href="#">Documentation</a>
<a class="font-['Inter'] text-[10px] text-[#f9f5fd]/30 hover:text-[#6bff8f] hover:underline transition-colors" href="#">API Reference</a>
<a class="font-['Inter'] text-[10px] text-[#f9f5fd]/30 hover:text-[#6bff8f] hover:underline transition-colors" href="#">Support</a>
</div>
</footer>
</div>
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
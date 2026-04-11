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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&amp;family=Space+Grotesk:wght@300;400;500;600;700&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<script id="tailwind-config">
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    "colors": {
                        "primary": "#6dddff",
                        "secondary": "#ac8aff",
                        "tertiary": "#ff97b5",
                        "background": "#0a0a0f",
                        "surface": "#0e0e13",
                        "on-surface": "#f9f5fd",
                        "on-surface-variant": "#acaab1",
                    },
                    "borderRadius": {
                        "DEFAULT": "0.125rem",
                        "lg": "0.25rem",
                        "xl": "0.5rem",
                    },
                    "fontFamily": {
                        "headline": ["Space Grotesk"],
                        "body": ["Inter"],
                        "mono": ["JetBrains Mono"],
                    }
                },
            },
        }
    </script>
<style>
        body {
            background-color: #0a0a0f;
            color: #f9f5fd;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 50% 50%, rgba(109, 221, 255, 0.02) 0%, transparent 50%),
                url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3Base-filter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/feTurbulence%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            background-blend-mode: overlay;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(24px);
            border: 0.5px solid rgba(255, 255, 255, 0.1);
            box-shadow: 
                0 4px 24px -1px rgba(0, 0, 0, 0.5),
                inset 0 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        .pro-shadow {
            box-shadow: 0 0 20px rgba(109, 221, 255, 0.05), 0 0 40px rgba(172, 138, 255, 0.02);
        }
        .headline-font { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.05em; }
        .mono-font { font-family: 'JetBrains Mono', monospace; }
        
        .hud-dot {
            width: 2px;
            height: 2px;
            background: rgba(109, 221, 255, 0.5);
        }
        
        @keyframes scanline {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }
        .scanline {
            animation: scanline 8s linear infinite;
        }
        
        .wireframe-container {
            perspective: 2000px;
        }
        .wireframe-grid {
            transform: rotateX(60deg) rotateZ(-45deg);
            transform-style: preserve-3d;
        }
        .wireframe-layer-2 {
            transform: translateZ(20px);
        }
        .wireframe-layer-3 {
            transform: translateZ(40px);
        }
        
        .custom-scrollbar::-webkit-scrollbar { width: 3px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(109, 221, 255, 0.1); }
    </style>
</head>
<body class="bg-background selection:bg-primary/30">
<!-- HUD Overlays -->
<div class="fixed inset-0 pointer-events-none z-[60] border-[16px] border-transparent">
<div class="absolute top-4 left-4 flex gap-2">
<div class="hud-dot"></div><div class="hud-dot"></div>
</div>
<div class="absolute bottom-4 right-4 flex gap-2">
<div class="hud-dot"></div><div class="hud-dot"></div>
</div>
<div class="absolute top-1/2 left-0 -translate-y-1/2 flex flex-col gap-1 px-1">
<span class="text-[8px] mono-font text-primary/20 -rotate-90">LAT: 37.7749</span>
<span class="text-[8px] mono-font text-primary/20 -rotate-90 mt-12">LON: -122.4194</span>
</div>
</div>
<!-- TopNavBar -->
<nav class="fixed top-0 w-full z-50 flex justify-between items-center px-6 h-14 bg-background/80 backdrop-blur-md border-b border-white/5 pro-shadow">
<div class="flex items-center gap-10">
<div class="flex items-center gap-3">
<div class="w-8 h-8 flex items-center justify-center border border-primary/30 bg-primary/5">
<span class="material-symbols-outlined text-primary text-lg" data-icon="terminal">terminal</span>
</div>
<span class="text-sm font-bold tracking-[0.3em] text-white headline-font uppercase">Kinetic_Intel</span>
</div>
<div class="hidden md:flex gap-10 mono-font text-[10px] uppercase tracking-widest">
<a class="text-primary border-b border-primary py-4" href="#">[ 01 ] Schedules</a>
<a class="text-on-surface-variant hover:text-white transition-colors py-4" href="#">[ 02 ] Neural_Net</a>
<a class="text-on-surface-variant hover:text-white transition-colors py-4" href="#">[ 03 ] Log_Stream</a>
</div>
</div>
<div class="flex items-center gap-8">
<div class="flex items-center gap-3 px-3 py-1 bg-emerald-500/5 border border-emerald-500/20">
<div class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></div>
<span class="text-[9px] mono-font tracking-widest uppercase text-emerald-400">SYS_ONLINE</span>
</div>
<div class="hidden lg:block text-right mono-font">
<div class="text-[9px] text-on-surface-variant uppercase tracking-tighter opacity-60">Session_Time</div>
<div class="text-xs text-primary font-medium">22:45:09.112 UTC</div>
</div>
<div class="flex items-center gap-5 text-on-surface-variant">
<span class="material-symbols-outlined text-lg cursor-pointer hover:text-primary transition-colors" data-icon="settings_input_antenna">settings_input_antenna</span>
<span class="material-symbols-outlined text-lg cursor-pointer hover:text-primary transition-colors" data-icon="database">database</span>
<img alt="Avatar" class="w-6 h-6 border border-white/10 grayscale hover:grayscale-0 transition-all cursor-pointer" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAy5EnRiUoEjc6oN_4p3S-7sqzw68DsX_xndJxnpXX1U1YxG5NqqCkbQRBhL1ZWhiTp50ZJ6UCFjPTbk4-gkwPnMUg8pNXEPL1iUuu5LcsmlLlFf7Zd_81-OfAod4eahrvD2cSpEU9NZ2kq8yW5eO_p0P9ZHg3uMtT6aUaZpUcIj-xzxKcewA99O0Po4TtMiDxg_IsblevkAuqsjCc-n8FdYJ6hYa4nOrKgnqHO2sZqeYrOUzj3INLqyKszQ7OLN_ykAFrvtXncGgTe"/>
</div>
</div>
</nav>
<!-- SideNavBar -->
<aside class="fixed left-0 top-14 h-[calc(100vh-3.5rem)] w-56 bg-surface/30 backdrop-blur-sm border-r border-white/5 flex flex-col z-40 hidden md:flex">
<div class="p-4 border-b border-white/5 bg-white/[0.02]">
<div class="flex flex-col gap-1">
<div class="flex justify-between items-center">
<span class="text-[9px] mono-font text-on-surface-variant tracking-widest">LOCAL_NODE</span>
<span class="text-[8px] mono-font text-primary">v4.0.2</span>
</div>
<div class="text-xs font-bold headline-font text-white uppercase tracking-wider">NODE_GAMMA_01</div>
</div>
</div>
<nav class="flex-1 py-4">
<div class="px-4 mb-2">
<span class="text-[8px] mono-font text-on-surface-variant/40 tracking-[0.2em] uppercase">Navigation_Matrix</span>
</div>
<a class="flex items-center gap-3 px-6 py-3 text-primary bg-primary/5 border-r border-primary mono-font text-[10px] uppercase tracking-widest group transition-all" href="#">
<span class="material-symbols-outlined text-sm group-hover:scale-110 transition-transform" data-icon="grid_view">grid_view</span> Command_Center
        </a>
<a class="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-white hover:bg-white/5 mono-font text-[10px] uppercase tracking-widest transition-all" href="#">
<span class="material-symbols-outlined text-sm" data-icon="analytics">analytics</span> Data_Analysis
        </a>
<a class="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-white hover:bg-white/5 mono-font text-[10px] uppercase tracking-widest transition-all" href="#">
<span class="material-symbols-outlined text-sm" data-icon="layers">layers</span> Env_Layers
        </a>
<a class="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-white hover:bg-white/5 mono-font text-[10px] uppercase tracking-widest transition-all" href="#">
<span class="material-symbols-outlined text-sm" data-icon="router">router</span> Node_Registry
        </a>
</nav>
<div class="mt-auto p-4 space-y-3">
<div class="bg-white/5 border border-white/5 p-3">
<div class="flex justify-between text-[8px] mono-font text-on-surface-variant mb-1">
<span>SYS_LOAD</span>
<span>42%</span>
</div>
<div class="h-1 bg-white/5 rounded-full overflow-hidden">
<div class="h-full bg-primary/40 w-[42%]"></div>
</div>
</div>
<button class="w-full py-2 bg-primary text-background headline-font font-bold text-[9px] uppercase tracking-[0.2em] hover:bg-white transition-all shadow-lg shadow-primary/20">
            INIT_OPTIMIZATION
        </button>
</div>
</aside>
<!-- Main Canvas -->
<main class="md:ml-56 pt-20 pb-10 px-6 min-h-screen relative">
<!-- Hero Section -->
<section class="mb-10 flex flex-col lg:flex-row items-center justify-between gap-10">
<div class="max-w-xl">
<div class="flex items-center gap-2 mb-4">
<div class="h-px w-8 bg-primary/40"></div>
<span class="text-[10px] mono-font text-primary tracking-[0.3em] uppercase">Project_OpenEnv</span>
</div>
<h1 class="headline-font font-bold text-4xl md:text-6xl mb-6 leading-none tracking-tight uppercase">
<span class="text-white">AI-DRIVEN</span><br/>
<span class="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">ORCHESTRATION</span>
</h1>
<p class="text-on-surface-variant font-light text-sm tracking-wide leading-relaxed border-l border-white/10 pl-6 max-w-lg">
                High-fidelity temporal scheduling engine optimized for multi-agent resource allocation within Gymnasium-compatible environments.
            </p>
</div>
<div class="relative w-64 h-64 md:w-80 md:h-80 wireframe-container flex items-center justify-center">
<div class="absolute inset-0 bg-primary/5 rounded-full blur-[80px]"></div>
<!-- Complex 3D Wireframe -->
<div class="wireframe-grid w-48 h-48 md:w-56 md:h-56 relative border border-white/10">
<!-- Layer 1 (Base) -->
<div class="absolute inset-0 grid grid-cols-4 grid-rows-4">
<div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div>
<div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div>
<div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div>
<div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div><div class="border-[0.5px] border-white/10"></div>
</div>
<!-- Layer 2 (Data Nodes) -->
<div class="wireframe-layer-2 absolute inset-0">
<div class="absolute top-1/4 left-1/4 w-2 h-2 bg-primary shadow-[0_0_10px_#6dddff]"></div>
<div class="absolute top-3/4 left-1/2 w-2 h-2 bg-secondary shadow-[0_0_10px_#ac8aff]"></div>
<div class="absolute top-1/2 left-3/4 w-1.5 h-1.5 bg-tertiary shadow-[0_0_10px_#ff97b5]"></div>
<div class="absolute inset-0 border-[0.5px] border-primary/20"></div>
</div>
<!-- Layer 3 (Ghost Frame) -->
<div class="wireframe-layer-3 absolute inset-0 border border-primary/10 bg-primary/5"></div>
</div>
<!-- Floating HUD text around wireframe -->
<div class="absolute top-0 right-0 mono-font text-[8px] text-primary/40 uppercase">X: 402.1<br/>Y: 011.8</div>
<div class="absolute bottom-0 left-0 mono-font text-[8px] text-secondary/40 uppercase">RENDER: TRUE<br/>V_SYNC: LOCKED</div>
</div>
</section>
<!-- Main Dashboard Grid -->
<div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
<!-- Col 1: System Params -->
<div class="lg:col-span-3 space-y-4">
<div class="glass-card p-4">
<div class="flex justify-between items-center mb-6">
<h3 class="headline-font text-[10px] font-bold uppercase tracking-[0.2em] text-white">Env_Stats</h3>
<div class="flex gap-1">
<div class="w-1 h-1 bg-primary"></div>
<div class="w-1 h-1 bg-white/20"></div>
</div>
</div>
<div class="space-y-4">
<div class="flex items-end gap-1 h-16 mb-6 px-1">
<div class="flex-1 bg-primary/10 h-[30%] border-t border-primary/20"></div>
<div class="flex-1 bg-primary/20 h-[60%] border-t border-primary/30"></div>
<div class="flex-1 bg-primary h-[100%] border-t border-primary shadow-[0_0_15px_rgba(109,221,255,0.3)]"></div>
<div class="flex-1 bg-primary/40 h-[45%] border-t border-primary/50"></div>
<div class="flex-1 bg-primary/20 h-[75%] border-t border-primary/30"></div>
</div>
<div class="space-y-2">
<div class="p-2 bg-white/[0.03] border border-white/5 flex justify-between items-center group cursor-pointer hover:bg-white/[0.05]">
<span class="text-[9px] mono-font text-on-surface-variant uppercase">Difficulty</span>
<span class="headline-font text-lg font-light text-primary tracking-tighter">LVL_03</span>
</div>
<div class="p-2 bg-white/[0.03] border border-white/5 flex justify-between items-center group cursor-pointer hover:bg-white/[0.05]">
<span class="text-[9px] mono-font text-on-surface-variant uppercase">Active_Evt</span>
<span class="headline-font text-lg font-light text-primary tracking-tighter">0025</span>
</div>
<div class="p-2 bg-white/[0.03] border border-white/5 flex justify-between items-center group cursor-pointer hover:bg-white/[0.05]">
<span class="text-[9px] mono-font text-on-surface-variant uppercase">Map_Sector</span>
<span class="headline-font text-lg font-light text-primary tracking-tighter">SEC_09</span>
</div>
</div>
</div>
</div>
<div class="glass-card p-4 bg-secondary/5 border-secondary/20">
<div class="flex items-center justify-between mb-2">
<span class="text-[8px] mono-font text-secondary tracking-widest uppercase">Net_Sync</span>
<span class="text-[8px] mono-font text-white">99.8%</span>
</div>
<div class="w-full h-0.5 bg-white/5">
<div class="h-full bg-secondary w-[99%] shadow-[0_0_8px_#ac8aff]"></div>
</div>
</div>
</div>
<!-- Col 2: Endpoint Control -->
<div class="lg:col-span-5">
<div class="glass-card p-5 h-full relative overflow-hidden">
<div class="absolute top-0 right-0 p-2 opacity-10">
<span class="material-symbols-outlined text-4xl" data-icon="terminal">terminal</span>
</div>
<h3 class="headline-font text-[10px] font-bold uppercase tracking-[0.2em] text-white mb-5 flex items-center gap-2">
<span class="w-1 h-3 bg-primary"></span>
                    API_CONTROL_MATRIX
                </h3>
<div class="space-y-2 custom-scrollbar overflow-y-auto max-h-[380px] pr-2">
<!-- API Rows -->
<div class="flex items-center justify-between p-3 bg-white/[0.02] border border-white/5 hover:border-primary/40 transition-all cursor-pointer group">
<div class="flex items-center gap-4">
<span class="text-[8px] mono-font px-1.5 py-0.5 border border-emerald-500/30 text-emerald-400 bg-emerald-500/5">GET</span>
<code class="text-xs mono-font text-on-surface group-hover:text-primary transition-colors">/system/health</code>
</div>
<div class="flex items-center gap-3">
<span class="text-[8px] mono-font text-on-surface-variant opacity-40">200_OK</span>
<div class="w-1 h-1 bg-emerald-400 shadow-[0_0_5px_#4ade80]"></div>
</div>
</div>
<div class="flex items-center justify-between p-3 bg-white/[0.02] border border-white/5 hover:border-secondary/40 transition-all cursor-pointer group">
<div class="flex items-center gap-4">
<span class="text-[8px] mono-font px-1.5 py-0.5 border border-secondary/30 text-secondary bg-secondary/5">POST</span>
<code class="text-xs mono-font text-on-surface group-hover:text-secondary transition-colors">/env/reset_all</code>
</div>
<div class="flex items-center gap-3">
<span class="text-[8px] mono-font text-on-surface-variant opacity-40">READY</span>
<div class="w-1 h-1 bg-secondary shadow-[0_0_5px_#ac8aff]"></div>
</div>
</div>
<div class="flex items-center justify-between p-3 bg-white/[0.02] border border-white/5 hover:border-tertiary/40 transition-all cursor-pointer group">
<div class="flex items-center gap-4">
<span class="text-[8px] mono-font px-1.5 py-0.5 border border-tertiary/30 text-tertiary bg-tertiary/5">POST</span>
<code class="text-xs mono-font text-on-surface group-hover:text-tertiary transition-colors">/agent/step_sync</code>
</div>
<div class="flex items-center gap-3">
<span class="text-[8px] mono-font text-on-surface-variant opacity-40">ACTIVE</span>
<div class="w-1 h-1 bg-tertiary animate-pulse shadow-[0_0_5px_#ff97b5]"></div>
</div>
</div>
<div class="flex items-center justify-between p-3 bg-white/[0.02] border border-white/5 hover:border-primary/40 transition-all cursor-pointer group">
<div class="flex items-center gap-4">
<span class="text-[8px] mono-font px-1.5 py-0.5 border border-primary/30 text-primary bg-primary/5">GET</span>
<code class="text-xs mono-font text-on-surface group-hover:text-primary transition-colors">/stream/state</code>
</div>
<div class="flex items-center gap-3">
<span class="text-[8px] mono-font text-on-surface-variant opacity-40">STABLE</span>
<div class="w-1 h-1 bg-primary"></div>
</div>
</div>
</div>
<div class="mt-4 pt-4 border-t border-white/5 flex justify-between items-center">
<span class="text-[8px] mono-font text-on-surface-variant uppercase tracking-widest">Global_Status</span>
<div class="flex gap-1">
<div class="w-6 h-1 bg-primary/40"></div>
<div class="w-6 h-1 bg-primary/10"></div>
<div class="w-6 h-1 bg-primary/10"></div>
</div>
</div>
</div>
</div>
<!-- Col 3: Agent Task Load -->
<div class="lg:col-span-4 flex flex-col gap-3">
<!-- Complex Progress Card 1 -->
<div class="glass-card p-4 relative group cursor-pointer overflow-hidden">
<div class="flex justify-between items-start mb-3">
<div>
<div class="text-[8px] mono-font text-emerald-400 mb-1 uppercase tracking-[0.2em]">Priority_Low</div>
<h4 class="text-xs font-bold headline-font uppercase text-white">Recursive_S_01</h4>
</div>
<div class="text-right">
<div class="text-[12px] headline-font font-bold text-emerald-400">85%</div>
<div class="text-[8px] mono-font text-on-surface-variant opacity-40">LOAD_STABLE</div>
</div>
</div>
<div class="h-1 bg-white/5 w-full relative">
<div class="absolute inset-0 bg-emerald-400/20 scanline w-full h-full"></div>
<div class="h-full bg-emerald-400 w-[85%] shadow-[0_0_10px_rgba(52,211,153,0.3)] relative"></div>
</div>
</div>
<!-- Complex Progress Card 2 -->
<div class="glass-card p-4 relative group cursor-pointer overflow-hidden border-primary/20 bg-primary/[0.02]">
<div class="flex justify-between items-start mb-3">
<div>
<div class="text-[8px] mono-font text-primary mb-1 uppercase tracking-[0.2em]">Priority_Mid</div>
<h4 class="text-xs font-bold headline-font uppercase text-white">Sync_Layer_B</h4>
</div>
<div class="text-right">
<div class="text-[12px] headline-font font-bold text-primary">45%</div>
<div class="text-[8px] mono-font text-on-surface-variant opacity-40">OPTIMIZING</div>
</div>
</div>
<div class="h-1 bg-white/5 w-full relative">
<div class="h-full bg-primary w-[45%] shadow-[0_0_10px_rgba(109,221,255,0.3)]"></div>
</div>
</div>
<!-- Complex Progress Card 3 -->
<div class="glass-card p-4 relative group cursor-pointer overflow-hidden border-tertiary/20 bg-tertiary/[0.02]">
<div class="flex justify-between items-start mb-3">
<div>
<div class="text-[8px] mono-font text-tertiary mb-1 uppercase tracking-[0.2em]">Priority_Crit</div>
<h4 class="text-xs font-bold headline-font uppercase text-white">Global_Cons_X</h4>
</div>
<div class="text-right">
<div class="text-[12px] headline-font font-bold text-tertiary">12%</div>
<div class="text-[8px] mono-font text-on-surface-variant opacity-40">WAIT_STATE</div>
</div>
</div>
<div class="h-1 bg-white/5 w-full relative">
<div class="h-full bg-tertiary w-[12%] shadow-[0_0_10px_rgba(255,151,181,0.3)]"></div>
</div>
</div>
<!-- Mini System Log HUD -->
<div class="glass-card p-3 bg-black/40 border-white/5 flex-1">
<div class="text-[8px] mono-font text-on-surface-variant/60 uppercase mb-2">Realtime_Log_Buffer</div>
<div class="space-y-1 overflow-hidden h-20 text-[9px] mono-font text-on-surface-variant/40">
<p>&gt; AUTH_TOKEN_VALIDATED [0.002ms]</p>
<p>&gt; BUFFER_ALLOCATED: 512MB</p>
<p class="text-primary/60">&gt; SOCKET_CONNECTED: 127.0.0.1:8080</p>
<p>&gt; RECV_PACKET_BATCH_ID: 991204</p>
<p class="text-tertiary/60">&gt; WARN: LATENCY_THRESHOLD_EXCEEDED</p>
</div>
</div>
</div>
</div>
<!-- Bottom Timeline Section -->
<section class="mt-4">
<div class="glass-card p-6 border-white/5 relative">
<div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-6">
<div class="flex items-center gap-10">
<div>
<div class="text-[9px] mono-font text-on-surface-variant uppercase mb-1 tracking-widest">Step_Counter</div>
<div class="text-3xl headline-font font-bold text-primary tracking-tighter">14,209.00</div>
</div>
<div class="h-8 w-px bg-white/10"></div>
<div>
<div class="text-[9px] mono-font text-on-surface-variant uppercase mb-1 tracking-widest">Perf_Efficiency</div>
<div class="text-3xl headline-font font-bold text-secondary tracking-tighter">0.982_α</div>
</div>
</div>
<div class="flex gap-2">
<button class="px-4 py-1.5 border border-white/10 text-[9px] mono-font uppercase tracking-widest hover:bg-white/5 transition-colors">Export_DMP</button>
<button class="px-6 py-1.5 bg-white text-background headline-font font-bold text-[9px] uppercase tracking-widest hover:bg-primary transition-all">Halt_Process</button>
</div>
</div>
<!-- Pro Timeline -->
<div class="relative pt-4">
<div class="flex justify-between text-[8px] mono-font text-on-surface-variant/40 font-bold mb-4 px-1">
<span>09:00:00</span><span>11:00:00</span><span>13:00:00</span><span>15:00:00</span><span>17:00:00</span>
</div>
<div class="h-10 bg-white/[0.02] border border-white/5 relative flex items-center px-1 overflow-hidden">
<!-- Tracks -->
<div class="absolute left-[10%] w-[12%] h-4 bg-primary/20 border-l-2 border-primary flex items-center px-2">
<span class="text-[7px] mono-font text-primary font-bold">TASK_A</span>
</div>
<div class="absolute left-[35%] w-[18%] h-4 bg-secondary/20 border-l-2 border-secondary flex items-center px-2">
<span class="text-[7px] mono-font text-secondary font-bold">BATCH_PROC</span>
</div>
<div class="absolute left-[60%] w-[8%] h-4 bg-tertiary/20 border-l-2 border-tertiary flex items-center px-2">
<span class="text-[7px] mono-font text-tertiary font-bold">SYNC</span>
</div>
<div class="absolute left-[75%] w-[15%] h-4 bg-emerald-400/20 border-l-2 border-emerald-400 flex items-center px-2">
<span class="text-[7px] mono-font text-emerald-400 font-bold">ARCHIVE</span>
</div>
<!-- Playhead -->
<div class="absolute left-[52%] top-0 bottom-0 w-px bg-white/80 shadow-[0_0_10px_white] z-20">
<div class="absolute top-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-white"></div>
</div>
</div>
</div>
</div>
</section>
<!-- Dynamic Background Watermark -->
<div class="fixed bottom-6 right-6 opacity-[0.03] headline-font text-8xl font-black select-none pointer-events-none tracking-tighter uppercase italic">
        Core_Env
    </div>
</main>
<!-- Mobile NavBar -->
<nav class="md:hidden fixed bottom-0 left-0 w-full h-14 bg-background/95 backdrop-blur-xl border-t border-white/10 flex justify-around items-center z-50 px-4">
<button class="text-primary flex flex-col items-center gap-1">
<span class="material-symbols-outlined text-xl" data-icon="terminal">terminal</span>
<span class="text-[7px] mono-font uppercase font-bold">Cmd</span>
</button>
<button class="text-on-surface-variant flex flex-col items-center gap-1">
<span class="material-symbols-outlined text-xl" data-icon="timeline">timeline</span>
<span class="text-[7px] mono-font uppercase">Data</span>
</button>
<button class="text-on-surface-variant flex flex-col items-center gap-1">
<span class="material-symbols-outlined text-xl" data-icon="hub">hub</span>
<span class="text-[7px] mono-font uppercase">Nodes</span>
</button>
</nav>
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
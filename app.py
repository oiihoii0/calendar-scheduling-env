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

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from scheduling_env.env import CalendarEnv

app = FastAPI(
    title="CalendarSchedulingEnv",
    description="OpenEnv-compatible calendar scheduling environment",
    version="1.0.0",
)

# ── Dashboard UI ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CalendarSchedulingEnv — OpenEnv Hackathon</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --p1:#8b5cf6;--p2:#6366f1;--p3:#3b82f6;
  --g1:linear-gradient(135deg,#8b5cf6,#6366f1,#3b82f6);
  --glass:rgba(255,255,255,0.04);
  --glass-border:rgba(255,255,255,0.08);
  --text:#f1f5f9;--muted:#64748b;--subtle:#1e293b;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  font-family:'Inter',sans-serif;background:#050508;
  color:var(--text);min-height:100vh;overflow-x:hidden;
}

/* ── Canvas BG ── */
#canvas{position:fixed;inset:0;z-index:0;pointer-events:none}

/* ── Noise overlay ── */
body::before{
  content:'';position:fixed;inset:0;z-index:1;pointer-events:none;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  background-size:200px;opacity:.5;
}

/* ── Main ── */
.wrap{position:relative;z-index:2;max-width:1160px;margin:0 auto;padding:0 24px 80px}

/* ── Nav ── */
nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:20px 0;border-bottom:1px solid rgba(255,255,255,.05);
  margin-bottom:80px;position:sticky;top:0;z-index:100;
  background:rgba(5,5,8,.8);backdrop-filter:blur(20px);
  margin-left:-24px;margin-right:-24px;padding-left:24px;padding-right:24px;
}
.nav-logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:15px;color:var(--text)}
.nav-logo .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 12px #22c55e;animation:breathe 2.5s ease-in-out infinite}
@keyframes breathe{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.6;transform:scale(.8)}}
.nav-links{display:flex;gap:4px}
.nav-link{
  padding:8px 16px;border-radius:10px;font-size:13px;font-weight:500;
  text-decoration:none;color:#94a3b8;transition:all .2s;border:1px solid transparent;
}
.nav-link:hover{color:#fff;background:var(--glass);border-color:var(--glass-border)}
.nav-link.primary{
  background:linear-gradient(135deg,rgba(139,92,246,.2),rgba(99,102,241,.2));
  border-color:rgba(139,92,246,.3);color:#c4b5fd;
}
.nav-link.primary:hover{background:linear-gradient(135deg,rgba(139,92,246,.3),rgba(99,102,241,.3));transform:translateY(-1px)}

/* ── Hero ── */
.hero{text-align:center;padding:40px 0 80px;position:relative}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  background:linear-gradient(135deg,rgba(139,92,246,.12),rgba(99,102,241,.12));
  border:1px solid rgba(139,92,246,.25);border-radius:999px;
  padding:8px 20px;font-size:12px;font-weight:600;color:#a78bfa;
  letter-spacing:.8px;text-transform:uppercase;margin-bottom:32px;
}
.hero-eyebrow::before{content:'✦';opacity:.7}
.hero-title{
  font-size:clamp(2.8rem,7vw,5.5rem);font-weight:900;line-height:1.05;
  letter-spacing:-2px;margin-bottom:24px;
  background:linear-gradient(135deg,#fff 0%,#c4b5fd 40%,#93c5fd 80%,#67e8f9 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-title span{
  background:var(--g1);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub{
  font-size:1.1rem;color:#94a3b8;max-width:560px;margin:0 auto 48px;
  line-height:1.7;font-weight:400;
}
.hero-cta{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:64px}
.btn{
  display:inline-flex;align-items:center;gap:8px;
  padding:13px 28px;border-radius:14px;font-size:14px;font-weight:600;
  text-decoration:none;transition:all .25s;border:1px solid transparent;cursor:pointer;
}
.btn-glow{
  background:var(--g1);color:#fff;
  box-shadow:0 0 0 0 rgba(139,92,246,0);
  animation:glow-btn 3s ease-in-out infinite;
}
@keyframes glow-btn{
  0%,100%{box-shadow:0 4px 24px rgba(139,92,246,.35)}
  50%{box-shadow:0 4px 48px rgba(139,92,246,.6),0 0 80px rgba(99,102,241,.2)}
}
.btn-glow:hover{transform:translateY(-3px) scale(1.02);filter:brightness(1.1)}
.btn-outline{
  background:var(--glass);border-color:var(--glass-border);color:#cbd5e1;
  backdrop-filter:blur(12px);
}
.btn-outline:hover{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.15);color:#fff;transform:translateY(-2px)}

/* ── Glow orbs ── */
.orb{position:absolute;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:-1}
.orb1{width:500px;height:500px;background:rgba(139,92,246,.12);top:-100px;left:50%;transform:translateX(-50%)}
.orb2{width:300px;height:300px;background:rgba(59,130,246,.08);bottom:-50px;right:-100px}

/* ── Stats ── */
.stats-row{
  display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:80px;
}
.stat-card{
  background:var(--glass);border:1px solid var(--glass-border);
  border-radius:20px;padding:28px 20px;text-align:center;
  transition:all .3s;position:relative;overflow:hidden;
}
.stat-card::before{
  content:'';position:absolute;inset:0;border-radius:20px;
  background:var(--g1);opacity:0;transition:opacity .3s;
}
.stat-card:hover::before{opacity:.05}
.stat-card:hover{transform:translateY(-4px);border-color:rgba(139,92,246,.25)}
.stat-num{
  font-size:2.6rem;font-weight:900;letter-spacing:-1px;
  background:var(--g1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.stat-label{font-size:12px;color:var(--muted);margin-top:6px;font-weight:500;letter-spacing:.3px}

/* ── Section heading ── */
.section-label{
  display:flex;align-items:center;gap:12px;margin-bottom:32px;
}
.section-label h2{font-size:1.4rem;font-weight:700;color:var(--text)}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(255,255,255,.07),transparent)}

/* ── Task cards ── */
.tasks-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:80px}
.task-card{
  background:var(--glass);border:1px solid var(--glass-border);
  border-radius:24px;padding:32px;transition:all .35s;
  position:relative;overflow:hidden;cursor:default;
}
.task-card::after{
  content:'';position:absolute;inset:-1px;border-radius:24px;
  background:linear-gradient(135deg,var(--c1),var(--c2));
  opacity:0;transition:opacity .35s;z-index:-1;
}
.task-card:hover{transform:translateY(-6px) scale(1.01);}
.task-card:hover::after{opacity:.15}
.task-card.easy{--c1:#10b981;--c2:#059669}
.task-card.medium{--c1:#f59e0b;--c2:#d97706}
.task-card.hard{--c1:#ef4444;--c2:#dc2626}
.task-card:hover.easy{border-color:rgba(16,185,129,.4);box-shadow:0 20px 60px rgba(16,185,129,.1)}
.task-card:hover.medium{border-color:rgba(245,158,11,.4);box-shadow:0 20px 60px rgba(245,158,11,.1)}
.task-card:hover.hard{border-color:rgba(239,68,68,.4);box-shadow:0 20px 60px rgba(239,68,68,.1)}

.diff-badge{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 14px;border-radius:999px;font-size:10px;font-weight:800;
  letter-spacing:1.2px;text-transform:uppercase;margin-bottom:20px;
}
.easy .diff-badge{background:rgba(16,185,129,.12);color:#34d399;border:1px solid rgba(16,185,129,.2)}
.medium .diff-badge{background:rgba(245,158,11,.12);color:#fbbf24;border:1px solid rgba(245,158,11,.2)}
.hard .diff-badge{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.2)}
.diff-badge::before{content:'';width:5px;height:5px;border-radius:50%;background:currentColor}

.task-name{font-size:1.25rem;font-weight:800;color:#f8fafc;margin-bottom:10px;letter-spacing:-.3px}
.task-desc{font-size:13.5px;color:#64748b;line-height:1.65;margin-bottom:24px}
.task-pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{
  display:inline-flex;align-items:center;gap:5px;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
  border-radius:8px;padding:5px 10px;font-size:11px;color:#475569;font-weight:500;
}

/* ── Endpoints ── */
.endpoints-grid{display:flex;flex-direction:column;gap:8px;margin-bottom:80px}
.ep{
  display:grid;grid-template-columns:60px 1fr auto;align-items:center;gap:16px;
  background:var(--glass);border:1px solid var(--glass-border);
  border-radius:14px;padding:16px 20px;transition:all .2s;
  text-decoration:none;
}
.ep:hover{background:rgba(255,255,255,.06);border-color:rgba(255,255,255,.12);transform:translateX(4px)}
.method{
  padding:5px 0;border-radius:7px;font-size:10px;font-weight:800;
  text-align:center;letter-spacing:.8px;font-family:'JetBrains Mono',monospace;
}
.GET{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.2)}
.POST{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.2)}
.DELETE{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.2)}
.ep-path{font-family:'JetBrains Mono',monospace;font-size:13px;color:#e2e8f0;font-weight:600}
.ep-desc{font-size:12px;color:var(--muted);text-align:right}

/* ── Reward table ── */
.reward-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:80px;
}
.reward-row{
  display:flex;justify-content:space-between;align-items:center;
  background:var(--glass);border:1px solid var(--glass-border);
  border-radius:12px;padding:14px 18px;transition:all .2s;
}
.reward-row:hover{background:rgba(255,255,255,.06)}
.reward-label{font-size:13px;color:#94a3b8}
.reward-val{
  font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;
  padding:3px 10px;border-radius:6px;
}
.pos{color:#34d399;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.15)}
.neg{color:#f87171;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.15)}

/* ── Footer ── */
footer{
  border-top:1px solid rgba(255,255,255,.05);padding-top:40px;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;
}
footer p{font-size:13px;color:#334155}
footer .links{display:flex;gap:20px}
footer a{font-size:13px;color:#475569;text-decoration:none;transition:color .2s}
footer a:hover{color:#a78bfa}

/* ── Animations ── */
.fade-up{opacity:0;transform:translateY(30px);transition:opacity .7s ease,transform .7s ease}
.fade-up.visible{opacity:1;transform:translateY(0)}

@media(max-width:768px){
  .stats-row{grid-template-columns:repeat(2,1fr)}
  .tasks-grid{grid-template-columns:1fr}
  .reward-grid{grid-template-columns:1fr}
  .hero-title{font-size:2.5rem}
}
</style>
</head>
<body>
<canvas id="canvas"></canvas>

<div class="wrap">
  <!-- Nav -->
  <nav>
    <div class="nav-logo">
      <div class="dot"></div>
      CalendarSchedulingEnv
    </div>
    <div class="nav-links">
      <a href="/docs" class="nav-link">API Docs</a>
      <a href="/tasks" class="nav-link">Tasks JSON</a>
      <a href="/health" class="nav-link">Health</a>
      <a href="/docs" class="nav-link primary">🚀 Try it</a>
    </div>
  </nav>

  <!-- Hero -->
  <section class="hero fade-up">
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="hero-eyebrow">OpenEnv Hackathon · Live Environment</div>
    <h1 class="hero-title">Calendar<br/><span>Scheduling</span><br/>Environment</h1>
    <p class="hero-sub">An OpenEnv-compatible AI training ground where agents master real-world meeting scheduling — juggling rooms, attendees, priorities and constraints.</p>
    <div class="hero-cta">
      <a href="/docs" class="btn btn-glow">🚀 Try the API</a>
      <a href="/tasks" class="btn btn-outline">📋 View Tasks</a>
      <a href="/health" class="btn btn-outline">💚 Health Check</a>
    </div>
  </section>

  <!-- Stats -->
  <div class="stats-row fade-up">
    <div class="stat-card">
      <div class="stat-num" data-target="3">0</div>
      <div class="stat-label">Difficulty Levels</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="12">0</div>
      <div class="stat-label">Max Events · Hard</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="5">0</div>
      <div class="stat-label">Rooms Available</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">0–1</div>
      <div class="stat-label">Normalised Score Range</div>
    </div>
  </div>

  <!-- Tasks -->
  <div class="section-label fade-up"><h2>🎯 Challenge Tasks</h2></div>
  <div class="tasks-grid fade-up">
    <div class="task-card easy">
      <div class="diff-badge">Easy</div>
      <div class="task-name">Simple Scheduling</div>
      <div class="task-desc">Schedule 5 meetings in an 8-hour work day. Wide time windows, single room — ideal for agents learning the basics of calendar management.</div>
      <div class="task-pills">
        <span class="pill">📅 5 events</span>
        <span class="pill">🏠 1 room</span>
        <span class="pill">⏱ 50 steps max</span>
      </div>
    </div>
    <div class="task-card medium">
      <div class="diff-badge">Medium</div>
      <div class="task-name">Constrained Scheduling</div>
      <div class="task-desc">8 meetings, 3 rooms, room requirements, lunch break enforced, attendee conflicts — agents must balance multiple hard and soft constraints.</div>
      <div class="task-pills">
        <span class="pill">📅 8 events</span>
        <span class="pill">🏠 3 rooms</span>
        <span class="pill">⏱ 80 steps max</span>
      </div>
    </div>
    <div class="task-card hard">
      <div class="diff-badge">Hard</div>
      <div class="task-name">Complex Scheduling</div>
      <div class="task-desc">12 meetings, 5 rooms across 2 buildings with travel time between floors, CEO availability caps, and razor-thin time windows. Only frontier models succeed.</div>
      <div class="task-pills">
        <span class="pill">📅 12 events</span>
        <span class="pill">🏠 5 rooms</span>
        <span class="pill">⏱ 120 steps max</span>
      </div>
    </div>
  </div>

  <!-- Endpoints -->
  <div class="section-label fade-up"><h2>🔌 API Endpoints</h2></div>
  <div class="endpoints-grid fade-up">
    <a href="/health" class="ep"><span class="method GET">GET</span><span class="ep-path">/health</span><span class="ep-desc">Server status check</span></a>
    <a href="/tasks" class="ep"><span class="method GET">GET</span><span class="ep-path">/tasks</span><span class="ep-desc">List all tasks</span></a>
    <div class="ep"><span class="method POST">POST</span><span class="ep-path">/reset</span><span class="ep-desc">Start new episode → initial observation</span></div>
    <div class="ep"><span class="method POST">POST</span><span class="ep-path">/step</span><span class="ep-desc">Schedule one meeting → reward + next obs</span></div>
    <div class="ep"><span class="method GET">GET</span><span class="ep-path">/state</span><span class="ep-desc">Full calendar state snapshot</span></div>
    <div class="ep"><span class="method DELETE">DELETE</span><span class="ep-path">/session/{id}</span><span class="ep-desc">Close active session</span></div>
  </div>

  <!-- Rewards -->
  <div class="section-label fade-up"><h2>⭐ Reward Structure</h2></div>
  <div class="reward-grid fade-up">
    <div class="reward-row"><span class="reward-label">Valid placement</span><span class="reward-val pos">+1.0</span></div>
    <div class="reward-row"><span class="reward-label">Room conflict penalty</span><span class="reward-val neg">−2.0</span></div>
    <div class="reward-row"><span class="reward-label">Priority bonus (×priority)</span><span class="reward-val pos">+0.2</span></div>
    <div class="reward-row"><span class="reward-label">Attendee conflict penalty</span><span class="reward-val neg">−2.0</span></div>
    <div class="reward-row"><span class="reward-label">Zero-conflict completion</span><span class="reward-val pos">+5.0</span></div>
    <div class="reward-row"><span class="reward-label">Hard constraint violation</span><span class="reward-val neg">−3.0</span></div>
    <div class="reward-row"><span class="reward-label">All events scheduled bonus</span><span class="reward-val pos">+0.5×N</span></div>
    <div class="reward-row"><span class="reward-label">Soft constraint violation</span><span class="reward-val neg">−1.0</span></div>
  </div>

  <!-- Footer -->
  <footer class="fade-up">
    <p>Built for the <strong>OpenEnv Hackathon</strong> · Powered by Gymnasium &amp; FastAPI</p>
    <div class="links">
      <a href="/docs">API Docs</a>
      <a href="/tasks">Tasks</a>
      <a href="/health">Status</a>
    </div>
  </footer>
</div>

<script>
// ── Particles ──
const canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d');
let W,H,pts=[];
function resize(){W=canvas.width=innerWidth;H=canvas.height=innerHeight;init()}
function init(){
  pts=[];
  const n=Math.floor((W*H)/18000);
  for(let i=0;i<n;i++){
    pts.push({
      x:Math.random()*W,y:Math.random()*H,
      vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,
      r:Math.random()*1.5+.5,
      a:Math.random()*.6+.1,
      hue:Math.random()<.5?260:220,
    });
  }
}
function draw(){
  ctx.clearRect(0,0,W,H);
  pts.forEach(p=>{
    p.x+=p.vx;p.y+=p.vy;
    if(p.x<0)p.x=W;if(p.x>W)p.x=0;
    if(p.y<0)p.y=H;if(p.y>H)p.y=0;
    ctx.beginPath();
    ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx.fillStyle=`hsla(${p.hue},80%,70%,${p.a})`;
    ctx.fill();
  });
  // lines between nearby pts
  for(let i=0;i<pts.length;i++){
    for(let j=i+1;j<pts.length;j++){
      const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y;
      const d=Math.sqrt(dx*dx+dy*dy);
      if(d<120){
        ctx.beginPath();
        ctx.moveTo(pts[i].x,pts[i].y);
        ctx.lineTo(pts[j].x,pts[j].y);
        ctx.strokeStyle=`rgba(139,92,246,${.08*(1-d/120)})`;
        ctx.lineWidth=.5;
        ctx.stroke();
      }
    }
  }
  requestAnimationFrame(draw);
}
window.addEventListener('resize',resize);
resize();draw();

// ── Counter animation ──
function animateCount(el,target,duration=1500){
  const start=performance.now();
  function step(now){
    const pct=Math.min((now-start)/duration,1);
    const ease=1-Math.pow(1-pct,4);
    el.textContent=Math.round(ease*target);
    if(pct<1)requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Scroll animations ──
const observer=new IntersectionObserver(entries=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      e.target.classList.add('visible');
      // trigger counters
      e.target.querySelectorAll('[data-target]').forEach(el=>{
        animateCount(el,+el.dataset.target);
      });
    }
  });
},{threshold:.15});
document.querySelectorAll('.fade-up').forEach(el=>observer.observe(el));
</script>
</body>
</html>"""

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CalendarSchedulingEnv — OpenEnv</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Inter',sans-serif;background:#0a0a0f;color:#e2e8f0;min-height:100vh;overflow-x:hidden}
  .bg{position:fixed;inset:0;background:radial-gradient(ellipse at 20% 20%,#1a0533 0%,transparent 50%),radial-gradient(ellipse at 80% 80%,#0d1f3c 0%,transparent 50%),#0a0a0f;z-index:-1}
  .container{max-width:1100px;margin:0 auto;padding:48px 24px}

  /* Hero */
  .hero{text-align:center;margin-bottom:64px}
  .badge{display:inline-flex;align-items:center;gap:8px;background:rgba(139,92,246,.15);border:1px solid rgba(139,92,246,.3);border-radius:999px;padding:6px 16px;font-size:13px;color:#a78bfa;margin-bottom:24px;letter-spacing:.5px}
  .badge::before{content:'';width:8px;height:8px;border-radius:50%;background:#a78bfa;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
  h1{font-size:clamp(2rem,5vw,3.5rem);font-weight:800;background:linear-gradient(135deg,#fff 0%,#a78bfa 50%,#60a5fa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;margin-bottom:16px}
  .subtitle{font-size:1.1rem;color:#94a3b8;max-width:600px;margin:0 auto 32px;line-height:1.6}
  .hero-actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
  .btn{padding:12px 28px;border-radius:12px;font-size:14px;font-weight:600;text-decoration:none;transition:all .2s;cursor:pointer;border:none}
  .btn-primary{background:linear-gradient(135deg,#7c3aed,#2563eb);color:#fff;box-shadow:0 4px 20px rgba(124,58,237,.3)}
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(124,58,237,.4)}
  .btn-ghost{background:rgba(255,255,255,.05);color:#cbd5e1;border:1px solid rgba(255,255,255,.1)}
  .btn-ghost:hover{background:rgba(255,255,255,.1);transform:translateY(-2px)}

  /* Stats bar */
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:64px}
  .stat{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:24px;text-align:center}
  .stat-num{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-label{font-size:13px;color:#64748b;margin-top:4px}

  /* Tasks */
  .section-title{font-size:1.5rem;font-weight:700;color:#f1f5f9;margin-bottom:24px}
  .tasks{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:64px}
  .task-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:20px;padding:28px;transition:all .3s;position:relative;overflow:hidden}
  .task-card:hover{transform:translateY(-4px);border-color:rgba(139,92,246,.3);background:rgba(139,92,246,.05)}
  .task-card::before{content:'';position:absolute;inset:0;background:var(--glow);opacity:0;transition:opacity .3s;pointer-events:none}
  .task-card:hover::before{opacity:1}
  .task-card.easy{--glow:radial-gradient(circle at top right,rgba(16,185,129,.08),transparent 60%)}
  .task-card.medium{--glow:radial-gradient(circle at top right,rgba(245,158,11,.08),transparent 60%)}
  .task-card.hard{--glow:radial-gradient(circle at top right,rgba(239,68,68,.08),transparent 60%)}
  .difficulty{display:inline-block;padding:4px 12px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
  .easy .difficulty{background:rgba(16,185,129,.15);color:#10b981}
  .medium .difficulty{background:rgba(245,158,11,.15);color:#f59e0b}
  .hard .difficulty{background:rgba(239,68,68,.15);color:#ef4444}
  .task-name{font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:8px}
  .task-desc{font-size:14px;color:#64748b;line-height:1.6;margin-bottom:16px}
  .task-meta{display:flex;gap:16px}
  .task-meta span{font-size:12px;color:#475569;display:flex;align-items:center;gap:4px}

  /* Endpoints */
  .endpoints{margin-bottom:64px}
  .endpoint{display:flex;align-items:center;gap:16px;padding:16px 20px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:12px;margin-bottom:8px;transition:all .2s}
  .endpoint:hover{background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.1)}
  .method{padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700;min-width:52px;text-align:center}
  .get{background:rgba(59,130,246,.2);color:#60a5fa}
  .post{background:rgba(16,185,129,.2);color:#10b981}
  .delete{background:rgba(239,68,68,.2);color:#ef4444}
  .path{font-family:monospace;font-size:14px;color:#e2e8f0;flex:1}
  .ep-desc{font-size:13px;color:#475569}

  /* Footer */
  .footer{text-align:center;padding-top:32px;border-top:1px solid rgba(255,255,255,.05);color:#334155;font-size:13px}
  .footer a{color:#7c3aed;text-decoration:none}
</style>
</head>
<body>
<div class="bg"></div>
<div class="container">

  <div class="hero">
    <div class="badge">🟢 Live on HuggingFace Spaces</div>
    <h1>Calendar Scheduling<br/>Environment</h1>
    <p class="subtitle">An OpenEnv-compatible AI training environment where agents learn to optimally schedule meetings — respecting rooms, attendees, priorities and real-world constraints.</p>
    <div class="hero-actions">
      <a href="/docs" class="btn btn-primary">🚀 Try the API</a>
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



# In-memory session store: session_id -> CalendarEnv
_sessions: Dict[str, CalendarEnv] = {}

TASKS = [
    "CalendarSchedulingEasy-v0",
    "CalendarSchedulingMedium-v0",
    "CalendarSchedulingHard-v0",
]

# Max possible scores per task for normalisation
_MAX_SCORES = {
    "CalendarSchedulingEasy-v0":   17.5,
    "CalendarSchedulingMedium-v0": 35.0,
    "CalendarSchedulingHard-v0":   56.0,
}


# ── Request / response models ─────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "CalendarSchedulingEasy-v0"
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
                "id": "CalendarSchedulingEasy-v0",
                "name": "Simple Scheduling",
                "difficulty": "easy",
                "description": "Schedule 5 meetings in an 8-hour day. No room constraints.",
                "max_steps": 50,
                "grader": "scheduling_env.grader:grade_schedule",
            },
            {
                "id": "CalendarSchedulingMedium-v0",
                "name": "Constrained Scheduling",
                "difficulty": "medium",
                "description": "Schedule 8 meetings with room requirements and lunch break.",
                "max_steps": 80,
                "grader": "scheduling_env.grader:grade_schedule",
            },
            {
                "id": "CalendarSchedulingHard-v0",
                "name": "Complex Scheduling",
                "difficulty": "hard",
                "description": "Schedule 12 meetings across 5 rooms with travel time constraints.",
                "max_steps": 120,
                "grader": "scheduling_env.grader:grade_schedule",
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

    return {
        "session_id": session_id,
        "observation": _serialize_obs(env),
        "info": info,
        "task": req.task_name,
        "difficulty": env.task_difficulty,
    }


@app.post("/step")
def step(req: StepRequest = Body(default=StepRequest())):
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

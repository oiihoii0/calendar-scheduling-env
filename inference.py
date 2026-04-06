"""
Inference Script — CalendarSchedulingEnv
=========================================
OpenEnv hackathon submission.

Environment variables required:
    API_BASE_URL    LLM endpoint  (default: HuggingFace router)
    MODEL_NAME      Model id      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        HF / API key
    LOCAL_IMAGE_NAME  Docker image name (if running locally)
    CALENDAR_ENV_URL  URL of the running CalendarSchedulingEnv server
                      (default: http://localhost:7860)
    CALENDAR_TASK     Task name   (default: simple_scheduling)

STDOUT format (mandatory):
    [START] task=<name> env=CalendarSchedulingEnv model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────

API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL       = os.getenv("CALENDAR_ENV_URL", "http://localhost:7860")
TASK_NAME     = os.getenv("CALENDAR_TASK", "simple_scheduling")
BENCHMARK     = "CalendarSchedulingEnv"
MAX_STEPS     = int(os.getenv("MAX_STEPS", "30"))
TEMPERATURE   = 0.2
MAX_TOKENS    = 300
SUCCESS_THRESHOLD = 0.5   # score ≥ 0.5 → success


# ── Structured logging ────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert scheduler. Your task is to assign meetings to time slots and rooms on a work calendar (09:00–17:00).

Rules:
- Each event must be scheduled within its earliest_start_hour and latest_start_hour window.
- No two events can overlap in the same room at the same time.
- No two events sharing attendees can overlap in time.
- If an event has a room_required field, it MUST be placed in that room.
- Prefer scheduling higher-priority events (priority 5 is highest) first.
- Avoid the lunch window (12:00–13:00) when possible.

Respond with ONLY a single JSON object, no explanation:
{
  "event_title": "<exact title from unscheduled_events>",
  "start_hour": <float, e.g. 9.0 or 10.5>,
  "room": "<room name from available_rooms>"
}
""").strip()


def _build_user_prompt(obs: Dict[str, Any], step: int) -> str:
    unscheduled = obs.get("unscheduled_events", [])
    scheduled = obs.get("scheduled_events", [])
    rooms = obs.get("available_rooms", [])
    conflicts = obs.get("conflicts", 0)

    unscheduled_block = "\n".join(
        f"  - {e['title']} | {e['duration_minutes']}min | "
        f"window [{e['earliest_start_hour']}:00–{e['latest_start_hour']}:00] | "
        f"priority {e['priority']} | attendees: {', '.join(e['attendees'])}"
        + (f" | ROOM REQUIRED: {e['room_required']}" if e.get("room_required") else "")
        for e in unscheduled
    ) or "  (all events scheduled)"

    scheduled_block = "\n".join(
        f"  - {e['title']} | {e['start_hour']:.1f}–{e['end_hour']:.1f} | {e['room']}"
        for e in scheduled
    ) or "  (none yet)"

    return textwrap.dedent(f"""
Step {step} | Conflicts so far: {conflicts}
Events remaining: {len(unscheduled)} / {obs.get('events_total', '?')}

UNSCHEDULED EVENTS:
{unscheduled_block}

ALREADY SCHEDULED:
{scheduled_block}

AVAILABLE ROOMS: {", ".join(rooms)}

Choose ONE event to schedule next. Reply with JSON only.
""").strip()


# ── LLM call ─────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Ask the LLM to pick the next scheduling action. Returns parsed JSON."""
    user_prompt = _build_user_prompt(obs, step)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback: pick first unscheduled event at earliest start
        unscheduled = obs.get("unscheduled_events", [])
        rooms = obs.get("available_rooms", ["Main Room"])
        if unscheduled:
            ev = unscheduled[0]
            return {
                "event_title": ev["title"],
                "start_hour": float(ev["earliest_start_hour"]),
                "room": ev.get("room_required") or rooms[0],
            }
        return {"event_title": "", "start_hour": 9.0, "room": rooms[0] if rooms else "Main Room"}


# ── HTTP client for the environment ──────────────────────────────────

async def env_reset(client: httpx.AsyncClient, task: str, seed: int = 42) -> Dict[str, Any]:
    resp = await client.post("/reset", json={"task_name": task, "seed": seed, "session_id": "default"})
    resp.raise_for_status()
    return resp.json()


async def env_step(client: httpx.AsyncClient, action: Dict[str, Any]) -> Dict[str, Any]:
    payload = {**action, "session_id": "default"}
    resp = await client.post("/step", json=payload)
    resp.raise_for_status()
    return resp.json()


async def env_state(client: httpx.AsyncClient) -> Dict[str, Any]:
    resp = await client.get("/state", params={"session_id": "default"})
    resp.raise_for_status()
    return resp.json()


async def env_close(client: httpx.AsyncClient) -> None:
    await client.delete("/session/default")


# ── Main loop ─────────────────────────────────────────────────────────

async def main() -> None:
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient(base_url=ENV_URL, timeout=30.0) as http:
        try:
            # Reset environment
            reset_data = await env_reset(http, TASK_NAME)
            obs = reset_data["observation"]
            session_id = reset_data.get("session_id", "default")

            for step in range(1, MAX_STEPS + 1):
                if not obs.get("unscheduled_events"):
                    # All events scheduled — done
                    log_step(step, "none", 0.0, True, None)
                    rewards.append(0.0)
                    steps_taken = step
                    break

                # Ask LLM for action
                action_json = _call_llm(openai_client, obs, step)
                action_str = (
                    f"{action_json.get('event_title','?')}@"
                    f"{action_json.get('start_hour','?')}"
                    f"/{action_json.get('room','?')}"
                )

                # Execute action
                error_msg = None
                try:
                    step_data = await env_step(http, action_json)
                    reward = step_data.get("reward", 0.0)
                    done = step_data.get("done", False)
                    obs = step_data.get("observation", obs)
                    score = step_data.get("score", 0.0)
                except Exception as exc:
                    reward = 0.0
                    done = False
                    error_msg = str(exc)[:80]

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

                if done:
                    break

            # Final state
            try:
                final_state = await env_state(http)
                score = final_state.get("normalized_score", score)
            except Exception:
                pass

            success = score >= SUCCESS_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            success = False

        finally:
            try:
                await env_close(http)
            except Exception as e:
                print(f"[DEBUG] env close error: {e}", flush=True)

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

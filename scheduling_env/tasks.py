"""
Task definitions for CalendarSchedulingEnv.

Three difficulty tiers:
    easy   – ``create_simple_scheduling_task``
    medium – ``create_constrained_scheduling_task``
    hard   – ``create_complex_scheduling_task``

Each factory returns a dict with keys: events, constraints, rooms,
max_steps, difficulty, description.
"""

from __future__ import annotations

from typing import Any, Dict, List

from scheduling_env.models import CalendarConstraint, Event


# ---------------------------------------------------------------------------
# EASY: simple_scheduling
# ---------------------------------------------------------------------------

def create_simple_scheduling_task() -> Dict[str, Any]:
    """Easy task: 5 non-overlapping events, wide time windows, no rooms.

    The agent just needs to place events into valid slots without conflicts.
    """
    events: List[Event] = [
        Event(
            id=0,
            title="Team Sync",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=16,
            priority=3,
            attendees=["Alice", "Bob"],
        ),
        Event(
            id=1,
            title="1:1 Review",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=16,
            priority=4,
            attendees=["Alice", "Charlie"],
        ),
        Event(
            id=2,
            title="Planning",
            duration_minutes=60,
            earliest_start_hour=10,
            latest_start_hour=15,
            priority=5,
            attendees=["Alice", "Bob", "Charlie"],
        ),
        Event(
            id=3,
            title="Code Review",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=16,
            priority=2,
            attendees=["Bob"],
        ),
        Event(
            id=4,
            title="Standup",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=11,
            priority=3,
            attendees=["Alice", "Bob", "Charlie"],
        ),
    ]

    constraints: List[CalendarConstraint] = [
        CalendarConstraint(
            constraint_type="no_overlap",
            severity="hard",
        ),
    ]

    return {
        "events": events,
        "constraints": constraints,
        "rooms": ["Main Room"],
        "max_steps": 50,
        "difficulty": "easy",
        "description": (
            "Schedule 5 meetings within an 8-hour work day (09:00–17:00). "
            "Events have wide time windows and no room constraints. "
            "Goal: zero conflicts and all events scheduled."
        ),
    }


# ---------------------------------------------------------------------------
# MEDIUM: constrained_scheduling
# ---------------------------------------------------------------------------

def create_constrained_scheduling_task() -> Dict[str, Any]:
    """Medium task: 8 events with room constraints, attendee overlaps, lunch.

    Adds room requirements, a mandatory lunch break, and tighter windows.
    """
    events: List[Event] = [
        Event(
            id=0,
            title="Sprint Planning",
            duration_minutes=90,
            earliest_start_hour=9,
            latest_start_hour=11,
            priority=5,
            attendees=["Alice", "Bob", "Charlie", "Diana"],
            room_required="Conference A",
        ),
        Event(
            id=1,
            title="Design Review",
            duration_minutes=60,
            earliest_start_hour=10,
            latest_start_hour=14,
            priority=4,
            attendees=["Alice", "Eve"],
            room_required="Conference B",
        ),
        Event(
            id=2,
            title="1:1 with Manager",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=16,
            priority=4,
            attendees=["Bob", "Frank"],
        ),
        Event(
            id=3,
            title="API Workshop",
            duration_minutes=90,
            earliest_start_hour=13,
            latest_start_hour=15,
            priority=3,
            attendees=["Charlie", "Diana", "Eve"],
            room_required="Conference A",
        ),
        Event(
            id=4,
            title="Team Standup",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=10,
            priority=5,
            attendees=["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        ),
        Event(
            id=5,
            title="Code Pairing",
            duration_minutes=60,
            earliest_start_hour=10,
            latest_start_hour=16,
            priority=2,
            attendees=["Bob", "Charlie"],
        ),
        Event(
            id=6,
            title="Product Demo",
            duration_minutes=60,
            earliest_start_hour=14,
            latest_start_hour=16,
            priority=4,
            attendees=["Alice", "Frank", "Diana"],
            room_required="Conference A",
        ),
        Event(
            id=7,
            title="Retrospective",
            duration_minutes=60,
            earliest_start_hour=15,
            latest_start_hour=16,
            priority=3,
            attendees=["Alice", "Bob", "Charlie", "Diana"],
            room_required="Conference B",
        ),
    ]

    constraints: List[CalendarConstraint] = [
        CalendarConstraint(
            constraint_type="no_overlap",
            severity="hard",
        ),
        CalendarConstraint(
            constraint_type="lunch_break",
            time_window=(12.0, 13.0),
            severity="soft",
        ),
        CalendarConstraint(
            constraint_type="attendee_unavailable",
            event_id=2,
            time_window=(12.0, 14.0),
            severity="hard",
        ),
    ]

    return {
        "events": events,
        "constraints": constraints,
        "rooms": ["Conference A", "Conference B", "Huddle Room"],
        "max_steps": 80,
        "difficulty": "medium",
        "description": (
            "Schedule 8 meetings with room requirements, a lunch-break "
            "constraint (12:00–13:00), and attendee-unavailability windows. "
            "Goal: respect all hard constraints and minimise soft violations."
        ),
    }


# ---------------------------------------------------------------------------
# HARD: complex_scheduling
# ---------------------------------------------------------------------------

def create_complex_scheduling_task() -> Dict[str, Any]:
    """Hard task: 12 events, travel time, tight windows, all constraint types.

    The agent must juggle rooms across buildings (travel time), respect
    lunch breaks, handle attendee unavailability, and maximise priority
    satisfaction under very tight scheduling windows.
    """
    events: List[Event] = [
        Event(
            id=0,
            title="Board Meeting",
            duration_minutes=90,
            earliest_start_hour=9,
            latest_start_hour=10,
            priority=5,
            attendees=["Alice", "Bob", "CEO", "CFO"],
            room_required="Boardroom",
        ),
        Event(
            id=1,
            title="Architecture Review",
            duration_minutes=90,
            earliest_start_hour=9,
            latest_start_hour=11,
            priority=5,
            attendees=["Alice", "Charlie", "Diana"],
            room_required="Conference A",
        ),
        Event(
            id=2,
            title="Security Audit",
            duration_minutes=60,
            earliest_start_hour=10,
            latest_start_hour=14,
            priority=4,
            attendees=["Eve", "Frank"],
            room_required="Lab",
        ),
        Event(
            id=3,
            title="Stakeholder Update",
            duration_minutes=60,
            earliest_start_hour=11,
            latest_start_hour=13,
            priority=5,
            attendees=["Alice", "CEO", "Product"],
            room_required="Boardroom",
        ),
        Event(
            id=4,
            title="Sprint Demo",
            duration_minutes=60,
            earliest_start_hour=14,
            latest_start_hour=16,
            priority=4,
            attendees=["Alice", "Bob", "Charlie", "Diana", "Eve"],
            room_required="Conference A",
        ),
        Event(
            id=5,
            title="1:1 Alice/Bob",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=16,
            priority=3,
            attendees=["Alice", "Bob"],
        ),
        Event(
            id=6,
            title="1:1 Charlie/Diana",
            duration_minutes=30,
            earliest_start_hour=9,
            latest_start_hour=16,
            priority=3,
            attendees=["Charlie", "Diana"],
        ),
        Event(
            id=7,
            title="API Design",
            duration_minutes=90,
            earliest_start_hour=13,
            latest_start_hour=15,
            priority=4,
            attendees=["Charlie", "Eve", "Frank"],
            room_required="Conference B",
        ),
        Event(
            id=8,
            title="Incident Postmortem",
            duration_minutes=60,
            earliest_start_hour=10,
            latest_start_hour=15,
            priority=5,
            attendees=["Alice", "Bob", "Eve"],
            room_required="Conference A",
        ),
        Event(
            id=9,
            title="Team Lunch",
            duration_minutes=60,
            earliest_start_hour=12,
            latest_start_hour=13,
            priority=2,
            attendees=["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        ),
        Event(
            id=10,
            title="Training Session",
            duration_minutes=90,
            earliest_start_hour=14,
            latest_start_hour=15,
            priority=3,
            attendees=["Frank", "Diana"],
            room_required="Lab",
        ),
        Event(
            id=11,
            title="Wrap-up Standup",
            duration_minutes=30,
            earliest_start_hour=16,
            latest_start_hour=16,
            priority=4,
            attendees=["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        ),
    ]

    constraints: List[CalendarConstraint] = [
        CalendarConstraint(
            constraint_type="no_overlap",
            severity="hard",
        ),
        CalendarConstraint(
            constraint_type="lunch_break",
            time_window=(12.0, 13.0),
            severity="hard",
        ),
        # Travel time between buildings (Boardroom ↔ Lab is 15 min)
        CalendarConstraint(
            constraint_type="travel_time",
            time_window=(0.25, 0.25),  # 15 min buffer encoded as hours
            severity="hard",
        ),
        # CEO unavailable after 14:00
        CalendarConstraint(
            constraint_type="attendee_unavailable",
            event_id=3,
            time_window=(14.0, 17.0),
            severity="hard",
        ),
        # Soft: prefer not to schedule back-to-back for Alice
        CalendarConstraint(
            constraint_type="no_overlap",
            event_id=5,
            severity="soft",
        ),
    ]

    return {
        "events": events,
        "constraints": constraints,
        "rooms": ["Boardroom", "Conference A", "Conference B", "Lab", "Huddle Room"],
        "max_steps": 120,
        "difficulty": "hard",
        "description": (
            "Schedule 12 meetings across 5 rooms in two buildings. "
            "Constraints include travel time between buildings, a hard "
            "lunch break, CEO availability limits, and attendee conflicts. "
            "Goal: maximise priority-weighted scheduling with zero hard "
            "constraint violations."
        ),
    }

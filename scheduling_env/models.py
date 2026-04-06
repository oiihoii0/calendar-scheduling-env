"""
Data models for the CalendarSchedulingEnv.

Defines the core data structures used throughout the environment:
Event, TimeSlot, CalendarConstraint, and ScheduleState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Event:
    """A calendar event to be scheduled.

    Attributes:
        id: Unique identifier for the event.
        title: Human-readable name (e.g. "Team Sync", "1:1 Review").
        duration_minutes: Length of the event in minutes (30, 60, or 90).
        earliest_start_hour: Earliest allowed start hour (9–17).
        latest_start_hour: Latest allowed start hour (9–17).
        priority: Importance level from 1 (lowest) to 5 (highest).
        attendees: List of attendee names.
        room_required: Optional specific room requirement.
    """

    id: int
    title: str
    duration_minutes: int
    earliest_start_hour: int
    latest_start_hour: int
    priority: int
    attendees: List[str]
    room_required: Optional[str] = None

    def __post_init__(self) -> None:
        if self.duration_minutes not in (30, 60, 90):
            raise ValueError(
                f"duration_minutes must be 30, 60, or 90, got {self.duration_minutes}"
            )
        if not (9 <= self.earliest_start_hour <= 17):
            raise ValueError(
                f"earliest_start_hour must be 9–17, got {self.earliest_start_hour}"
            )
        if not (9 <= self.latest_start_hour <= 17):
            raise ValueError(
                f"latest_start_hour must be 9–17, got {self.latest_start_hour}"
            )
        if self.earliest_start_hour > self.latest_start_hour:
            raise ValueError(
                f"earliest_start_hour ({self.earliest_start_hour}) "
                f"cannot exceed latest_start_hour ({self.latest_start_hour})"
            )
        if not (1 <= self.priority <= 5):
            raise ValueError(f"priority must be 1–5, got {self.priority}")

    @property
    def duration_hours(self) -> float:
        """Duration expressed in fractional hours."""
        return self.duration_minutes / 60.0

    def __repr__(self) -> str:
        room = f", room={self.room_required!r}" if self.room_required else ""
        return (
            f"Event(id={self.id}, title={self.title!r}, "
            f"dur={self.duration_minutes}m, "
            f"window=[{self.earliest_start_hour}:00–{self.latest_start_hour}:00], "
            f"pri={self.priority}, attendees={self.attendees}{room})"
        )


@dataclass
class TimeSlot:
    """A discrete time slot on the calendar.

    Attributes:
        start_hour: Start time as fractional hour (e.g. 9.5 = 09:30).
        duration_minutes: Slot length in minutes.
        room: Room name assigned to this slot.
        is_available: Whether the slot is currently free.
    """

    start_hour: float
    duration_minutes: int
    room: str
    is_available: bool = True

    def __post_init__(self) -> None:
        if not (9.0 <= self.start_hour <= 17.0):
            raise ValueError(
                f"start_hour must be 9.0–17.0, got {self.start_hour}"
            )

    @property
    def end_hour(self) -> float:
        """Compute ending hour from start + duration."""
        return self.start_hour + self.duration_minutes / 60.0

    def overlaps(self, other: "TimeSlot") -> bool:
        """Return True if this slot overlaps with *other*."""
        return self.start_hour < other.end_hour and other.start_hour < self.end_hour

    def __repr__(self) -> str:
        status = "free" if self.is_available else "booked"
        return (
            f"TimeSlot(start={self.start_hour:.1f}, "
            f"dur={self.duration_minutes}m, "
            f"room={self.room!r}, {status})"
        )


@dataclass
class CalendarConstraint:
    """A scheduling constraint.

    Attributes:
        constraint_type: One of "no_overlap", "lunch_break",
                         "travel_time", "attendee_unavailable".
        event_id: Event this constraint applies to, if any.
        time_window: (start_hour, end_hour) tuple for time-based constraints.
        severity: "hard" (must satisfy) or "soft" (penalised if violated).
    """

    constraint_type: str
    event_id: Optional[int] = None
    time_window: Optional[Tuple[float, float]] = None
    severity: str = "hard"

    _VALID_TYPES = frozenset(
        {"no_overlap", "lunch_break", "travel_time", "attendee_unavailable"}
    )
    _VALID_SEVERITIES = frozenset({"hard", "soft"})

    def __post_init__(self) -> None:
        if self.constraint_type not in self._VALID_TYPES:
            raise ValueError(
                f"constraint_type must be one of {self._VALID_TYPES}, "
                f"got {self.constraint_type!r}"
            )
        if self.severity not in self._VALID_SEVERITIES:
            raise ValueError(
                f"severity must be 'hard' or 'soft', got {self.severity!r}"
            )

    def __repr__(self) -> str:
        parts = [f"CalendarConstraint(type={self.constraint_type!r}"]
        if self.event_id is not None:
            parts.append(f"event={self.event_id}")
        if self.time_window is not None:
            parts.append(f"window={self.time_window}")
        parts.append(f"severity={self.severity!r})")
        return ", ".join(parts)


@dataclass
class ScheduleState:
    """Snapshot of the current schedule quality.

    Attributes:
        scheduled_events: List of events placed on the calendar so far.
        conflicts: Pairs of conflicting event ids.
        total_conflicts: Count of detected conflicts.
        utilization_score: Fraction of working hours used (0.0–1.0).
        constraint_violations: Number of constraint violations.
    """

    scheduled_events: List[Event] = field(default_factory=list)
    conflicts: List[Tuple[int, int]] = field(default_factory=list)
    total_conflicts: int = 0
    utilization_score: float = 0.0
    constraint_violations: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.utilization_score <= 1.0):
            raise ValueError(
                f"utilization_score must be 0.0–1.0, got {self.utilization_score}"
            )

    @property
    def is_conflict_free(self) -> bool:
        """True when the schedule has zero conflicts."""
        return self.total_conflicts == 0

    @property
    def quality_score(self) -> float:
        """Composite quality metric in [0, 1].

        Higher is better.  Penalises conflicts and violations while
        rewarding calendar utilization.
        """
        conflict_penalty = min(self.total_conflicts * 0.15, 1.0)
        violation_penalty = min(self.constraint_violations * 0.10, 1.0)
        return max(
            0.0,
            self.utilization_score - conflict_penalty - violation_penalty,
        )

    def __repr__(self) -> str:
        return (
            f"ScheduleState(events={len(self.scheduled_events)}, "
            f"conflicts={self.total_conflicts}, "
            f"utilization={self.utilization_score:.2f}, "
            f"violations={self.constraint_violations}, "
            f"quality={self.quality_score:.2f})"
        )

# model_boarding.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import numpy as np



class Method(Enum):
    RANDOM = auto()
    BACK_TO_FRONT = auto()
    FRONT_TO_BACK = auto()
    BLOCKS_K2 = auto()  # creative method (allowed in spec §4)

@dataclass(frozen=True)
class Plane:
    rows: int = 50
    cols: int = 6  # ABC | DEF

def simulate_boarding(method: Method, plane: Plane = Plane(), seed: int | None = None) -> float:
    """
    Returns total boarding time in MINUTES, following the given spec exactly.
    """
    rng = np.random.default_rng(seed)

    seats = _all_seats(plane)
    order = _boarding_order(method, seats, plane, rng)

    # --- Timing per spec (all in minutes) ---
    MEAN_LUGGAGE = 0.5  # Exp
    BASE_SEATING = 0.0  # no delay if no blockers
    # if blocked: Exp(mean = 0.5 + 0.25 * #blockers)

    # Aisle blocking logic (rules 4–5):
    # If next passenger row >= previous row => must wait until previous finishes.
    # If next passenger row < previous row   => can start in parallel (same start time).
    start_prev = 0.0
    finish_prev = 0.0

    # Seats already occupied (at the moment we start a passenger) – used to count blockers in-row.
    seated_now: set[str] = set()

    # To respect “occupied only after seating finished”, we queue updates of 'seated_now'
    # at their finish times via a small event list processed in time order.
    # Here a tiny list suffices because dependencies are simple (prev or same-start chains).
    future_events: list[tuple[float, str]] = []  # (finish_time, seat)

    def _flush_events_until(t: float):
        # Mark as seated everyone who already finished by time t
        nonlocal future_events
        if not future_events:
            return
        future_events.sort()
        i = 0
        while i < len(future_events) and future_events[i][0] <= t + 1e-12:
            _, seat_id = future_events[i]
            seated_now.add(seat_id)
            i += 1
        if i:
            future_events = future_events[i:]

    for idx, seat in enumerate(order):
        row_i = int(seat[:2])

        # choose start time by rule 4–5
        if idx == 0:
            start_i = 0.0
        else:
            start_i = finish_prev if row_i >= int(order[idx - 1][:2]) else start_prev

        # before computing blockers, update who has already finished by this start time
        _flush_events_until(start_i)

        # count blockers in the same row that are already seated
        blockers = _blocking_count(seat, seated_now)

        # service time components
        t_luggage = rng.exponential(MEAN_LUGGAGE)
        if blockers == 0:
            t_seating = BASE_SEATING
        else:
            t_seating = rng.exponential(0.5 + 0.25 * blockers)

        service = t_luggage + t_seating  # walk time is zero by spec
        finish_i = start_i + service

        # schedule the moment this seat becomes "occupied"
        future_events.append((finish_i, seat))

        # update recurrence trackers
        start_prev = start_i
        finish_prev = finish_i

    # flush remaining events (not really required, but keeps 'seated_now' consistent)
    _flush_events_until(finish_prev + 1e-9)

    return finish_prev  # already in minutes

# ---------- Internals ----------

def _all_seats(plane: Plane) -> List[str]:
    letters = [chr(ord("A") + i) for i in range(plane.cols)]
    return [f"{r:02d}{c}" for r in range(1, plane.rows + 1) for c in letters]

def _boarding_order(method: Method, seats: List[str], plane: Plane, rng: np.random.Generator) -> List[str]:
    letters = [chr(ord("A") + i) for i in range(plane.cols)]
    rows = list(range(1, plane.rows + 1))

    if method is Method.RANDOM:
        order = seats[:]
        rng.shuffle(order)
        return order

    if method is Method.FRONT_TO_BACK:
        # rows ascending; seat order WITHIN ROW is random (as required)
        order: List[str] = []
        for r in rows:
            shuffled = letters[:]
            rng.shuffle(shuffled)
            order.extend([f"{r:02d}{c}" for c in shuffled])
        return order

    if method is Method.BACK_TO_FRONT:
        # rows descending; seat order WITHIN ROW is random (as required)
        order = []
        for r in reversed(rows):
            shuffled = letters[:]
            rng.shuffle(shuffled)
            order.extend([f"{r:02d}{c}" for c in shuffled])
        return order

    if method is Method.BLOCKS_K2:
        # Allowed creativity in §4:
        # Windows -> Middle -> Aisle; within each sub-group board in 2-row blocks back->front.
        windows, middle, aisle = ['A', 'F'], ['B', 'E'], ['C', 'D']
        order: List[str] = []

        def two_row_blocks_back_to_front(rs: List[int]) -> List[int]:
            rb = list(reversed(rs))
            out: List[int] = []
            for i in range(0, len(rb), 2):
                out.extend(rb[i:i+2])  # blocks of 2 rows
            return out

        for group in (windows, middle, aisle):
            for r in two_row_blocks_back_to_front(rows):
                g = group[:]  # tiny shuffle inside the pair keeps realism
                rng.shuffle(g)
                order.extend([f"{r:02d}{c}" for c in g])
        return order

    raise ValueError(f"Unsupported method: {method}")

def _blocking_count(seat: str, seated_now: set[str]) -> int:
    row = seat[:2]
    col = seat[2]
    seated_cols = {s[2] for s in seated_now if s.startswith(row)}
    if col == 'A':  # left window blocked by B,C
        return int('B' in seated_cols) + int('C' in seated_cols)
    if col == 'B':  # middle blocked by C
        return int('C' in seated_cols)
    if col == 'C':  # aisle
        return 0
    if col == 'D':  # aisle
        return 0
    if col == 'E':  # middle blocked by D
        return int('D' in seated_cols)
    if col == 'F':  # window blocked by D,E
        return int('D' in seated_cols) + int('E' in seated_cols)
    return 0

"""Parsing of Extended OpenTTGames event labels into structured events.

Label strings look like:
    "bounce"                                              ball bounce on the table
    "net"                                                 ball crossing the net (NOT a net fault)
    "empty_event"                                         negative sample from original dataset
    "left_forehand_loop neutral both_feet_planted"        stroke: side_hand_technique lean feet
    "right_out"                                           rally ending, prefixed by the player who CAUSED it
"""
from __future__ import annotations

from dataclasses import dataclass

SIDES = ("left", "right")
HANDS = ("forehand", "backhand")
TECHNIQUES = ("serve", "loop", "block", "push", "flick", "lob", "chop", "smash")
LEANS = ("neutral", "back_heavy", "front_heavy", "right_leaning", "left_leaning", "unknown")
FEET = ("both_feet_planted", "left_foot_lifted", "right_foot_lifted", "both_feet_lifted", "unknown")
OUTCOMES = ("net", "not_hitting_ball", "winner", "double_bounce", "out", "miss_on_own_side")
# Outcomes where the player who caused the ending WINS the point; all others lose it.
WINNING_OUTCOMES = ("winner", "double_bounce")


@dataclass(frozen=True)
class Event:
    frame: int
    kind: str  # "stroke" | "bounce" | "net_cross" | "empty" | "rally_end"
    side: str | None = None
    hand: str | None = None
    technique: str | None = None
    lean: str | None = None
    feet: str | None = None
    outcome: str | None = None
    raw: str = ""

    @property
    def point_winner(self) -> str | None:
        if self.kind != "rally_end":
            return None
        if self.outcome in WINNING_OUTCOMES:
            return self.side
        return "left" if self.side == "right" else "right"


def _match_prefix(token: str, vocab: tuple[str, ...]) -> str:
    """Tolerate annotation typos like 'back_heavyn' by prefix matching."""
    if token in vocab:
        return token
    for v in vocab:
        if token.startswith(v) or v.startswith(token):
            return v
    return "unknown"


def parse_label(frame: int, raw: str) -> Event | None:
    parts = raw.strip().split()
    if not parts:
        return None
    head = parts[0].lstrip("x")  # one label is "xright_backhand_chop"

    if head == "bounce":
        return Event(frame, "bounce", raw=raw)
    if head == "net":
        return Event(frame, "net_cross", raw=raw)
    if head == "empty_event":
        return Event(frame, "empty", raw=raw)

    side, _, rest = head.partition("_")
    if side not in SIDES:
        return None  # e.g. stray "point"
    if rest in OUTCOMES:
        return Event(frame, "rally_end", side=side, outcome=rest, raw=raw)

    hand, _, technique = rest.partition("_")
    if hand not in HANDS or technique not in TECHNIQUES:
        return None
    lean = _match_prefix(parts[1], LEANS) if len(parts) > 1 else "unknown"
    feet = _match_prefix(parts[2], FEET) if len(parts) > 2 else "unknown"
    return Event(frame, "stroke", side=side, hand=hand, technique=technique, lean=lean, feet=feet, raw=raw)

"""Compatibility wrapper for the Phase 1.12 market-structure stop engine."""

from risk.structure_stop_engine import (
    STOP_TYPES,
    StructureStopEngine,
    StructureStopResult,
    SupportZone,
    calculate_atr,
    detect_recent_swing_low,
    detect_support_zone,
    find_recent_swing_low,
)

__all__ = [
    "STOP_TYPES",
    "StructureStopEngine",
    "StructureStopResult",
    "SupportZone",
    "calculate_atr",
    "detect_recent_swing_low",
    "detect_support_zone",
    "find_recent_swing_low",
]


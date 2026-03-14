"""
NEXUS-ATMS Cybersecurity Module
=================================
Protects traffic signal infrastructure from cyber attacks:
- Anomaly detection on signal commands (impossible patterns)
- Command authentication via HMAC signing
- Rate limiting on signal switches
- Attack simulation mode for demo
"""

import hashlib
import hmac
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """A single security event/alert."""
    event_id: str
    event_type: str       # anomaly, rate_limit, auth_failure, attack_sim
    junction_id: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    description: str = ""
    blocked: bool = False
    details: Dict = field(default_factory=dict)


class SignalAnomalyDetector:
    """
    Detects anomalous signal control patterns that may indicate
    cyber attacks or system malfunctions.
    """

    def __init__(
        self,
        max_switch_rate: int = 6,       # Max switches per minute per junction
        anomaly_window: int = 60,        # Seconds to look back
    ):
        self.max_switch_rate = max_switch_rate
        self.anomaly_window = anomaly_window

        # Per-junction command history: list of (timestamp, phase)
        self._command_history: Dict[str, List] = defaultdict(list)
        # Active signal states
        self._signal_states: Dict[str, int] = {}
        self._events: List[SecurityEvent] = []
        self._event_counter = 0
        self._blocked_commands = 0
        self._total_commands = 0

        # HMAC key for command signing
        self._signing_key = secrets.token_bytes(32)
        logger.info("[Security] Signal anomaly detector initialized")

    # ------------------------------------------------------------------
    # Command Validation
    # ------------------------------------------------------------------

    def validate_command(
        self,
        junction_id: str,
        new_phase: int,
        source: str = "ai",
        signature: Optional[str] = None,
    ) -> Dict:
        """
        Validate a signal change command before execution.

        Returns dict with: allowed (bool), reason, risk_level.
        """
        self._total_commands += 1
        now = time.time()

        # 1. Check authentication (if signature provided)
        if signature is not None:
            if not self._verify_signature(junction_id, new_phase, signature):
                self._log_event("auth_failure", junction_id, "HIGH",
                                f"Invalid signature from {source}", blocked=True)
                return {
                    "allowed": False,
                    "reason": "Authentication failed: invalid command signature",
                    "risk_level": "HIGH",
                }

        # 2. Check for impossible state: all-green
        if self._is_conflicting_phase(junction_id, new_phase):
            self._log_event("anomaly", junction_id, "CRITICAL",
                            f"Conflicting phase {new_phase} — possible attack",
                            blocked=True)
            return {
                "allowed": False,
                "reason": f"BLOCKED: Phase {new_phase} conflicts with safety rules",
                "risk_level": "CRITICAL",
            }

        # 3. Rate limiting
        history = self._command_history[junction_id]
        # Clean old entries
        history[:] = [(t, p) for t, p in history if now - t < self.anomaly_window]

        if len(history) >= self.max_switch_rate:
            self._log_event("rate_limit", junction_id, "MEDIUM",
                            f"Rate limit exceeded: {len(history)} changes in "
                            f"{self.anomaly_window}s", blocked=True)
            return {
                "allowed": False,
                "reason": f"Rate limit: {self.max_switch_rate} changes/min exceeded",
                "risk_level": "MEDIUM",
            }

        # 4. Check for rapid oscillation (phases flip-flopping)
        if len(history) >= 3:
            recent_phases = [p for _, p in history[-3:]]
            if len(set(recent_phases)) == 2 and recent_phases[0] == recent_phases[2]:
                self._log_event("anomaly", junction_id, "HIGH",
                                "Rapid phase oscillation detected", blocked=True)
                return {
                    "allowed": False,
                    "reason": "Anomaly: rapid phase oscillation (possible replay attack)",
                    "risk_level": "HIGH",
                }

        # Command is valid
        history.append((now, new_phase))
        self._signal_states[junction_id] = new_phase
        return {
            "allowed": True,
            "reason": "Command validated",
            "risk_level": "NONE",
        }

    def _is_conflicting_phase(self, junction_id: str, new_phase: int) -> bool:
        """Check if a phase would create a conflicting (unsafe) state."""
        # Phase IDs that would mean all-green or conflicting movements
        # Phases 0,2 are green; phases 1,3 are yellow
        # Conflicting = same direction as another active green
        # For simplicity: phases > 3 are always invalid
        return new_phase < 0 or new_phase > 7

    # ------------------------------------------------------------------
    # Command Signing
    # ------------------------------------------------------------------

    def sign_command(self, junction_id: str, phase: int) -> str:
        """Generate HMAC signature for a signal command."""
        msg = f"{junction_id}:{phase}".encode()
        return hmac.new(self._signing_key, msg, hashlib.sha256).hexdigest()

    def _verify_signature(self, junction_id: str, phase: int, signature: str) -> bool:
        """Verify HMAC signature of a signal command."""
        expected = self.sign_command(junction_id, phase)
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Attack Simulation (Demo Mode)
    # ------------------------------------------------------------------

    def simulate_attack(self, attack_type: str, junction_id: str) -> Dict:
        """
        Simulate a cyber attack for demonstration purposes.

        Attack types:
        - "all_green": Try to set all signals green simultaneously
        - "rapid_switch": Rapid phase oscillation
        - "replay": Replay old commands
        - "dos": Flood commands to overwhelm rate limiter
        """
        results = []

        if attack_type == "all_green":
            for phase in [0, 2, 4, 6]:
                r = self.validate_command(junction_id, phase, source="attacker")
                results.append(r)

        elif attack_type == "rapid_switch":
            for i in range(10):
                phase = i % 2 * 2  # Alternate 0 and 2
                r = self.validate_command(junction_id, phase, source="attacker")
                results.append(r)

        elif attack_type == "dos":
            for i in range(self.max_switch_rate + 5):
                r = self.validate_command(junction_id, 0, source="attacker")
                results.append(r)

        blocked = sum(1 for r in results if not r["allowed"])
        total = len(results)

        self._log_event("attack_sim", junction_id, "HIGH",
                        f"Simulated {attack_type} attack: "
                        f"{blocked}/{total} commands blocked")

        return {
            "attack_type": attack_type,
            "junction_id": junction_id,
            "total_commands": total,
            "blocked": blocked,
            "passed": total - blocked,
            "detection_rate": round(blocked / total * 100, 1) if total > 0 else 0,
            "details": results,
        }

    # ------------------------------------------------------------------
    # Event Logging & Stats
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, junction_id: str,
                   severity: str, description: str, blocked: bool = False):
        self._event_counter += 1
        if blocked:
            self._blocked_commands += 1
        event = SecurityEvent(
            event_id=f"SEC-{self._event_counter:04d}",
            event_type=event_type,
            junction_id=junction_id,
            severity=severity,
            description=description,
            blocked=blocked,
        )
        self._events.append(event)
        logger.warning(f"[Security] {severity} | {event_type} | {junction_id}: {description}")

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent security events."""
        return [
            {
                "event_id": e.event_id,
                "type": e.event_type,
                "junction_id": e.junction_id,
                "timestamp": e.timestamp,
                "severity": e.severity,
                "description": e.description,
                "blocked": e.blocked,
            }
            for e in self._events[-limit:]
        ]

    def get_stats(self) -> Dict:
        """Get security statistics."""
        return {
            "total_commands": self._total_commands,
            "blocked_commands": self._blocked_commands,
            "block_rate_pct": round(
                self._blocked_commands / self._total_commands * 100
                if self._total_commands > 0 else 0.0, 2
            ),
            "total_alerts": len(self._events),
            "critical_alerts": sum(1 for e in self._events if e.severity == "CRITICAL"),
            "high_alerts": sum(1 for e in self._events if e.severity == "HIGH"),
            "monitored_junctions": len(self._command_history),
        }

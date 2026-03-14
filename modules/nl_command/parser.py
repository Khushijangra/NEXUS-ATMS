"""
NEXUS-ATMS Natural Language Command Parser
============================================
Parses plain-English commands from traffic authorities and converts
them into executable signal/system actions using spaCy NLP.

Examples:
  "Close all signals near Junction 7 for 30 minutes"
  "Clear corridor for ambulance from J1_0 to J3_3"
  "Enable school zone near J2_1"
  "Show prediction for Junction 5"
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# spaCy import
_SPACY_OK = False
try:
    import spacy
    _SPACY_OK = True
except ImportError:
    logger.warning("[NLCmd] spacy not installed. Using regex-only parser.")


@dataclass
class ParsedCommand:
    """Structured representation of a parsed NL command."""
    intent: str               # close_road, emergency, override, school_zone, etc.
    raw_text: str
    junctions: List[str]      # Extracted junction IDs
    duration_minutes: Optional[int] = None
    direction: Optional[str] = None       # NS, EW
    vehicle_type: Optional[str] = None    # ambulance, fire_truck
    phase: Optional[int] = None           # Signal phase to set
    confidence: float = 0.0
    parameters: Dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Intent patterns: keyword lists that map to intent types
INTENT_PATTERNS = {
    "close_road": ["close", "block", "restrict", "shut", "barricade"],
    "emergency": ["ambulance", "fire", "emergency", "clear corridor", "clear path",
                   "urgent", "paramedic"],
    "override_signal": ["override", "force", "set green", "set red", "hold",
                         "manual", "lock"],
    "school_zone": ["school", "children", "kids", "playground"],
    "event_mode": ["match", "concert", "stadium", "crowd", "rally", "festival",
                    "event"],
    "weather_mode": ["rain", "fog", "storm", "snow", "flooding", "heavy rain"],
    "vip_convoy": ["vip", "convoy", "minister", "president", "dignitary", "escort"],
    "prediction": ["predict", "forecast", "status", "show", "check", "what will"],
    "reset": ["reset", "restore", "normal", "default", "revert", "cancel"],
}


class NLCommandParser:
    """
    Parses natural language traffic management commands.
    Uses spaCy for entity extraction and pattern matching for intent classification.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        if _SPACY_OK:
            try:
                self._nlp = spacy.load(spacy_model)
                logger.info(f"[NLCmd] spaCy model '{spacy_model}' loaded")
            except OSError:
                logger.warning(f"[NLCmd] spaCy model '{spacy_model}' not found. "
                               "Using regex-only parser.")

        # Junction name patterns
        self._junction_pattern = re.compile(
            r'(?:junction|junct|j|J)\s*(\d+[_]\d+|\d+)',
            re.IGNORECASE,
        )
        self._duration_pattern = re.compile(
            r'(\d+)\s*(min|minute|minutes|hour|hours|hr|h)',
            re.IGNORECASE,
        )
        self._direction_pattern = re.compile(
            r'\b(north|south|east|west|N-S|E-W|NS|EW|N|S|E|W)\b',
            re.IGNORECASE,
        )

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse a natural language command into a structured ParsedCommand.
        """
        text_lower = text.lower().strip()

        # 1. Detect intent
        intent, confidence = self._detect_intent(text_lower)

        # 2. Extract junction IDs
        junctions = self._extract_junctions(text)

        # 3. Extract duration
        duration = self._extract_duration(text)

        # 4. Extract direction
        direction = self._extract_direction(text)

        # 5. Extract vehicle type
        vehicle_type = self._extract_vehicle_type(text_lower)

        # 6. Extract phase info
        phase = self._extract_phase(text_lower)

        # 7. spaCy entity extraction for richer parsing
        params = {}
        if self._nlp:
            doc = self._nlp(text)
            params["entities"] = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ]
            # Extract locations from spaCy
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):
                    params.setdefault("locations", []).append(ent.text)

        cmd = ParsedCommand(
            intent=intent,
            raw_text=text,
            junctions=junctions,
            duration_minutes=duration,
            direction=direction,
            vehicle_type=vehicle_type,
            phase=phase,
            confidence=confidence,
            parameters=params,
        )

        logger.info(f"[NLCmd] Parsed: intent={intent} conf={confidence:.2f} "
                     f"junctions={junctions} duration={duration}min")
        return cmd

    # ------------------------------------------------------------------
    # Intent Detection
    # ------------------------------------------------------------------

    def _detect_intent(self, text: str) -> Tuple[str, float]:
        """Detect command intent via keyword matching."""
        scores = {}
        for intent, keywords in INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[intent] = score / len(keywords)

        if not scores:
            return "unknown", 0.0

        best_intent = max(scores, key=scores.get)
        return best_intent, min(scores[best_intent] * 2, 1.0)

    # ------------------------------------------------------------------
    # Entity Extraction
    # ------------------------------------------------------------------

    def _extract_junctions(self, text: str) -> List[str]:
        """Extract junction IDs like J1_2 or Junction 7."""
        matches = self._junction_pattern.findall(text)
        result = []
        for m in matches:
            if "_" in m:
                result.append(f"J{m}")
            else:
                # Single number — could be any junction
                result.append(f"J{m}")
        return result

    def _extract_duration(self, text: str) -> Optional[int]:
        """Extract duration in minutes."""
        match = self._duration_pattern.search(text)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            if unit.startswith("h"):
                return value * 60
            return value
        return None

    def _extract_direction(self, text: str) -> Optional[str]:
        """Extract traffic direction."""
        match = self._direction_pattern.search(text)
        if match:
            d = match.group(1).upper()
            if d in ("N", "S", "NORTH", "SOUTH", "NS", "N-S"):
                return "NS"
            if d in ("E", "W", "EAST", "WEST", "EW", "E-W"):
                return "EW"
            return d
        return None

    def _extract_vehicle_type(self, text: str) -> Optional[str]:
        """Extract emergency vehicle type."""
        if "ambulance" in text:
            return "ambulance"
        if "fire" in text:
            return "fire_truck"
        if "police" in text:
            return "police"
        return None

    def _extract_phase(self, text: str) -> Optional[int]:
        """Extract signal phase number."""
        if "green" in text:
            return 0
        if "red" in text:
            return 1
        if "yellow" in text or "amber" in text:
            return 3
        return None

    # ------------------------------------------------------------------
    # Command Execution Helpers
    # ------------------------------------------------------------------

    def to_action(self, cmd: ParsedCommand) -> Dict:
        """
        Convert ParsedCommand to an executable action dict
        that the backend can process.
        """
        action = {
            "type": cmd.intent,
            "junctions": cmd.junctions,
            "parameters": cmd.parameters,
        }

        if cmd.intent == "emergency":
            action["vehicle_type"] = cmd.vehicle_type or "ambulance"
            if len(cmd.junctions) >= 2:
                action["origin"] = cmd.junctions[0]
                action["destination"] = cmd.junctions[-1]

        elif cmd.intent == "override_signal":
            action["phase"] = cmd.phase
            action["duration_minutes"] = cmd.duration_minutes or 5

        elif cmd.intent == "close_road":
            action["duration_minutes"] = cmd.duration_minutes or 30

        elif cmd.intent == "school_zone":
            action["active"] = True

        elif cmd.intent == "reset":
            action["reset_all"] = len(cmd.junctions) == 0

        return action

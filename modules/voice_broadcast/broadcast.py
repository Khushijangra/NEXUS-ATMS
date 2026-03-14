"""
NEXUS-ATMS Voice Broadcast System
====================================
Generates voice announcements for traffic incidents and emergencies
using Google TTS (gTTS) with multilingual support.
"""

import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_GTTS_OK = False
try:
    from gtts import gTTS
    _GTTS_OK = True
except ImportError:
    logger.warning("[Voice] gTTS not installed. Voice broadcast disabled.")

_PYGAME_OK = False
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    logger.warning("[Voice] pygame not installed. Cannot play audio.")


LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "marathi": "mr",
    "bengali": "bn",
    "gujarati": "gu",
    "malayalam": "ml",
    "punjabi": "pa",
}


class VoiceBroadcast:
    """
    Generates and plays multilingual voice announcements.
    """

    def __init__(
        self,
        language: str = "en",
        fallback_language: str = "hi",
        output_dir: str = "audio_cache",
    ):
        self.language = language
        self.fallback = fallback_language
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._audio_initialized = False
        if _PYGAME_OK:
            try:
                pygame.mixer.init()
                self._audio_initialized = True
                logger.info("[Voice] Audio system initialized")
            except Exception as e:
                logger.warning(f"[Voice] Could not init audio: {e}")

        self._broadcast_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Announcement Generation
    # ------------------------------------------------------------------

    def announce(
        self,
        message: str,
        language: Optional[str] = None,
        play: bool = True,
    ) -> Optional[str]:
        """
        Generate and optionally play a voice announcement.

        Returns the path to the generated audio file.
        """
        lang = language or self.language
        if lang in LANGUAGE_MAP:
            lang = LANGUAGE_MAP[lang]

        audio_path = os.path.join(
            self.output_dir,
            f"broadcast_{int(time.time())}_{lang}.mp3",
        )

        if not _GTTS_OK:
            logger.warning("[Voice] gTTS unavailable — announcement logged only")
            self._log_broadcast(message, lang, None)
            return None

        try:
            tts = gTTS(text=message, lang=lang, slow=False)
            tts.save(audio_path)
            logger.info(f"[Voice] Generated: {audio_path}")
        except Exception as e:
            logger.error(f"[Voice] TTS generation failed: {e}")
            self._log_broadcast(message, lang, None)
            return None

        self._log_broadcast(message, lang, audio_path)

        if play and self._audio_initialized:
            self._play_audio(audio_path)

        return audio_path

    def _play_audio(self, path: str):
        """Play an audio file using pygame."""
        if not self._audio_initialized:
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            # Don't block — audio plays in background
            logger.info(f"[Voice] Playing: {path}")
        except Exception as e:
            logger.error(f"[Voice] Playback failed: {e}")

    # ------------------------------------------------------------------
    # Pre-built Announcements
    # ------------------------------------------------------------------

    def announce_incident(self, incident_type: str, location: str, language: Optional[str] = None):
        """Generate an incident announcement."""
        templates = {
            "en": {
                "accident": f"Attention: Accident reported near {location}. Please use alternate routes. Emergency services are en route.",
                "breakdown": f"Attention: Vehicle breakdown reported at {location}. Expect delays. Please drive carefully.",
                "congestion": f"Traffic advisory: Heavy congestion near {location}. Consider alternate routes for faster travel.",
                "wrong_way": f"Warning: Wrong-way vehicle detected near {location}. Exercise extreme caution.",
                "road_damage": f"Advisory: Road damage reported at {location}. Maintenance crew has been dispatched.",
            },
            "hi": {
                "accident": f"ध्यान दें: {location} के पास दुर्घटना की सूचना। कृपया वैकल्पिक मार्ग अपनाएं।",
                "breakdown": f"ध्यान दें: {location} पर वाहन खराबी। देरी की संभावना है।",
                "congestion": f"यातायात सूचना: {location} के पास भारी भीड़। वैकल्पिक मार्ग अपनाएं।",
                "wrong_way": f"चेतावनी: {location} के पास गलत दिशा में वाहन। सावधानी बरतें।",
                "road_damage": f"सूचना: {location} पर सड़क क्षतिग्रस्त। मरम्मत दल भेज दिया गया है।",
            },
        }

        lang = language or self.language
        lang_key = lang if lang in templates else "en"
        msg = templates.get(lang_key, templates["en"]).get(
            incident_type, f"Alert: {incident_type} at {location}"
        )
        return self.announce(msg, language=lang)

    def announce_emergency_corridor(
        self,
        vehicle_type: str,
        origin: str,
        destination: str,
        language: Optional[str] = None,
    ):
        """Generate an emergency corridor announcement."""
        msg = (f"Attention: {vehicle_type} emergency corridor activated "
               f"from {origin} to {destination}. "
               f"Please yield and clear the route immediately.")
        return self.announce(msg, language=language)

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log_broadcast(self, message: str, lang: str, audio_path: Optional[str]):
        self._broadcast_log.append({
            "timestamp": time.time(),
            "message": message,
            "language": lang,
            "audio_path": audio_path,
        })

    def get_broadcast_log(self, limit: int = 20) -> List[Dict]:
        return self._broadcast_log[-limit:]

    def get_stats(self) -> Dict:
        return {
            "total_broadcasts": len(self._broadcast_log),
            "audio_system_active": self._audio_initialized,
            "tts_available": _GTTS_OK,
        }

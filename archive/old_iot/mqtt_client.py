"""
MQTT Client Wrapper
====================
Thin wrapper around paho-mqtt that:
  • Publishes SensorReadings as JSON to topic  traffic/{int_id}/{sensor_type}
  • Subscribes to control topics            traffic/{int_id}/signal_cmd
  • Falls back to a no-op in-process bus when paho-mqtt is not installed

Topic schema (MQTT)
--------------------
  Publish  : traffic/<intersection_id>/sensors/<sensor_type>
  Subscribe: traffic/<intersection_id>/control

In-process fallback
--------------------
  Call register_handler(topic_prefix, callback) to receive messages in-process.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_PAHO_AVAILABLE = False
try:
    import paho.mqtt.client as paho  # type: ignore
    _PAHO_AVAILABLE = True
except ImportError:
    pass


class _InProcessBus:
    """Minimal pub/sub bus used when paho is not installed."""

    def __init__(self) -> None:
        self._handlers: Dict[str, list] = {}

    def publish(self, topic: str, payload: str) -> None:
        for prefix, cb in [
            (p, c)
            for p, handlers in self._handlers.items()
            for c in handlers
            if topic.startswith(p)
        ]:
            try:
                cb(topic, payload)
            except Exception as exc:
                logger.warning(f"[Bus] Handler error: {exc}")

    def subscribe(self, topic_prefix: str, callback: Callable) -> None:
        self._handlers.setdefault(topic_prefix, []).append(callback)


class MQTTClient:
    """
    Publish / subscribe interface for IoT sensor data.

    Parameters
    ----------
    broker_host : str   Broker hostname (default localhost).
    broker_port : int   Broker port (default 1883).
    client_id   : str   MQTT client ID.
    use_tls     : bool  Enable TLS (requires CA cert path).
    ca_cert     : str   Path to CA certificate for TLS.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "traffic_ai_system",
        use_tls: bool = False,
        ca_cert: Optional[str] = None,
    ) -> None:
        self.host = broker_host
        self.port = broker_port
        self._connected = False
        self._bus = _InProcessBus()

        if _PAHO_AVAILABLE:
            self._client = paho.Client(client_id=client_id,
                                       protocol=paho.MQTTv5)
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            if use_tls and ca_cert:
                self._client.tls_set(ca_certs=ca_cert)
            self._connect()
        else:
            logger.info(
                "[MQTT] paho-mqtt not installed — using in-process bus.\n"
                "  Install: pip install paho-mqtt"
            )
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_reading(self, reading: Any) -> None:
        """Serialise and publish a SensorReading."""
        topic = (
            f"traffic/{reading.intersection_id}"
            f"/sensors/{reading.sensor_type.value}"
        )
        payload = json.dumps(reading.to_dict())
        self._publish(topic, payload)

    def subscribe_control(
        self,
        intersection_id: str,
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to signal control commands for an intersection."""
        topic = f"traffic/{intersection_id}/control"
        if self._client:
            self._client.subscribe(topic)
            self._bus.subscribe(topic, lambda t, p: callback(json.loads(p)))
        else:
            self._bus.subscribe(topic, lambda t, p: callback(json.loads(p)))

    def send_signal_command(
        self, intersection_id: str, command: dict
    ) -> None:
        """Send a signal phase command to a physical controller."""
        topic = f"traffic/{intersection_id}/control"
        self._publish(topic, json.dumps(command))

    def disconnect(self) -> None:
        if self._client and self._connected:
            self._client.disconnect()
            self._connected = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            self._client.connect_async(self.host, self.port, keepalive=60)
            self._client.loop_start()
            logger.info(f"[MQTT] Connecting to {self.host}:{self.port} …")
        except Exception as exc:
            logger.warning(
                f"[MQTT] Could not connect to broker ({exc}). "
                "Falling back to in-process bus."
            )
            self._client = None

    def _publish(self, topic: str, payload: str) -> None:
        if self._client and self._connected:
            self._client.publish(topic, payload, qos=1)
        else:
            self._bus.publish(topic, payload)

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            self._connected = True
            logger.info("[MQTT] Connected to broker.")
        else:
            logger.warning(f"[MQTT] Connection refused (rc={rc}).")

    def _on_message(self, client, userdata, msg) -> None:
        self._bus.publish(
            msg.topic, msg.payload.decode("utf-8", errors="replace")
        )

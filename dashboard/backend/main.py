"""
NEXUS-ATMS — FastAPI Backend
===============================
Integrates all NEXUS modules into a unified REST + WebSocket API.

Endpoints:
  /api/status              — System status
  /api/snapshot            — Single traffic snapshot
  /api/history             — Historic traffic data
  /api/intersections       — All junction states
  /api/signal/override     — Manual signal override (POST)
  /api/emergency/activate  — Activate emergency corridor (POST)
  /api/emergency/active    — List active emergency events
  /api/carbon/today        — Carbon savings for today
  /api/carbon/certificate  — Download PDF certificate
  /api/pedestrian/analyze  — Pedestrian safety status
  /api/security/validate   — Validate a signal command (POST)
  /api/security/simulate   — Simulate an attack (POST)
  /api/security/events     — Recent security events
  /api/maintenance/orders  — Road maintenance work orders
  /api/nl/command          — Natural-language command (POST)
  /api/counterfactual      — AI vs baseline comparison
  /api/voice/announce      — Trigger voice broadcast (POST)
  /api/voice/log           — Broadcast log
  /api/metrics/overview    — Aggregated metrics overview
  /ws/live                 — WebSocket: live data stream (~1 Hz)
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from dashboard.demo_data import DemoDataGenerator

# Module imports — guarded so backend still starts if a module fails
_import_errors: Dict[str, str] = {}


def _safe_import(module_path: str, class_name: str):
    """Import a class, returning None on failure."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except Exception as exc:
        _import_errors[module_path] = str(exc)
        return None


EmergencyCorridorEngine = _safe_import("modules.emergency.corridor", "EmergencyCorridorEngine")
CarbonCreditEngine = _safe_import("modules.carbon.engine", "CarbonCreditEngine")
PedestrianSafetyAI = _safe_import("modules.pedestrian_safety.safety", "PedestrianSafetyAI")
SignalAnomalyDetector = _safe_import("modules.cybersecurity.signal_security", "SignalAnomalyDetector")
RoadMaintenanceAI = _safe_import("modules.road_maintenance.maintenance", "RoadMaintenanceAI")
NLCommandParser = _safe_import("modules.nl_command.parser", "NLCommandParser")
CounterfactualEngine = _safe_import("modules.counterfactual.engine", "CounterfactualEngine")
VoiceBroadcast = _safe_import("modules.voice_broadcast.broadcast", "VoiceBroadcast")

# Real data pipeline imports (optional; backend still runs if unavailable)
VehicleDetector = _safe_import("vision.detector", "VehicleDetector")
VehicleTracker = _safe_import("vision.tracker", "VehicleTracker")
ZoneCounter = _safe_import("vision.counter", "ZoneCounter")
SpeedEstimator = _safe_import("vision.speed_estimator", "SpeedEstimator")
IncidentDetector = _safe_import("vision.incident_detector", "IncidentDetector")

SensorSimulator = _safe_import("iot.sensor_simulator", "SensorSimulator")
SensorFusion = _safe_import("iot.data_fusion", "SensorFusion")
MQTTClient = _safe_import("iot.mqtt_client", "MQTTClient")

LSTMPredictor = _safe_import("prediction.lstm_predictor", "LSTMPredictor")
MLAnomalyDetector = _safe_import("prediction.ml_anomaly_detector", "MLAnomalyDetector")

RLController = _safe_import("control.rl_controller", "RLController")

logger = logging.getLogger("nexus-backend")

# ---------------------------------------------------------------
# App Initialisation
# ---------------------------------------------------------------
app = FastAPI(title="NEXUS-ATMS Dashboard", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Demo mode flag
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
LIVE_MODE = os.getenv("LIVE_MODE", "true").lower() == "true"
demo_gen = DemoDataGenerator(mode="rl") if DEMO_MODE else None

# ---------------------------------------------------------------
# Module Singletons
# ---------------------------------------------------------------
emergency_engine = EmergencyCorridorEngine() if EmergencyCorridorEngine else None
carbon_engine = CarbonCreditEngine() if CarbonCreditEngine else None
pedestrian_ai = PedestrianSafetyAI(device="cpu") if PedestrianSafetyAI else None  # CPU for backend
security_detector = SignalAnomalyDetector() if SignalAnomalyDetector else None
maintenance_ai = RoadMaintenanceAI() if RoadMaintenanceAI else None
nl_parser = NLCommandParser() if NLCommandParser else None
counterfactual = CounterfactualEngine() if CounterfactualEngine else None
voice = VoiceBroadcast(language="en") if VoiceBroadcast else None

# Build the graph for emergency routing
if emergency_engine:
    emergency_engine.build_grid_graph(rows=4, cols=4)

# ---------------------------------------------------------------
# Pydantic Models for POST Bodies
# ---------------------------------------------------------------
class SignalOverrideRequest(BaseModel):
    junction_id: str
    phase: str  # "NS_GREEN", "EW_GREEN", "YELLOW", "ALL_RED"
    duration: int = 60
    source: str = "operator"


class EmergencyActivateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "ambulance"
    origin: str  # Junction ID
    destination: str  # Junction ID


class SecurityValidateRequest(BaseModel):
    junction_id: str
    new_phase: int
    source: str = "ai"


class SecuritySimulateRequest(BaseModel):
    attack_type: str  # replay, dos, mitm, conflicting
    junction_id: str = "J1_1"


class NLCommandRequest(BaseModel):
    text: str


class VoiceAnnounceRequest(BaseModel):
    message: str
    language: str = "en"
    play: bool = False  # Don't play audio on server by default


# ---------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket):
        self._active.discard(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def count(self) -> int:
        return len(self._active)


ws_manager = ConnectionManager()

# ---------------------------------------------------------------
# Junction state cache (updated by background loop / SUMO)
# ---------------------------------------------------------------
_junction_states: Dict[str, Dict] = {}
_signal_overrides: Dict[str, Dict] = {}


class LiveRuntime:
    """Connect vision, IoT, prediction, anomaly, and RL into one live tick."""

    def __init__(self) -> None:
        self.enabled = LIVE_MODE
        self.intersection_id = os.getenv("PRIMARY_INTERSECTION", "J1_1")
        self.video_source = os.getenv("TRAFFIC_VIDEO_SOURCE", "0")
        self.vision_backend = os.getenv("VISION_BACKEND", "yolo")
        self.vision_device = os.getenv("VISION_DEVICE", "cpu")
        self._lock = asyncio.Lock()
        self._tick = 0

        self.cap = None
        self.detector = None
        self.tracker = None
        self.counter = None
        self.speed = None
        self.incident = None

        self.sensor_sim = None
        self.sensor_fusion = None
        self.mqtt = None

        self.lstm = None
        self.ml_anomaly = None
        self.rl = None

        self.latest_traffic: Dict = {}
        self.latest_incidents: List[Dict] = []
        self.latest_anomaly: Optional[Dict] = None
        self.latest_prediction: Optional[List[float]] = None
        self.last_error: str = ""
        self.frame_ok = False

        if not self.enabled:
            return

        # Vision stack
        if VehicleDetector and VehicleTracker and ZoneCounter and SpeedEstimator and IncidentDetector:
            self.detector = VehicleDetector(
                backend=self.vision_backend,
                conf_threshold=float(os.getenv("VISION_CONF", "0.45")),
                device=self.vision_device,
            )
            self.tracker = VehicleTracker()
            self.counter = ZoneCounter(frame_shape=(720, 1280))
            self.speed = SpeedEstimator(fps=float(os.getenv("VISION_FPS", "25")))
            self.incident = IncidentDetector()

            try:
                import cv2  # type: ignore

                source = int(self.video_source) if self.video_source.isdigit() else self.video_source
                self.cap = cv2.VideoCapture(source)
                self.frame_ok = bool(self.cap and self.cap.isOpened())
                if not self.frame_ok:
                    logger.warning("[LiveRuntime] Video source unavailable: %s", self.video_source)
            except Exception as exc:
                self.last_error = f"video init: {exc}"

        # IoT stack
        if SensorSimulator and SensorFusion:
            self.sensor_fusion = SensorFusion()
            self.sensor_sim = SensorSimulator(
                intersection_ids=[self.intersection_id],
                real_time=False,
                sim_step_s=float(os.getenv("IOT_STEP_SECONDS", "5")),
            )

        if MQTTClient:
            self.mqtt = MQTTClient(
                broker_host=os.getenv("MQTT_HOST", "localhost"),
                broker_port=int(os.getenv("MQTT_PORT", "1883")),
            )

        # Prediction + anomaly
        if LSTMPredictor:
            try:
                self.lstm = LSTMPredictor(
                    device=os.getenv("LSTM_DEVICE", "cpu"),
                    model_path=os.getenv("LSTM_MODEL_PATH", "models/lstm_live.pt"),
                )
            except Exception as exc:
                logger.warning("[LiveRuntime] LSTM init failed (%s). Using no-predict mode.", exc)
                self.lstm = None
        if MLAnomalyDetector:
            self.ml_anomaly = MLAnomalyDetector(device=os.getenv("ANOMALY_DEVICE", "cpu"))
            try:
                self.ml_anomaly.load("models/ml_anomaly")
            except Exception:
                logger.info("[LiveRuntime] ML anomaly models not loaded yet.")

        # RL controller for live phase recommendation
        if RLController:
            self.rl = RLController(
                intersection_id=self.intersection_id,
                device=os.getenv("RL_DEVICE", "cpu"),
            )

    async def tick(self) -> Dict:
        async with self._lock:
            self._tick += 1

            snapshot = None
            if self.sensor_sim and self.sensor_fusion:
                readings = self.sensor_sim.tick()
                if self.mqtt:
                    for rd in readings:
                        self.mqtt.publish_reading(rd)
                fused = self.sensor_fusion.ingest(readings)
                snapshot = fused.get(self.intersection_id)

            # Vision updates (if a real frame is available)
            if self.cap and self.detector and self.tracker and self.counter and self.speed and self.incident:
                ok, frame = self.cap.read()
                self.frame_ok = bool(ok)
                if ok and frame is not None:
                    det = self.detector.detect(frame)
                    tracked = self.tracker.update(det)
                    zone_stats = self.counter.update(tracked)
                    track_speeds = self.speed.estimate(self.tracker.active_tracks())
                    incidents = self.incident.update(
                        {k: v["queue"] for k, v in zone_stats.items()},
                        track_speeds,
                    )
                    self.latest_incidents = [
                        {
                            "id": i.incident_id,
                            "type": i.incident_type,
                            "severity": i.severity,
                            "zone": i.zone,
                            "description": i.description,
                        }
                        for i in incidents
                    ]

                    if self.sensor_fusion:
                        avg_speed = float(np.mean(list(track_speeds.values()))) if track_speeds else 0.0
                        for ap in ("north", "south", "east", "west"):
                            q = int(zone_stats.get(ap, {}).get("queue", 0))
                            self.sensor_fusion.ingest_vision(self.intersection_id, ap, q, avg_speed)
                        snapshot = self.sensor_fusion.snapshot(self.intersection_id)

            traffic = self._to_traffic_snapshot(snapshot)
            self.latest_traffic = traffic

            # LSTM online prediction
            if self.lstm and snapshot is not None:
                try:
                    self.lstm.add_observation(snapshot)
                    pred = self.lstm.predict()
                    if pred is not None:
                        self.latest_prediction = pred[0].astype(float).tolist()
                except Exception as exc:
                    self.last_error = f"lstm: {exc}"

            # ML anomaly from fused feature vector
            if self.ml_anomaly and snapshot is not None:
                try:
                    vec = self._to_anomaly_vector(snapshot)
                    self.ml_anomaly.add_observation(vec)
                    if not getattr(self.ml_anomaly, "_fitted", False) and self._tick % 120 == 0:
                        self.ml_anomaly.fit(ae_epochs=6)
                    alert = self.ml_anomaly.detect(vec) if getattr(self.ml_anomaly, "_fitted", False) else None
                    if alert:
                        self.latest_anomaly = {
                            "severity": alert.severity,
                            "score": float(alert.anomaly_score),
                            "detectors": alert.detectors_fired,
                            "message": alert.message,
                        }
                except Exception as exc:
                    self.last_error = f"anomaly: {exc}"

            # RL decision from current observation
            if self.rl and snapshot is not None:
                try:
                    obs = snapshot.to_feature_vector()
                    action = self.rl.predict(obs)
                    phase = self._action_to_phase(action)
                    if self.intersection_id in _junction_states:
                        _junction_states[self.intersection_id]["phase"] = phase
                        _junction_states[self.intersection_id]["ai_confidence"] = 0.92
                except Exception as exc:
                    self.last_error = f"rl: {exc}"

            return traffic

    def status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "frame_ok": self.frame_ok,
            "video_source": self.video_source,
            "vision_backend": self.vision_backend,
            "intersection_id": self.intersection_id,
            "has_traffic": bool(self.latest_traffic),
            "has_prediction": self.latest_prediction is not None,
            "has_anomaly": self.latest_anomaly is not None,
            "last_error": self.last_error,
        }

    def _to_traffic_snapshot(self, snap) -> Dict:
        if snap is None:
            return {
                "tick": self._tick,
                "hour": time.localtime().tm_hour,
                "mode": "live",
                "queues": {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0},
                "waiting_times": {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0},
                "total_queue": 0.0,
                "avg_waiting_time": 0.0,
                "throughput": 0.0,
                "phase": _junction_states.get(self.intersection_id, {}).get("phase", "NS_GREEN"),
                "time_factor": 1.0,
            }

        queues = {}
        waits = {}
        throughput = 0.0
        for ap in ("north", "south", "east", "west"):
            st = snap.approaches.get(ap)
            q = float(st.queue_length if st else 0.0)
            queues[ap] = q
            waits[ap] = float(q * 3.0)
            throughput += float(st.flow_veh_h if st else 0.0)

        return {
            "tick": self._tick,
            "hour": time.localtime().tm_hour,
            "mode": "live",
            "queues": queues,
            "waiting_times": waits,
            "total_queue": float(sum(queues.values())),
            "avg_waiting_time": float(sum(waits.values()) / 4.0),
            "throughput": float(throughput),
            "phase": _junction_states.get(self.intersection_id, {}).get("phase", "NS_GREEN"),
            "time_factor": 1.0,
            "rainfall": float(snap.rainfall_mm_h),
            "visibility_m": float(snap.visibility_m),
            "aqi": float(snap.aqi),
        }

    @staticmethod
    def _action_to_phase(action: int) -> str:
        if action in (0, 1):
            return "NS_GREEN" if action == 0 else "EW_GREEN"
        if action == 2:
            return "YELLOW"
        return "ALL_RED"

    @staticmethod
    def _to_anomaly_vector(snap) -> np.ndarray:
        def _apv(ap: str, key: str, default: float = 0.0) -> float:
            st = snap.approaches.get(ap)
            return float(getattr(st, key, default) if st else default)

        hour = time.localtime().tm_hour + time.localtime().tm_min / 60.0
        return np.array([
            _apv("north", "vehicle_count"), _apv("south", "vehicle_count"),
            _apv("east", "vehicle_count"), _apv("west", "vehicle_count"),
            _apv("north", "speed_kmh"), _apv("south", "speed_kmh"),
            _apv("east", "speed_kmh"), _apv("west", "speed_kmh"),
            _apv("north", "queue_length"), _apv("south", "queue_length"),
            _apv("east", "queue_length"), _apv("west", "queue_length"),
            float(np.mean([_apv("north", "occupancy_pct"), _apv("south", "occupancy_pct"), _apv("east", "occupancy_pct"), _apv("west", "occupancy_pct") ])),
            float(np.sin(2 * np.pi * hour / 24.0)),
            float(np.cos(2 * np.pi * hour / 24.0)),
        ], dtype=np.float32)


live_runtime = LiveRuntime()


def _init_junctions():
    """Initialize 4x4 grid junction states."""
    for r in range(4):
        for c in range(4):
            jid = f"J{r}_{c}"
            _junction_states[jid] = {
                "junction_id": jid,
                "phase": "NS_GREEN",
                "queue_n": 0, "queue_s": 0, "queue_e": 0, "queue_w": 0,
                "wait_time": 0.0,
                "ai_confidence": 0.95,
                "is_corridor": False,
                "is_overridden": False,
            }


_init_junctions()


# ---------------------------------------------------------------
# REST Endpoints — Core
# ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>NEXUS-ATMS Dashboard</h1><p>Frontend not found.</p>")


@app.get("/api/status")
async def status():
    modules_status = {}
    module_map = {
        "emergency": emergency_engine,
        "carbon": carbon_engine,
        "pedestrian_safety": pedestrian_ai,
        "cybersecurity": security_detector,
        "road_maintenance": maintenance_ai,
        "nl_command": nl_parser,
        "counterfactual": counterfactual,
        "voice_broadcast": voice,
    }
    for name, instance in module_map.items():
        modules_status[name] = "active" if instance else f"failed: {_import_errors.get(f'modules.{name}', 'unknown')}"

    return {
        "status": "running",
        "version": "2.0.0",
        "demo_mode": DEMO_MODE,
        "live_mode": LIVE_MODE,
        "ws_clients": ws_manager.count,
        "junctions": len(_junction_states),
        "modules": modules_status,
        "runtime": live_runtime.status(),
    }


@app.get("/api/snapshot")
async def snapshot():
    if live_runtime.enabled:
        snap = await live_runtime.tick()
        if counterfactual:
            queues = {
                "N": int(snap["queues"].get("north", 0)),
                "S": int(snap["queues"].get("south", 0)),
                "E": int(snap["queues"].get("east", 0)),
                "W": int(snap["queues"].get("west", 0)),
            }
            counterfactual.record_comparison(
                ai_avg_wait=snap["avg_waiting_time"],
                ai_total_queue=int(snap["total_queue"]),
                ai_throughput=int(snap["throughput"]),
                queue_lengths=queues,
            )
        if carbon_engine:
            idle_ai = snap["avg_waiting_time"] / 60.0
            idle_baseline = idle_ai * 1.35
            carbon_engine.record_snapshot(idle_ai, idle_baseline, max(1, int(snap["total_queue"])))
        return snap

    if demo_gen:
        snap = demo_gen.get_snapshot()
        # Feed counterfactual engine
        if counterfactual:
            queues = {
                "N": int(snap["queues"].get("north", 0)),
                "S": int(snap["queues"].get("south", 0)),
                "E": int(snap["queues"].get("east", 0)),
                "W": int(snap["queues"].get("west", 0)),
            }
            counterfactual.record_comparison(
                ai_avg_wait=snap["avg_waiting_time"],
                ai_total_queue=int(snap["total_queue"]),
                ai_throughput=int(snap["throughput"]),
                queue_lengths=queues,
            )
        # Feed carbon engine
        if carbon_engine:
            idle_ai = snap["avg_waiting_time"] / 60.0
            idle_baseline = idle_ai * 1.4
            carbon_engine.record_snapshot(idle_ai, idle_baseline, 100)
        return snap
    return {"error": "No data source available. Enable DEMO_MODE or connect SUMO."}


@app.get("/api/history")
async def history(n: int = Query(100, ge=1, le=1000)):
    if live_runtime.enabled and live_runtime.latest_traffic:
        return [live_runtime.latest_traffic for _ in range(min(n, 10))]
    if demo_gen:
        gen = DemoDataGenerator(mode="rl")
        return gen.get_history(n)
    return []


@app.get("/api/intersections")
async def intersections():
    return list(_junction_states.values())


# ---------------------------------------------------------------
# REST Endpoints — Signal Override
# ---------------------------------------------------------------
@app.post("/api/signal/override")
async def signal_override(req: SignalOverrideRequest):
    jid = req.junction_id
    if jid not in _junction_states:
        return JSONResponse({"error": f"Unknown junction {jid}"}, status_code=404)

    # Security check
    if security_detector:
        result = security_detector.validate_command(jid, hash(req.phase) % 8, source=req.source)
        if not result.get("allowed"):
            return JSONResponse({"error": "Security blocked", "details": result}, status_code=403)

    _junction_states[jid]["phase"] = req.phase
    _junction_states[jid]["is_overridden"] = True
    _signal_overrides[jid] = {"phase": req.phase, "expires": time.time() + req.duration}

    return {"status": "ok", "junction": jid, "phase": req.phase, "duration": req.duration}


# ---------------------------------------------------------------
# REST Endpoints — Emergency
# ---------------------------------------------------------------
@app.post("/api/emergency/activate")
async def emergency_activate(req: EmergencyActivateRequest):
    if not emergency_engine:
        return JSONResponse({"error": "Emergency module not available"}, status_code=503)

    event = emergency_engine.activate_corridor(
        vehicle_id=req.vehicle_id,
        vehicle_type=req.vehicle_type,
        origin=req.origin,
        destination=req.destination,
    )
    if not event:
        return JSONResponse({"error": "Could not compute corridor path"}, status_code=400)

    # Mark junctions as corridor
    for jid in event.path:
        if jid in _junction_states:
            _junction_states[jid]["is_corridor"] = True

    # Voice announcement
    if voice:
        voice.announce_emergency_corridor(req.vehicle_type, req.origin, req.destination)

    return {
        "event_id": event.event_id,
        "vehicle_id": event.vehicle_id,
        "path": event.path,
        "eta_seconds": event.eta_seconds,
        "signal_overrides": emergency_engine.get_corridor_signal_overrides(),
    }


@app.get("/api/emergency/active")
async def emergency_active():
    if not emergency_engine:
        return []
    return [
        {
            "event_id": e.event_id,
            "vehicle_id": e.vehicle_id,
            "vehicle_type": e.vehicle_type,
            "path": e.path,
            "current_junction": (e.path[0] if e.path else ""),
            "eta_seconds": round(max(0.0, e.estimated_time_s - (time.time() - e.activated_at)), 1),
        }
        for e in emergency_engine._active_events.values()
    ]


# ---------------------------------------------------------------
# REST Endpoints — Carbon
# ---------------------------------------------------------------
@app.get("/api/carbon/today")
async def carbon_today():
    if not carbon_engine:
        return JSONResponse({"error": "Carbon module not available"}, status_code=503)
    return carbon_engine.get_today_stats()


@app.get("/api/carbon/certificate")
async def carbon_certificate():
    if not carbon_engine:
        return JSONResponse({"error": "Carbon module not available"}, status_code=503)
    cert_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "carbon_certificate.pdf")
    carbon_engine.generate_certificate(output_path=cert_path)
    if os.path.exists(cert_path):
        return FileResponse(cert_path, media_type="application/pdf",
                            filename="nexus_carbon_certificate.pdf")
    return JSONResponse({"error": "Certificate generation failed"}, status_code=500)


@app.get("/api/carbon/history")
async def carbon_history():
    if not carbon_engine:
        return []
    return carbon_engine.get_all_daily_stats()


# ---------------------------------------------------------------
# REST Endpoints — Pedestrian Safety
# ---------------------------------------------------------------
@app.get("/api/pedestrian/analyze")
async def pedestrian_analyze(junction_id: str = "J1_1"):
    if not pedestrian_ai:
        return JSONResponse({"error": "Pedestrian module not available"}, status_code=503)
    result = pedestrian_ai.analyze_frame(frame=None, junction_id=junction_id)
    return result


# ---------------------------------------------------------------
# REST Endpoints — Cybersecurity
# ---------------------------------------------------------------
@app.post("/api/security/validate")
async def security_validate(req: SecurityValidateRequest):
    if not security_detector:
        return JSONResponse({"error": "Security module not available"}, status_code=503)
    return security_detector.validate_command(req.junction_id, req.new_phase, source=req.source)


@app.post("/api/security/simulate")
async def security_simulate(req: SecuritySimulateRequest):
    if not security_detector:
        return JSONResponse({"error": "Security module not available"}, status_code=503)
    return security_detector.simulate_attack(req.attack_type, req.junction_id)


@app.get("/api/security/events")
async def security_events():
    if not security_detector:
        return []
    return security_detector.get_events(limit=50)


# ---------------------------------------------------------------
# REST Endpoints — Road Maintenance
# ---------------------------------------------------------------
@app.get("/api/maintenance/orders")
async def maintenance_orders():
    if not maintenance_ai:
        return []
    return maintenance_ai.get_open_orders()


# ---------------------------------------------------------------
# REST Endpoints — NL Command
# ---------------------------------------------------------------
@app.post("/api/nl/command")
async def nl_command(req: NLCommandRequest):
    if not nl_parser:
        return JSONResponse({"error": "NL command module not available"}, status_code=503)
    parsed = nl_parser.parse(req.text)
    response = {
        "intent": parsed.intent,
        "confidence": parsed.confidence,
        "junctions": parsed.junctions,
        "duration_minutes": parsed.duration_minutes,
        "direction": parsed.direction,
        "vehicle_type": parsed.vehicle_type,
        "phase": parsed.phase,
        "parameters": parsed.parameters,
        "raw_text": parsed.raw_text,
    }

    # Auto-execute high-confidence commands
    if parsed.confidence >= 0.8:
        if parsed.intent == "emergency" and emergency_engine:
            origin = parsed.junctions[0] if parsed.junctions else "J0_0"
            destination = parsed.junctions[1] if len(parsed.junctions) > 1 else "J3_3"
            vtype = parsed.vehicle_type or "ambulance"
            event = emergency_engine.activate_corridor("nl_" + str(int(time.time())), vtype, origin, destination)
            if event:
                response["action_taken"] = {"emergency_activated": True, "path": event.path}

        elif parsed.intent == "override_signal":
            jid = parsed.junctions[0] if parsed.junctions else "J1_1"
            phase = "NS_GREEN" if parsed.direction == "NS" else "EW_GREEN"
            if jid in _junction_states:
                _junction_states[jid]["phase"] = phase
                _junction_states[jid]["is_overridden"] = True
                response["action_taken"] = {"signal_overridden": True, "junction": jid, "phase": phase}

    return response


# ---------------------------------------------------------------
# REST Endpoints — Counterfactual
# ---------------------------------------------------------------
@app.get("/api/counterfactual")
async def counterfactual_comparison():
    if not counterfactual:
        return JSONResponse({"error": "Counterfactual module not available"}, status_code=503)
    return counterfactual.get_comparison()


# ---------------------------------------------------------------
# REST Endpoints — Voice Broadcast
# ---------------------------------------------------------------
@app.post("/api/voice/announce")
async def voice_announce(req: VoiceAnnounceRequest):
    if not voice:
        return JSONResponse({"error": "Voice module not available"}, status_code=503)
    path = voice.announce(req.message, language=req.language, play=req.play)
    return {"status": "ok", "audio_file": path}


@app.get("/api/voice/log")
async def voice_log(limit: int = 20):
    if not voice:
        return []
    return voice.get_broadcast_log(limit=limit)


# ---------------------------------------------------------------
# REST Endpoints — Aggregated Metrics
# ---------------------------------------------------------------
@app.get("/api/metrics/overview")
async def metrics_overview():
    if live_runtime.enabled and live_runtime.latest_traffic:
        snapshot_data = live_runtime.latest_traffic
    else:
        snapshot_data = demo_gen.get_snapshot() if demo_gen else {}

    overview = {
        "traffic": {
            "total_queue": snapshot_data.get("total_queue", 0),
            "avg_waiting_time": snapshot_data.get("avg_waiting_time", 0),
            "throughput": snapshot_data.get("throughput", 0),
            "phase": snapshot_data.get("phase", "unknown"),
        },
        "carbon": carbon_engine.get_today_stats() if carbon_engine else {},
        "counterfactual": counterfactual.get_comparison() if counterfactual else {},
        "emergency_active": len(emergency_engine._active_events) if emergency_engine else 0,
        "security_events_24h": len(security_detector.get_events()) if security_detector else 0,
        "maintenance_orders": len(maintenance_ai.get_open_orders()) if maintenance_ai else 0,
        "voice_broadcasts": voice.get_stats() if voice else {},
        "ws_clients": ws_manager.count,
    }
    return overview


# ---------------------------------------------------------------
# REST Endpoints — AI / ML Analytics
# ---------------------------------------------------------------
MLAnomalyDetector = _safe_import("prediction.ml_anomaly_detector", "MLAnomalyDetector")
TrafficXAI = _safe_import("src.explainability.explainer", "TrafficXAI")

_ml_anomaly = None
_xai_engine = None

@app.get("/api/ai/status")
async def ai_status():
    """AI/ML model availability and status."""
    models = {
        "dqn_agent": os.path.isfile("models/dqn_20260226_014406/best/best_model.zip"),
        "lstm_predictor": os.path.isfile("models/lstm_predictor.pt"),
        "ml_anomaly_iforest": os.path.isfile("models/ml_anomaly/iforest.pkl"),
        "ml_anomaly_autoencoder": os.path.isfile("models/ml_anomaly/autoencoder.pt"),
    }
    return {
        "models": models,
        "trained_count": sum(models.values()),
        "total_count": len(models),
        "gpu_available": _check_gpu(),
    }

@app.get("/api/ai/lstm/results")
async def ai_lstm_results():
    """Get LSTM training results if available."""
    results = _load_json_safe("results/lstm/lstm_training_results.json")
    if not results:
        return {"status": "not_trained", "message": "Run: python scripts/train_lstm.py"}
    return results

@app.get("/api/ai/anomaly/results")
async def ai_anomaly_results():
    """Get ML anomaly detection results."""
    results = _load_json_safe("results/anomaly/anomaly_detection_results.json")
    if not results:
        return {"status": "not_evaluated", "message": "Run: python -m prediction.ml_anomaly_detector --generate"}
    return results

@app.get("/api/ai/xai/importance")
async def ai_xai_importance():
    """Get feature importance analysis."""
    results = _load_json_safe("results/xai/xai_report.json")
    if not results:
        return {"status": "not_computed", "message": "Run: python -m src.explainability.explainer --model <path>"}
    return results

@app.get("/api/ai/comparison")
async def ai_agent_comparison():
    """Get agent comparison results (DQN vs PPO vs A2C)."""
    results = _load_json_safe("results/comparison/comparison_results.json")
    if not results:
        return {"status": "not_run", "message": "Run: python scripts/compare_agents.py"}
    return results

@app.get("/api/ai/explain")
async def ai_explain_decision(
    queue_n: float = Query(0.5), queue_s: float = Query(0.3),
    queue_e: float = Query(0.4), queue_w: float = Query(0.2),
    wait_n: float = Query(0.3), wait_s: float = Query(0.2),
    wait_e: float = Query(0.4), wait_w: float = Query(0.1),
):
    """Get AI explanation for a signal decision given current state."""
    global _xai_engine
    if _xai_engine is None and TrafficXAI:
        model_path = "models/dqn_20260226_014406/best/best_model.zip"
        _xai_engine = TrafficXAI(model_path=model_path if os.path.isfile(model_path) else None)

    obs = np.array([queue_n, queue_s, queue_e, queue_w,
                     wait_n, wait_s, wait_e, wait_w,
                     1, 0, 0, 0, 0.5], dtype=np.float32)

    if _xai_engine:
        explanation = _xai_engine.explain_decision(obs, action=1)
        return explanation
    return {"error": "XAI engine not available"}

@app.get("/api/ai/training-history")
async def ai_training_history():
    """Return LSTM loss curves for frontend charting."""
    lstm = _load_json_safe("results/lstm/lstm_training_results.json")
    if lstm and "history" in lstm:
        return {"history": lstm["history"], "epochs": lstm.get("epochs_trained", 0)}
    return {"history": {}, "epochs": 0}


def _check_gpu() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "name": torch.cuda.get_device_name(0),
                    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)}
    except Exception:
        pass
    return {"available": False}


def _load_json_safe(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------
# WebSocket — Real-Time Stream
# ---------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Stream live data at ~1 Hz to connected clients."""
    await ws_manager.connect(ws)
    try:
        while True:
            if live_runtime.enabled:
                traffic = await live_runtime.tick()
            elif demo_gen:
                traffic = demo_gen.get_snapshot()
            else:
                traffic = {}

            if counterfactual and traffic:
                queues = {
                    "N": int(traffic.get("queues", {}).get("north", 0)),
                    "S": int(traffic.get("queues", {}).get("south", 0)),
                    "E": int(traffic.get("queues", {}).get("east", 0)),
                    "W": int(traffic.get("queues", {}).get("west", 0)),
                }
                counterfactual.record_comparison(
                    ai_avg_wait=traffic.get("avg_waiting_time", 0.0),
                    ai_total_queue=int(traffic.get("total_queue", 0)),
                    ai_throughput=int(traffic.get("throughput", 0)),
                    queue_lengths=queues,
                )
            if carbon_engine and traffic:
                idle_ai = float(traffic.get("avg_waiting_time", 0.0)) / 60.0
                carbon_engine.record_snapshot(
                    idle_ai,
                    idle_ai * 1.35,
                    max(1, int(traffic.get("total_queue", 1))),
                )

            payload = {
                "timestamp": time.time(),
                "junctions": list(_junction_states.values()),
            }
            payload["traffic"] = traffic
            if counterfactual:
                payload["counterfactual"] = counterfactual.get_comparison()
            if carbon_engine:
                payload["carbon"] = carbon_engine.get_today_stats()
            if emergency_engine:
                payload["emergency_events"] = len(emergency_engine._active_events)
            if live_runtime.latest_incidents:
                payload["incidents"] = live_runtime.latest_incidents
            if live_runtime.latest_anomaly:
                payload["anomaly"] = live_runtime.latest_anomaly
            if live_runtime.latest_prediction:
                payload["prediction"] = live_runtime.latest_prediction

            await ws.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# Keep old /ws endpoint for backward compatibility
@app.websocket("/ws")
async def websocket_legacy(ws: WebSocket):
    await ws.accept()
    gen = DemoDataGenerator(mode="rl") if DEMO_MODE else None
    try:
        while True:
            if live_runtime.enabled:
                await ws.send_json(await live_runtime.tick())
            elif gen:
                await ws.send_json(gen.get_snapshot())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------
# Background: Expire signal overrides
# ---------------------------------------------------------------
async def _override_expiry_loop():
    while True:
        now = time.time()
        expired = [jid for jid, info in _signal_overrides.items() if info["expires"] < now]
        for jid in expired:
            del _signal_overrides[jid]
            if jid in _junction_states:
                _junction_states[jid]["is_overridden"] = False
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_override_expiry_loop())
    logger.info("[NEXUS] Runtime mode: live=%s demo=%s", LIVE_MODE, DEMO_MODE)
    logger.info("[NEXUS] Live runtime status: %s", live_runtime.status())
    logger.info("[NEXUS] Backend started. Modules loaded: %d/8",
                8 - len(_import_errors))


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  NEXUS-ATMS  —  Intelligent Traffic Management Backend")
    print("=" * 60)
    print(f"  Demo Mode : {DEMO_MODE}")
    print(f"  Modules   : {8 - len(_import_errors)}/8 loaded")
    if _import_errors:
        for mod, err in _import_errors.items():
            print(f"    [WARN] {mod}: {err}")
    print(f"  Dashboard : http://localhost:8000")
    print(f"  API Docs  : http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

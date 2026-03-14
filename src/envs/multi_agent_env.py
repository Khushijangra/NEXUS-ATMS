"""
Multi-Agent SUMO Environment for 4×4 Grid
==========================================
Each of the 16 junctions is controlled by an independent RL sub-agent
that shares the same policy network (parameter sharing).

State space per junction (21 dimensions):
  - Local queue lengths [N, S, E, W]            (4)
  - Local avg waiting times [N, S, E, W]        (4)
  - Current phase one-hot [4]                   (4)
  - Time since last phase change [1]            (1)
  - Neighbor pressure [N, S, E, W]              (4)  ← queue at adjacent junctions
  - Global hour-of-day [sin, cos]               (2)
  - Global congestion level [1]                 (1)
  - Emergency flag [1]                          (1)
                                         Total: 21

Action space per junction (Discrete 2):
  0 = Keep current phase
  1 = Switch to next phase

Reward: Combination of local + global components to encourage cooperation.

This environment wraps all 16 junctions into a single Gymnasium env using
a flattened observation/action space so it works directly with
Stable-Baselines3 (DQN, PPO, A2C).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import math
from typing import Dict, List, Optional, Tuple

logger_name = __name__

try:
    import traci
    import sumolib
    _TRACI_OK = True
except ImportError:
    _TRACI_OK = False


class MultiAgentSumoEnv(gym.Env):
    """
    Multi-agent traffic signal control for a 4×4 intersection grid.

    All 16 junctions share one policy (parameter sharing) via a flattened
    observation/action space compatible with SB3.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Grid layout parameters
    GRID_ROWS = 4
    GRID_COLS = 4
    N_JUNCTIONS = 16

    # Per-junction state dims
    LOCAL_STATE_DIM = 21
    N_ACTIONS_PER_JUNCTION = 2

    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 3,
        min_green: int = 10,
        max_green: int = 60,
        reward_type: str = "cooperative",
        render_mode: Optional[str] = None,
        cooperation_weight: float = 0.3,
    ):
        super().__init__()

        self.net_file = os.path.abspath(net_file)
        self.route_file = os.path.abspath(route_file)
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.cooperation_weight = cooperation_weight

        # Junction IDs — 4x4 grid named J{row}_{col}
        self.junction_ids: List[str] = []
        self._junction_grid: Dict[Tuple[int, int], str] = {}
        self._junction_neighbors: Dict[str, List[str]] = {}

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                jid = f"J{r}_{c}"
                self.junction_ids.append(jid)
                self._junction_grid[(r, c)] = jid

        # Build neighbor map
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                jid = self._junction_grid[(r, c)]
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                        neighbors.append(self._junction_grid[(nr, nc)])
                    else:
                        neighbors.append(None)  # boundary — no neighbor
                self._junction_neighbors[jid] = neighbors  # [N, S, W, E]

        # Gymnasium spaces — flattened for all 16 junctions
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.N_JUNCTIONS * self.LOCAL_STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [self.N_ACTIONS_PER_JUNCTION] * self.N_JUNCTIONS
        )

        # Per-junction internal state
        self._current_phase: Dict[str, int] = {}
        self._time_since_change: Dict[str, int] = {}
        self._is_yellow: Dict[str, bool] = {}
        self._green_phases: Dict[str, List[int]] = {}
        self._incoming_edges: Dict[str, Dict[str, str]] = {}

        # Episode metrics
        self._step_count = 0
        self._sumo_running = False
        self._conn_label = "multi_agent"
        self._episode_metrics = {
            "total_waiting_time": 0.0,
            "total_queue_length": 0.0,
            "throughput": 0,
            "total_reward": 0.0,
            "phase_changes": 0,
        }

    def _discover_junction_topology(self):
        """Auto-discover incoming edges and phases for each traffic light."""
        for jid in self.junction_ids:
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(jid)
                # Get unique incoming edges
                edges = list(dict.fromkeys(
                    traci.lane.getEdgeID(lane) for lane in controlled_lanes
                ))
                # Map to cardinal directions (approximate by edge name or index)
                direction_map = {}
                direction_names = ["north", "south", "east", "west"]
                for i, edge in enumerate(edges[:4]):
                    direction_map[direction_names[i]] = edge
                self._incoming_edges[jid] = direction_map

                # Discover green phases
                logic = traci.trafficlight.getAllProgramLogics(jid)
                if logic:
                    phases = logic[0].phases
                    green_phases = []
                    for idx, phase in enumerate(phases):
                        # Green phases contain 'G' but not 'y'
                        if 'G' in phase.state and 'y' not in phase.state:
                            green_phases.append(idx)
                    self._green_phases[jid] = green_phases if green_phases else [0, 2]
                else:
                    self._green_phases[jid] = [0, 2]
            except Exception:
                self._incoming_edges[jid] = {}
                self._green_phases[jid] = [0, 2]

    def _start_sumo(self):
        """Start SUMO simulation."""
        if self._sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        if "SUMO_HOME" not in os.environ:
            sys.exit("SUMO_HOME not set")

        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--time-to-teleport", "-1",
            "--random",
        ]
        traci.start(sumo_cmd, label=self._conn_label)
        self._sumo_running = True

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_junction_state(self, jid: str, global_congestion: float,
                            sim_time: float) -> np.ndarray:
        """Build the 21-dim state vector for one junction."""
        state = []
        max_queue = 30.0
        max_wait = 150.0

        edges = self._incoming_edges.get(jid, {})

        # Local queue lengths (4)
        local_queues = []
        for direction in ["north", "south", "east", "west"]:
            edge = edges.get(direction)
            if edge:
                q = traci.edge.getLastStepHaltingNumber(edge)
                local_queues.append(min(q / max_queue, 1.0))
            else:
                local_queues.append(0.0)
        state.extend(local_queues)

        # Local waiting times (4)
        for direction in ["north", "south", "east", "west"]:
            edge = edges.get(direction)
            if edge:
                w = traci.edge.getWaitingTime(edge)
                state.append(min(w / max_wait, 1.0))
            else:
                state.append(0.0)

        # Phase one-hot (4)
        green_phases = self._green_phases.get(jid, [0, 2])
        n_phases = max(len(green_phases), 2)
        phase_onehot = [0.0] * 4
        cur_phase = self._current_phase.get(jid, 0)
        is_yellow = self._is_yellow.get(jid, False)
        if is_yellow:
            phase_onehot[min(cur_phase + n_phases, 3)] = 1.0
        else:
            phase_onehot[min(cur_phase, 3)] = 1.0
        state.extend(phase_onehot)

        # Time since change (1)
        tsc = self._time_since_change.get(jid, 0)
        state.append(min(tsc / self.max_green, 1.0))

        # Neighbor pressure (4) — sum of queues at adjacent junctions
        neighbors = self._junction_neighbors.get(jid, [None]*4)
        for neighbor_jid in neighbors:
            if neighbor_jid and neighbor_jid in self._incoming_edges:
                n_edges = self._incoming_edges[neighbor_jid]
                total_q = sum(
                    traci.edge.getLastStepHaltingNumber(e)
                    for e in n_edges.values() if e
                )
                state.append(min(total_q / (max_queue * 4), 1.0))
            else:
                state.append(0.0)

        # Global hour-of-day (sin/cos) (2)
        hour_frac = (sim_time % 86400) / 86400.0
        state.append(math.sin(2 * math.pi * hour_frac))
        state.append(math.cos(2 * math.pi * hour_frac))

        # Global congestion level (1)
        state.append(min(global_congestion, 1.0))

        # Emergency flag (1) — placeholder, can be set by corridor engine
        state.append(0.0)

        return np.array(state, dtype=np.float32)

    def _get_full_state(self) -> np.ndarray:
        """Build the full observation for all 16 junctions."""
        sim_time = traci.simulation.getTime()

        # Global congestion: fraction of vehicles that are halted
        total_vehicles = max(traci.vehicle.getIDCount(), 1)
        halted = sum(
            1 for vid in traci.vehicle.getIDList()
            if traci.vehicle.getSpeed(vid) < 0.1
        )
        global_congestion = halted / total_vehicles

        states = []
        for jid in self.junction_ids:
            states.append(self._get_junction_state(jid, global_congestion, sim_time))
        return np.concatenate(states)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _apply_junction_action(self, jid: str, action: int):
        """Apply action for a single junction."""
        green_phases = self._green_phases.get(jid, [0, 2])
        is_yellow = self._is_yellow.get(jid, False)
        cur_phase = self._current_phase.get(jid, 0)
        tsc = self._time_since_change.get(jid, 0)

        if action == 1 and not is_yellow:
            if tsc >= self.min_green:
                # Transition to yellow
                self._is_yellow[jid] = True
                yellow_idx = green_phases[cur_phase % len(green_phases)] + 1
                try:
                    traci.trafficlight.setPhase(jid, yellow_idx)
                except Exception:
                    pass
                self._episode_metrics["phase_changes"] += 1

        # Handle yellow → green transition
        if self._is_yellow.get(jid, False) and tsc >= self.yellow_time:
            self._is_yellow[jid] = False
            next_phase = (cur_phase + 1) % len(green_phases)
            self._current_phase[jid] = next_phase
            try:
                traci.trafficlight.setPhase(jid, green_phases[next_phase])
            except Exception:
                pass
            self._time_since_change[jid] = 0
        elif not self._is_yellow.get(jid, False):
            # Enforce max green
            if tsc >= self.max_green:
                self._is_yellow[jid] = True
                yellow_idx = green_phases[cur_phase % len(green_phases)] + 1
                try:
                    traci.trafficlight.setPhase(jid, yellow_idx)
                except Exception:
                    pass
                self._time_since_change[jid] = 0
                self._episode_metrics["phase_changes"] += 1

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        Cooperative reward: weighted combination of local + global components.

        R = (1 - w) * avg_local_reward + w * global_reward
        where local_reward = -(queue + wait) per junction
        and global_reward penalizes total network congestion.
        """
        total_wait = 0.0
        total_queue = 0
        junction_rewards = []

        for jid in self.junction_ids:
            local_wait = 0.0
            local_queue = 0
            edges = self._incoming_edges.get(jid, {})
            for edge in edges.values():
                if edge:
                    local_wait += traci.edge.getWaitingTime(edge)
                    local_queue += traci.edge.getLastStepHaltingNumber(edge)
            total_wait += local_wait
            total_queue += local_queue
            junction_rewards.append(-local_wait / 200.0 - local_queue / 30.0)

        throughput = traci.simulation.getArrivedNumber()

        # Local: average of per-junction rewards
        avg_local = np.mean(junction_rewards) if junction_rewards else 0.0

        # Global: penalize network-wide congestion, reward throughput
        global_reward = -total_wait / (200.0 * self.N_JUNCTIONS) + throughput / 20.0

        # Cooperative reward
        w = self.cooperation_weight
        reward = (1 - w) * avg_local + w * global_reward

        # Update metrics
        self._episode_metrics["total_waiting_time"] += total_wait
        self._episode_metrics["total_queue_length"] += total_queue
        self._episode_metrics["throughput"] += throughput
        self._episode_metrics["total_reward"] += reward

        return float(reward)

    # ------------------------------------------------------------------
    # Gym Interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start_sumo()

        # Discover topology
        self._discover_junction_topology()

        # Reset per-junction state
        for jid in self.junction_ids:
            self._current_phase[jid] = 0
            self._time_since_change[jid] = 0
            self._is_yellow[jid] = False
            green_phases = self._green_phases.get(jid, [0, 2])
            try:
                traci.trafficlight.setPhase(jid, green_phases[0])
            except Exception:
                pass

        self._step_count = 0
        self._episode_metrics = {
            "total_waiting_time": 0.0,
            "total_queue_length": 0.0,
            "throughput": 0,
            "total_reward": 0.0,
            "phase_changes": 0,
        }

        # Warm up
        for _ in range(10):
            traci.simulationStep()

        obs = self._get_full_state()
        return obs, {"step": 0}

    def step(self, action):
        """
        Execute one step with actions for all 16 junctions.

        action: array of 16 ints, each in {0, 1}
        """
        # Apply per-junction actions
        for i, jid in enumerate(self.junction_ids):
            self._apply_junction_action(jid, int(action[i]))

        # Advance simulation
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_count += 1
            for jid in self.junction_ids:
                self._time_since_change[jid] = self._time_since_change.get(jid, 0) + 1

        reward = self._compute_reward()
        obs = self._get_full_state()

        terminated = self._step_count >= self.max_steps
        if traci.simulation.getMinExpectedNumber() <= 0:
            terminated = True
        truncated = False

        info = {"step": self._step_count}
        if terminated:
            avg_steps = max(self._step_count // self.delta_time, 1)
            info["metrics"] = {
                "total_waiting_time": self._episode_metrics["total_waiting_time"],
                "avg_waiting_time": self._episode_metrics["total_waiting_time"] / avg_steps,
                "avg_queue_length": self._episode_metrics["total_queue_length"] / avg_steps,
                "throughput": self._episode_metrics["throughput"],
                "total_reward": self._episode_metrics["total_reward"],
                "phase_changes": self._episode_metrics["phase_changes"],
            }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    def get_metrics(self) -> Dict:
        avg_steps = max(self._step_count // self.delta_time, 1)
        return {
            "step": self._step_count,
            "avg_waiting_time": self._episode_metrics["total_waiting_time"] / avg_steps,
            "avg_queue_length": self._episode_metrics["total_queue_length"] / avg_steps,
            "throughput": self._episode_metrics["throughput"],
            "total_reward": self._episode_metrics["total_reward"],
        }

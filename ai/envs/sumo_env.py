"""
SUMO Traffic Environment for Reinforcement Learning
Single Intersection Environment with Gymnasium API
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
from typing import Dict, Tuple, Optional, List
import traci
import sumolib


class SumoEnvironment(gym.Env):
    """
    Custom Gymnasium environment for traffic signal control using SUMO.

    The RL agent controls a single intersection's traffic signal to minimize
    vehicle waiting times and queue lengths. The agent observes queue lengths,
    waiting times, the current phase, and elapsed time, then decides whether
    to keep the current phase or switch.

    State Space (13 dimensions):
        - Queue lengths per approach: [N, S, E, W]
        - Avg waiting times per approach: [N, S, E, W]
        - Current phase (one-hot): [4]
        - Normalized time since last phase change: [1]

    Action Space (Discrete 2):
        - 0: Keep current phase
        - 1: Switch to next phase
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

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
        reward_type: str = "combined",
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the SUMO environment.

        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            use_gui: Whether to use SUMO GUI for visualization
            max_steps: Maximum simulation steps per episode
            delta_time: Seconds between agent decisions
            yellow_time: Yellow phase duration in seconds
            min_green: Minimum green phase duration
            max_green: Maximum green phase duration
            reward_type: Reward function type (waiting_time | queue | combined)
            render_mode: Gymnasium render mode
        """
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

        # Traffic light settings
        self.tl_id = "center"           # Traffic light ID in the network
        self.num_phases = 4             # NS-green, NS-yellow, EW-green, EW-yellow
        self.num_green_phases = 2       # Only green phases (0, 2)
        self.green_phases = [0, 2]      # Phase indices for green

        # State dimensions
        self.num_approaches = 4         # N, S, E, W
        # State: queue(4) + wait(4) + phase_onehot(4) + time_since_change(1)
        self.state_dim = self.num_approaches * 2 + self.num_green_phases + 2 + 1

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0=keep, 1=switch

        # Approach edges for each direction
        self.incoming_edges = {
            "north": "north_in",
            "south": "south_in",
            "east": "east_in",
            "west": "west_in",
        }
        self.outgoing_edges = {
            "north": "north_out",
            "south": "south_out",
            "east": "east_out",
            "west": "west_out",
        }

        # Internal state
        self._step_count = 0
        self._current_phase_idx = 0     # Index into green_phases
        self._time_since_change = 0
        self._is_yellow = False
        self._sumo_running = False
        self._conn_label = "default"

        # Episode metrics
        self._episode_waiting_time = 0.0
        self._episode_queue_length = 0.0
        self._episode_throughput = 0
        self._episode_rewards = 0.0
        self._prev_waiting_time = 0.0
        self._total_vehicles_entered = 0
        self._total_vehicles_left = 0
        self._phase_changes = 0

    def _start_sumo(self) -> None:
        """Start SUMO simulation process."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        # Check SUMO_HOME
        if "SUMO_HOME" not in os.environ:
            sys.exit("Please declare environment variable 'SUMO_HOME'")

        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--time-to-teleport", "-1",    # Disable teleporting
            "--random",
        ]

        traci.start(sumo_cmd, label=self._conn_label)
        self._sumo_running = True

    def _get_state(self) -> np.ndarray:
        """
        Get current state observation vector.

        Returns:
            Normalized state vector of shape (state_dim,).
        """
        state = []

        # Queue lengths per approach (normalized by max ~50 vehicles)
        max_queue = 50.0
        for direction in ["north", "south", "east", "west"]:
            edge = self.incoming_edges[direction]
            queue = traci.edge.getLastStepHaltingNumber(edge)
            state.append(min(queue / max_queue, 1.0))

        # Average waiting times per approach (normalized by max ~200s)
        max_wait = 200.0
        for direction in ["north", "south", "east", "west"]:
            edge = self.incoming_edges[direction]
            wait = traci.edge.getWaitingTime(edge)
            state.append(min(wait / max_wait, 1.0))

        # Current phase one-hot encoding
        phase_onehot = [0.0] * (self.num_green_phases + 2)
        if self._is_yellow:
            phase_onehot[self.num_green_phases + (self._current_phase_idx % 2)] = 1.0
        else:
            phase_onehot[self._current_phase_idx] = 1.0
        state.extend(phase_onehot)

        # Time since last phase change (normalized by max_green)
        state.append(min(self._time_since_change / self.max_green, 1.0))

        return np.array(state, dtype=np.float32)

    def _compute_reward(self) -> float:
        """
        Compute reward based on current traffic state.

        Supports three reward types:
            - waiting_time: Minimize total waiting time
            - queue: Minimize total queue length
            - combined: Weighted combination of waiting time, queue, and throughput

        Returns:
            Reward value (negative is bad, positive is good).
        """
        # Gather metrics
        total_waiting = 0.0
        total_queue = 0
        for direction in ["north", "south", "east", "west"]:
            edge = self.incoming_edges[direction]
            total_waiting += traci.edge.getWaitingTime(edge)
            total_queue += traci.edge.getLastStepHaltingNumber(edge)

        # Throughput: vehicles that left the network this step
        throughput = traci.simulation.getArrivedNumber()

        # Update episode metrics
        self._episode_waiting_time += total_waiting
        self._episode_queue_length += total_queue
        self._episode_throughput += throughput

        if self.reward_type == "waiting_time":
            # Reward = negative change in waiting time
            reward = self._prev_waiting_time - total_waiting
            self._prev_waiting_time = total_waiting

        elif self.reward_type == "queue":
            # Reward = negative queue length
            reward = -total_queue / 50.0

        else:  # combined
            # Weighted combination
            wait_penalty = -total_waiting / 200.0
            queue_penalty = -total_queue / 50.0
            throughput_bonus = throughput / 10.0
            reward = 0.5 * wait_penalty + 0.3 * queue_penalty + 0.2 * throughput_bonus

        return float(reward)

    def _apply_action(self, action: int) -> None:
        """
        Apply agent action to traffic signal.

        Args:
            action: 0 = keep current phase, 1 = switch to next phase.
        """
        if action == 1 and not self._is_yellow:
            # Check minimum green time
            if self._time_since_change >= self.min_green:
                # Set yellow phase
                self._is_yellow = True
                yellow_phase = self.green_phases[self._current_phase_idx] + 1
                traci.trafficlight.setPhase(self.tl_id, yellow_phase)
                self._phase_changes += 1

        # If currently in yellow, check if yellow time has elapsed
        if self._is_yellow and self._time_since_change >= self.yellow_time:
            self._is_yellow = False
            self._current_phase_idx = (self._current_phase_idx + 1) % self.num_green_phases
            traci.trafficlight.setPhase(
                self.tl_id, self.green_phases[self._current_phase_idx]
            )
            self._time_since_change = 0
        elif not self._is_yellow:
            # Enforce max green time
            if self._time_since_change >= self.max_green:
                self._is_yellow = True
                yellow_phase = self.green_phases[self._current_phase_idx] + 1
                traci.trafficlight.setPhase(self.tl_id, yellow_phase)
                self._time_since_change = 0
                self._phase_changes += 1

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options.

        Returns:
            Tuple of (initial observation, info dict).
        """
        super().reset(seed=seed)

        # Reset SUMO
        self._start_sumo()

        # Reset internal state
        self._step_count = 0
        self._current_phase_idx = 0
        self._time_since_change = 0
        self._is_yellow = False
        self._episode_waiting_time = 0.0
        self._episode_queue_length = 0.0
        self._episode_throughput = 0
        self._episode_rewards = 0.0
        self._prev_waiting_time = 0.0
        self._phase_changes = 0

        # Set initial phase
        traci.trafficlight.setPhase(self.tl_id, self.green_phases[0])

        # Warm up simulation (let some vehicles enter)
        for _ in range(10):
            traci.simulationStep()

        obs = self._get_state()
        info = {"step": 0, "phase": self._current_phase_idx}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Agent action (0 = keep, 1 = switch).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Apply action
        self._apply_action(action)

        # Advance simulation by delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_count += 1
            self._time_since_change += 1

        # Compute reward
        reward = self._compute_reward()
        self._episode_rewards += reward

        # Get new observation
        obs = self._get_state()

        # Check termination
        terminated = self._step_count >= self.max_steps
        truncated = False

        # Check if simulation ended early (no more vehicles)
        if traci.simulation.getMinExpectedNumber() <= 0:
            terminated = True

        # Info dict with metrics
        info = {
            "step": self._step_count,
            "phase": self._current_phase_idx,
            "is_yellow": self._is_yellow,
            "phase_changes": self._phase_changes,
        }

        if terminated:
            avg_steps = max(self._step_count // self.delta_time, 1)
            info["metrics"] = {
                "total_waiting_time": self._episode_waiting_time,
                "avg_waiting_time": self._episode_waiting_time / avg_steps,
                "total_queue_length": self._episode_queue_length,
                "avg_queue_length": self._episode_queue_length / avg_steps,
                "throughput": self._episode_throughput,
                "total_reward": self._episode_rewards,
                "phase_changes": self._phase_changes,
            }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (handled by SUMO GUI if enabled)."""
        if self.render_mode == "human" and not self.use_gui:
            print(
                f"Step {self._step_count} | Phase {self._current_phase_idx} | "
                f"Yellow: {self._is_yellow} | "
                f"Reward: {self._episode_rewards:.2f}"
            )

    def close(self) -> None:
        """Clean up SUMO resources."""
        if self._sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    def get_metrics(self) -> Dict:
        """Get current episode metrics snapshot."""
        avg_steps = max(self._step_count // self.delta_time, 1)
        return {
            "step": self._step_count,
            "total_waiting_time": self._episode_waiting_time,
            "avg_waiting_time": self._episode_waiting_time / avg_steps,
            "total_queue_length": self._episode_queue_length,
            "avg_queue_length": self._episode_queue_length / avg_steps,
            "throughput": self._episode_throughput,
            "total_reward": self._episode_rewards,
            "phase_changes": self._phase_changes,
        }


# Register environment with Gymnasium
gym.register(
    id="SumoTraffic-v0",
    entry_point="src.envs.sumo_env:SumoEnvironment",
)

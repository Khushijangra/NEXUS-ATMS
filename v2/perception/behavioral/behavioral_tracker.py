import numpy as np

class BehavioralTracker:
    def __init__(self):
        self.trajectories = {}
        
    def update_vehicle(self, veh_id, x, y, timestamp):
        if veh_id not in self.trajectories:
            self.trajectories[veh_id] = []
        self.trajectories[veh_id].append((x, y, timestamp))
        
    def compute_speed(self, veh_id):
        traj = self.trajectories.get(veh_id, [])
        if len(traj) < 2: return 0.0
        dx = traj[-1][0] - traj[-2][0]
        dy = traj[-1][1] - traj[-2][1]
        dt = traj[-1][2] - traj[-2][2]
        return np.sqrt(dx**2 + dy**2) / (dt + 1e-8)
        
    def compute_acceleration(self, veh_id):
        traj = self.trajectories.get(veh_id, [])
        if len(traj) < 3: return 0.0
        v2 = self.compute_speed(veh_id)
        # simplistic approx
        dt = traj[-1][2] - traj[-2][2]
        # mock v1
        return v2 / (dt + 1e-8)
        
    def compute_jerk(self, veh_id):
        return 0.0 # mock derivative of accel

import numpy as np

class ConstraintEngine:
    def __init__(self, num_phases=4):
        self.num_phases = num_phases
        self.illegal_transition_matrix = np.zeros((num_phases, num_phases))
        # Example: Phase 0 cannot transition to Phase 2 directly
        if num_phases >= 3:
            self.illegal_transition_matrix[0, 2] = 1
            
        self.phase_timer = 0
        self.max_phase_duration = 60 # seconds
        self.current_phase = 0
        
    def validate_action(self, action):
        if self.illegal_transition_matrix[self.current_phase, action] == 1:
            return self.current_phase # Reject transition
        return action
        
    def phase_lock_detector(self):
        if self.phase_timer > self.max_phase_duration:
            return True
        return False
        
    def emergency_override(self, emergency_detected):
        if emergency_detected:
            # Force transition to all-red or specific green
            return 0
        return None

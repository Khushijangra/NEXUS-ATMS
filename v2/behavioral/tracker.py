import numpy as np
from scipy.optimize import linear_sum_assignment

def box_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)

class Track:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.history = [detection]
        self.time_since_update = 0
        self.hits = 1
        self.state = detection.bbox
        
    def update(self, detection):
        self.history.append(detection)
        self.time_since_update = 0
        self.hits += 1
        self.state = detection.bbox
        
    def predict(self):
        self.time_since_update += 1
        return self.state

class DeepSORTTracker:
    """
    A lightweight kinematic tracker (SORT) providing the identity association
    required by DeepSORT without the heavy visual Re-ID network overhead.
    """
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections):
        """
        Takes a list of Detection objects and assigns track_ids.
        Updates internal tracks.
        """
        # Predict next positions
        predicted_boxes = [trk.predict() for trk in self.tracks]
        
        det_boxes = [det.bbox for det in detections]
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._match(predicted_boxes, det_boxes)
        
        # Update matched tracks
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])
            detections[det_idx].track_id = self.tracks[trk_idx].track_id
            
        # Create new tracks
        for det_idx in unmatched_dets:
            trk = Track(self.next_id, detections[det_idx])
            self.tracks.append(trk)
            detections[det_idx].track_id = self.next_id
            self.next_id += 1
            
        # Remove dead tracks
        self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
        
        return detections
        
    def _match(self, predicted_boxes, det_boxes):
        if len(predicted_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(predicted_boxes)))
            
        iou_matrix = np.zeros((len(predicted_boxes), len(det_boxes)), dtype=np.float32)
        for d, det in enumerate(det_boxes):
            for t, trk in enumerate(predicted_boxes):
                iou_matrix[t, d] = box_iou(trk, det)
                
        # Hungarian algorithm (maximize IoU -> minimize -IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched_indices = []
        unmatched_dets = []
        unmatched_trks = []
        
        for d in range(len(det_boxes)):
            if d not in col_ind:
                unmatched_dets.append(d)
                
        for t in range(len(predicted_boxes)):
            if t not in row_ind:
                unmatched_trks.append(t)
                
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_trks.append(r)
                unmatched_dets.append(c)
            else:
                matched_indices.append((r, c))
                
        return matched_indices, unmatched_dets, unmatched_trks

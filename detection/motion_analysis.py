import numpy as np

def motion_energy(prev_pts, cur_pts):
    if len(prev_pts) == 0 or len(cur_pts) == 0:
        return 0
    diff = np.linalg.norm(prev_pts - cur_pts, axis=1)
    return np.mean(diff)

def compute_suspicious_score(pose_pts, num_people, motion):
    if len(pose_pts) == 0:
        return 0.05
    score = min(1.0, 0.4 + 0.5 * motion + 0.05 * num_people)
    return round(score, 2)

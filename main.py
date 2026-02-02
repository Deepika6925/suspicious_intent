import cv2
from detection.yolo_detector import detect_people
from detection.pose_extractor import extract_pose
from detection.motion_analysis import motion_energy, compute_suspicious_score
from detection.gradcam import gradcam_overlay
from utils.logger import log_frame_data, save_metrics

def run(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open source.")
        return

    prev_pose = []
    frame_id = 0
    log = []

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        boxes = detect_people(frame)
        cur_pose = extract_pose(frame)
        motion = motion_energy(prev_pose, cur_pose)
        score = compute_suspicious_score(cur_pose, len(boxes), motion)
        prev_pose = cur_pose

        level = "Normal"
        if score > 0.7: level = "High"
        elif score > 0.5: level = "Medium"
        color = (0,255,0) if level=="Normal" else (0,255,255) if level=="Medium" else (0,0,255)

        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        overlay = gradcam_overlay(frame, gradcam_overlay.__globals__['cnn_model'])
        cv2.putText(overlay, f"Score:{score}  Level:{level}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        log = log_frame_data(log, frame_id, score, motion, len(boxes), level)
        frame_id += 1

        cv2.imshow("Suspicious Activity Monitor", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    save_metrics(log)

if __name__=="__main__":
    print("1: Webcam | 2: Video File")
    choice = input("Select input: ")
    if choice=="1":
        run(0)
    elif choice=="2":
        path = input("Enter video path: ").strip('"')
        run(path)
    else:
        print("Invalid choice.")





#pip install fastapi uvicorn
#pip insatll pydantic-settings
#uvicorn app.main:app --reload

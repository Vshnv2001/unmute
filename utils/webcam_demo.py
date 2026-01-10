#!/usr/bin/env python3
"""
webcam_demo.py
Quick demo: hold SPACE to record a sign (1–2s), release to classify via soft-DTW NN.
"""
import cv2, time, numpy as np, pickle
from softdtw_nn_pipeline import SoftDTWNN, make_features_from_blob  # reuse feature builder
# from your earlier script:
from generate_pose_data import extract_from_frames  # <-- implement/import this

MODEL_PROTOS = "./processed/prototypes"
WINDOW_SEC = 1.5

def main():
    nn = SoftDTWNN(MODEL_PROTOS, tau=10.0, gamma=0.1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam.")
        return

    rec = False
    buf_frames, buf_durs = [], []
    last_ts = time.time()

    print("Hold SPACE to record a sign; release to classify. Press q to quit.")
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        now = time.time()
        dur_ms = (now - last_ts) * 1000.0
        last_ts = now

        if rec:
            buf_frames.append(rgb.copy())
            buf_durs.append(dur_ms)
            cv2.putText(bgr, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow("SgSL demo", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # toggle record
            rec = not rec
            if not rec:
                # classify if we have enough frames
                if len(buf_frames) > 6:
                    blob = extract_from_frames(buf_frames, buf_durs)  # same shape as pose.pkl
                    topk = nn.topk(blob, k=3)
                    print("Prediction:", topk)
                buf_frames, buf_durs = [], []
                last_ts = time.time()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

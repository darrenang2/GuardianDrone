import cv2
from ultralytics import YOLO
import os
import shutil
import subprocess
import numpy as np

def start_rtsp_stream(width, height):
    """
    Starts an FFmpeg process that will publish raw frames to an RTSP stream.
    """
    ffmpeg_bin = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError(
            "FFmpeg executable not found. Install FFmpeg and/or set FFMPEG_PATH to the ffmpeg.exe location."
        )

    cmd = [
        ffmpeg_bin,
        '-re',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-f', 'rtsp',
        'rtsp://localhost:8554/guardian'
    ]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start RTSP publishing pipe
    ffmpeg = start_rtsp_stream(width, height)

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # ---------- MOTION DETECTION ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected = False
        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                if cv2.contourArea(c) < 1500:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        prev_gray = gray

        # ---------- YOLO OBJECT DETECTION ----------
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Overlay motion status
        status_text = "MOTION" if motion_detected else "NO MOTION"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if motion_detected else (0, 255, 0),
            2
        )

        # ---------- SHOW WINDOW ----------
        cv2.imshow("Guardian Drone - Motion + Object Detection", frame)

        # ---------- SEND FRAME TO RTSP STREAM ----------
        try:
            ffmpeg.stdin.write(frame.tobytes())
        except:
            pass  # FFmpeg process may restart if viewer disconnects

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

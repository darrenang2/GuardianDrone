import cv2
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 model (nano = fastest, good for testing)
    model = YOLO("yolov8n.pt")  # this will auto-download the first time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Run inference (use stream=True to iterate results efficiently)
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Class id and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Class name (e.g., 'person', 'dog', etc.)
                cls_name = model.names[cls_id]

                # Draw rectangle and label
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

        cv2.imshow("Guardian Drone - Object Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
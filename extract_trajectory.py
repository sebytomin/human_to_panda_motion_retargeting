
import cv2
import csv
import numpy as np
from pathlib import Path

VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initial HSV range for a blue object.
# Adjust after inspecting detection results.
LOWER_BLUE = np.array([90, 70, 40])
UPPER_BLUE = np.array([135, 255, 255])

for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    csv_path = OUTPUT_DIR / f"{video_path.stem}_trajectory.csv"
    preview_path = OUTPUT_DIR / f"{video_path.stem}_tracked.mp4"

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    trajectory = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        x_norm = None
        y_norm = None

        if contours:
            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) > 100:
                moments = cv2.moments(contour)

                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    x_norm = cx / width
                    y_norm = cy / height

                    cv2.circle(
                        frame, (cx, cy), 8, (0, 255, 0), -1
                    )

                    cv2.putText(
                        frame,
                        "Tracked object",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        trajectory.append([
            frame_id,
            frame_id / fps,
            x_norm,
            y_norm
        ])

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([
            "frame", "time_s", "x_norm", "y_norm"
        ])
        csv_writer.writerows(trajectory)

    valid = sum(row[2] is not None for row in trajectory)

    print(f"\nProcessed: {video_path.name}")
    print(f"Frames: {frame_id}")
    print(f"Detected frames: {valid}")
    print(f"Detection rate: {valid / max(frame_id, 1):.1%}")
    print(f"Saved: {csv_path}")
    print(f"Preview: {preview_path}")

print("\nTrajectory extraction complete.")

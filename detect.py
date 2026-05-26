import cv2
from ultralytics import YOLO
import time

# config
VIDEO_PATH = "traffic.mp4"  # Path to your video file
CONFIDENCE = 0.4  # Minimum confidence for detections
# Classes to detect
CLASSES = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
    13,  # bench
    15,  # cat
    16,  # dog
]

# Load the YOLOv8 nano model
model = YOLO("yolov8n.pt")

# colors of each class (BGR format - OpenCV uses BGR instead of RGB)
COLORS = {
    0:  (0, 255, 0),     # person       → green
    1:  (255, 0, 0),     # bicycle      → blue
    2:  (0, 0, 255),     # car          → red
    3:  (255, 165, 0),   # motorcycle   → orange
    5:  (0, 255, 255),   # bus          → yellow
    7:  (255, 0, 255),   # truck        → pink
    9:  (0, 128, 255),   # traffic light→ light orange
    11: (0, 0, 128),     # stop sign    → dark red
    13: (128, 128, 0),   # bench        → olive
    15: (255, 128, 0),   # cat          → sky blue
    16: (0, 128, 128),   # dog          → teal
}

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    print("Please check the file path and try again.")
    exit()

print("video opened successfully!")

# get video properties so output matches input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

# create video writer to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps_input,
                      (frame_width, frame_height))

print(f"Video resolution: {frame_width}x{frame_height} at FPS: {fps_input}")
print("saving output to output.mp4")


print(f"press Q to quit the detection window")


# main loop
while True:

    start_time = time.time()

    # read a frame from the video
    ret, frame = cap.read()

    # if video ended, break the loop
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to the first frame
        continue

    # run detection on the frame
    results = model(frame, conf=CONFIDENCE, classes=CLASSES, verbose=False)

    # dictionary to count detections per class
    counts = {}

    # loop through each detection
    for result in results:
        for box in result.boxes:

            # get box corner coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # get class id, name and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # count this detection
            counts[class_name] = counts.get(class_name, 0) + 1

            # pick color for the class
            # default to white if class not in COLORS
            color = COLORS.get(class_id, (255, 255, 255))

            # draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # build the label text
            label = f"{class_name} {confidence:.0%}"

            # draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), color, -1)

            # draw label text on top of the background
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # calculate and display FPS
    fps = 1 / (time.time() - start_time)

    # draw FPS on the frame on top-left corner
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # draw object counts in top-right corner
    y_offset = 40
    for class_name, count in counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (frame.shape[1] - 180, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    # write the frame to the output video
    out.write(frame)

    # show frame on screen
    cv2.imshow("ADAS Detection", frame)

    # press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


out.release()  # release the video writer
# cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped")

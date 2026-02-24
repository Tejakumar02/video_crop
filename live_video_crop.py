import cv2
import datetime

# RTSP Stream URL
rtsp_url = r"C:\Users\maity\Downloads\vlc-record-2025-03-18-18h36m30s-rtsp___192.168.0.100_stream1-.avi"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

# Get video properties
frame_width = 1141  # int(cap.get(3))
frame_height = 703  # int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # Default to 25 FPS if unknown

# Generate a unique filename with a timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = fr"C:\Users\maity\Downloads\rtsp_recordf1_4.avi"  # Use .mp4 for MP4

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'mp4v' for MP4
out = cv2.VideoWriter(output_filename, fourcc, fps,
                      (frame_width, frame_height))

print(f"Recording started: {output_filename}  (Press 'q' to stop)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame. Stopping recording.")
        break
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    angle = 3  # -10
    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate frame without cropping
    frame = cv2.warpAffine(frame, matrix, (w, h))
    # frame[338:1026, 454:1593]  # frame[472:1094,758:1911]
    frame = frame[454:1157, 747:1888]  # frame[324:946, 454:1593]
    out.write(frame)  # Save the frame
    cv2.imshow('RTSP Stream Recording', cv2.resize(
        frame, (960, 540)))  # Show the live stream

    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording stopped by user.")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

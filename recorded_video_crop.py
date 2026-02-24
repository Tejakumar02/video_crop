import cv2

def trim_video(input_file, output_file, start_time_sec, end_time_sec):
    cap = cv2.VideoCapture(input_file)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the FPS (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get video dimensions (width, height)
    (h, w) = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Calculate the start and end frames based on time
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)

    # Setup video writer with the original video size
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    # Set the capture position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Loop through the video frames and apply trimming
    current_frame = start_frame
    while True:
        ret, frame = cap.read()

        if not ret or current_frame > end_frame:
            break  # Stop when we reach the end frame or the video ends

        # Write the frame to output file
        out.write(frame)

        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
input_file = r"D:\rtsp_videos\rtsp_2025-11-19_14-13-16.mp4"
output_file = r"D:\rtsp_output\test.mp4"

# Start at 10 minutes (600 seconds) and end at 60 minutes (3600 seconds)
start_time = 60  # 10 minutes in seconds
end_time =  120  # 60 minutes in second 36003600   3600

# Call the function
trim_video(input_file, output_file, start_time, end_time)

# ROI coordinates: (347, 730, 423, 165)
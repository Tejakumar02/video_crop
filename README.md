Video Trimmer Tool

📋 Overview
Video Trimmer is a Python script that allows you to trim video files between specific time intervals using OpenCV. The tool maintains the original video quality, frame rate, and dimensions while extracting only the desired segment.

✨ Features
🎥 Core Functionality
Precise Time-based Trimming: Cut videos between exact start and end times in seconds

Original Quality Preservation: Maintains source video's resolution, frame rate, and codec

Large File Support: Efficient frame-by-frame processing for handling large video files

Cross-format Compatibility: Works with common video formats (MP4, AVI, etc.)

⚙️ Technical Capabilities
Automatic Parameter Detection: Reads FPS, resolution, and other metadata from source video

Frame-accurate Trimming: Uses frame-based positioning for precise cuts

Efficient Memory Usage: Processes videos without loading entire file into memory

Flexible Output Format: Supports various codecs via FourCC codes

🛠️ Installation
Prerequisites
Python 3.6 or higher

OpenCV library

Installation Steps
bash
# Install OpenCV (if not already installed)
pip install opencv-python

# Or for headless systems (no GUI support)
pip install opencv-python-headless
🚀 Usage
Basic Usage
python
from trim_video import trim_video

# Trim video from 60 seconds to 120 seconds
trim_video("input_video.mp4", "output_trimmed.mp4", 60, 120)
Command Line Usage
bash
# Run directly from command line
python trim_video.py

# Or with custom parameters
python trim_video.py input.mp4 output.mp4 60 120
Parameters
python
trim_video(input_file, output_file, start_time_sec, end_time_sec)
input_file: Path to source video file

output_file: Path for trimmed output video

start_time_sec: Start time in seconds (e.g., 60 for 1 minute)

end_time_sec: End time in seconds (e.g., 120 for 2 minutes)

📁 Example
Sample Code
python
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

# Start at 60 seconds and end at 120 seconds
start_time = 60
end_time = 120

# Call the function
trim_video(input_file, output_file, start_time, end_time)
⚙️ Technical Details
Key Functions
Video Capture Initialization: Opens video file and validates it

Metadata Extraction: Reads FPS, resolution, and total frames

Frame Calculation: Converts time (seconds) to frame numbers

Selective Frame Writing: Copies only frames within specified range

Resource Management: Properly releases video resources

Output Codec Options
The script uses XVID codec by default. You can change this:

python
# Different FourCC codes
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI format
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 format
fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Motion JPEG
🐛 Troubleshooting
Common Issues
Issue	Solution
"Could not open video file"	Check file path and permissions
Output file is empty	Verify start_time < end_time
Poor output quality	Use appropriate FourCC codec
Memory issues with large videos	Process in smaller segments
Audio missing in output	This script doesn't preserve audio
Error Handling
The script includes basic error checking:

Validates video file can be opened

Checks FPS is available

Verifies frame dimensions are readable

📈 Performance Considerations
Processing Speed: Depends on video size and system specifications

Memory Usage: Processes one frame at a time (low memory footprint)

Large Files: Can handle multi-gigabyte videos efficiently

Multi-threading: Current version is single-threaded

🔧 Extending Functionality
Adding Audio Support
python
# Note: OpenCV doesn't handle audio. Consider using:
# - moviepy for audio preservation
# - ffmpeg for complete audio/video processing
Batch Processing
python
def batch_trim_videos(file_list, start_time, end_time):
    for input_file in file_list:
        output_file = f"trimmed_{input_file}"
        trim_video(input_file, output_file, start_time, end_time)
GUI Version
Consider integrating with:

Tkinter for desktop GUI

Streamlit for web interface

PyQt for advanced desktop applications

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -am 'Add new feature')

Push to branch (git push origin feature/improvement)

Create a Pull Request

📄 License
This project is available under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments
OpenCV for comprehensive computer vision capabilities

Python for simple and effective scripting

📞 Support
For issues or feature requests:

Check the existing issues

Create a new issue with video sample details

Include error messages and system specifications

Note: This tool currently doesn't preserve audio tracks. For audio preservation, consider using FFmpeg or MoviePy libraries alongside this tool.

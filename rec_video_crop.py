import cv2

input_file = r"D:\rtsp_videos\rtsp_2025-11-26_13-27-47.mp4"
output_file = r"D:\rtsp_output\cropped_2025-11-26_13-27-47.mp4"

cap = cv2.VideoCapture(input_file)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
ret, frame = cap.read()
(h, w) = frame.shape[:2]

# Crop region:
x1, y1 = 550, 450
x2, y2 = 2000, 1500

crop_w = x2 - x1
crop_h = y2 - y1

out = cv2.VideoWriter(output_file, fourcc, 25, (crop_w, crop_h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    crop = frame[y1:y2, x1:x2]
    out.write(crop)

cap.release()
out.release()
cv2.destroyAllWindows()


#D:\labelimg\labelImg\labelImg.py "D:\Defect_Scanner\obj_train_data\obj_train_data" classes.txt"

#D:\labelimg\labelImg>python labelImg.py D:\Defect_Scanner\obj_train_data\obj_train_data classes.txt
import os

import cv2


def extract_frames(
    video_path: str = "./tmp.mov",
    output_folder: str = "./buffer_data",
    fps=10,  # frames per second
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"can't open video in '{video_path}'")
        return

    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps: {video_fps}, frame amount: {total_frames}")

    frame_interval = int(video_fps / fps)
    frame_count = 0
    extracted_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_name, frame)
            print(f"save frame '{frame_name}' to {output_folder}")
            extracted_count += 1

        frame_count += 1

    video_capture.release()

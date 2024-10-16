import os

import cv2


def make_animate(
    frames: list, animate_file_path: str = "./media", animate_name: str = "animate.mp4"
):
    video = cv2.VideoWriter(
        filename=os.path.join(animate_file_path, animate_name),
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=30,
        frameSize=frames[0].shape[:2],
    )

    for image in frames:
        video.write(
            cv2.rotate(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                cv2.ROTATE_90_CLOCKWISE,
            )
        )

    video.release()

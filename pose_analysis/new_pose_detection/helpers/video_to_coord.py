import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_CONNECTIONS = [
    # Upper body
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    # Torso
    (11, 23), (12, 24),
    (23, 24),

    # Legs
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]
LANDMARK_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]



def draw_pose_landmarks(image, landmarks, connections=POSE_CONNECTIONS):
    h, w, _ = image.shape

    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    info = {
        "cap": cap,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration,
    }

    return info


def print_video_info(info, process_first_n_seconds, process_every_n_frames):
    print("Video info:")
    print(f"  - FPS: {info['fps']}")
    print(f"  - Resolution: {info['width']}x{info['height']}")
    print(f"  - Total frames: {info['total_frames']}")
    print(f"  - Duration: {info['duration']:.2f} seconds")

    if process_first_n_seconds:
        max_frame = min(
            info["total_frames"],
            int(process_first_n_seconds * info["fps"])
        )
        print(f"  - Processing first {process_first_n_seconds} seconds only")
    else:
        max_frame = info["total_frames"]
        print("  - Processing full video")

    expected_processed = (
        max_frame + process_every_n_frames - 1
    ) // process_every_n_frames

    print(
        f"  - Processing every {process_every_n_frames} frames "
        f"(~{expected_processed} frames will be analyzed)"
    )

    return max_frame

def create_pose_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    return vision.PoseLandmarker.create_from_options(options)


def process_single_frame(frame, detector, frame_count, fps):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int(frame_count * (1000 / fps))
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        draw_pose_landmarks(frame, landmarks)
        return frame, landmarks

    return frame, None

def process_video(
    video_path,
    model_path,
    output_path,
    process_every_n_frames=1,
    process_first_n_seconds=None,
    skip_unprocessed_frames=True,
):
    info = get_video_info(video_path)
    cap = info["cap"]

    max_frame = print_video_info(
        info,
        process_first_n_seconds,
        process_every_n_frames
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        info["fps"],
        (info["width"], info["height"])
    )

    detected_landmarks = []

    frame_count = 0
    processed_frames = 0
    skipped_frames = 0

    print("\nStarting processing...")

    with create_pose_detector(model_path) as detector:
        while True:
            ret, frame = cap.read()

            if not ret or frame_count >= max_frame:
                break

            should_process = frame_count % process_every_n_frames == 0

            if should_process:
                frame, landmarks = process_single_frame(
                    frame,
                    detector,
                    frame_count,
                    info["fps"]
                )

                detected_landmarks.append({
                    "frame": frame_count,
                    "time_seconds": frame_count / info["fps"],
                    "landmarks": landmarks,  # can be None
                }) 

                processed_frames += 1

                if processed_frames % 30 == 0:
                    print(
                        f"  Processed {processed_frames} frames "
                        f"(frame #{frame_count})"
                    )

            else:
                skipped_frames += 1

                if skip_unprocessed_frames:
                    pass

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()

    print("\n✓ Processing complete!")
    print(f"  - Total frames in output: {frame_count}")
    print(f"  - Frames with pose detection attempted: {processed_frames}")
    print(f"  - Frames skipped: {skipped_frames}")
    print(f"  - Frames with detected pose: {len(detected_landmarks)}")
    print(f"  - Output video saved to: {output_path}")

    return detected_landmarks


def landmarks_to_dataframe(detected_landmarks, landmark_ids=LANDMARK_IDS):
    rows = []

    for item in detected_landmarks:
        row = {
            "Frame": item["frame"]
        }

        landmarks = item["landmarks"]

        if landmarks is None:
            for idx in landmark_ids:
                row[f"p{idx}"] = float("nan")
        else:
            for idx in landmark_ids:
                lm = landmarks[idx]
                row[f"p{idx}"] = f"({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f})"

        rows.append(row)

    return pd.DataFrame(rows)
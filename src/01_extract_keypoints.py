"""
01_extract_keypoints.py

Squat Dataset Publication Pipeline
----------------------------------

Purpose:
    Extract MediaPipe Pose Landmarker Full landmarks from every
    front-view and side-view squat video.

Input:
    data/raw/front/*.mp4
    data/raw/side/*.mp4

Output:
    data/processed/keypoints/front/*.csv
    data/processed/keypoints/side/*.csv

For every frame, the CSV stores:
    - video name
    - view
    - frame number
    - timestamp
    - FPS
    - resolution
    - landmark ID
    - landmark name
    - x
    - y
    - z
    - visibility
    - presence

The complete 33-landmark pose output is retained so that the
dataset can be reused for future biomechanical analyses.

Pose model:
    MediaPipe Pose Landmarker Full
"""

from pathlib import Path
import csv
import time

import cv2
import mediapipe as mp
from tqdm import tqdm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"

RAW_DIR = ROOT / "data" / "raw"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "keypoints"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


# ============================================================
# 2. MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# 3. MEDIA PIPE 33 LANDMARK NAMES
# ============================================================

LANDMARK_NAMES = [
    "nose",

    "left_eye_inner",
    "left_eye",
    "left_eye_outer",

    "right_eye_inner",
    "right_eye",
    "right_eye_outer",

    "left_ear",
    "right_ear",

    "mouth_left",
    "mouth_right",

    "left_shoulder",
    "right_shoulder",

    "left_elbow",
    "right_elbow",

    "left_wrist",
    "right_wrist",

    "left_pinky",
    "right_pinky",

    "left_index",
    "right_index",

    "left_thumb",
    "right_thumb",

    "left_hip",
    "right_hip",

    "left_knee",
    "right_knee",

    "left_ankle",
    "right_ankle",

    "left_heel",
    "right_heel",

    "left_foot_index",
    "right_foot_index",
]


# ============================================================
# 4. CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    for view in ["front", "side"]:

        output_view_dir = (
            OUTPUT_DIR / view
        )

        output_view_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 5. FIND VIDEOS
# ============================================================

def find_videos(view):

    directory = RAW_DIR / view

    if not directory.exists():

        print()
        print(
            f"[ERROR] Folder does not exist:"
        )
        print(directory)

        return []

    videos = []

    for file in directory.iterdir():

        if (
            file.is_file()
            and file.suffix.lower()
            in VIDEO_EXTENSIONS
        ):
            videos.append(file)

    return sorted(videos)


# ============================================================
# 6. CREATE MEDIAPIPE LANDMARKER
# ============================================================

def create_landmarker():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nMediaPipe model not found.\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Make sure pose_landmarker_full.task "
            "is inside the models folder."
        )

    options = PoseLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        ),

        running_mode=(
            VisionRunningMode.VIDEO
        ),

        # One participant per recording.
        num_poses=1,

        # Detection confidence.
        min_pose_detection_confidence=0.5,

        # Pose presence confidence.
        min_pose_presence_confidence=0.5,

        # Tracking confidence.
        min_tracking_confidence=0.5,
    )

    return PoseLandmarker.create_from_options(
        options
    )


# ============================================================
# 7. PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    view
):

    output_path = (
        OUTPUT_DIR
        / view
        / f"{video_path.stem}.csv"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print()
        print(
            f"[ERROR] Could not open video:"
        )
        print(video_path)

        return False

    # --------------------------------------------------------
    # Read video properties
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        print(
            "[WARNING] Invalid FPS detected. "
            "Using 30 FPS fallback."
        )

        fps = 30.0

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()
    print("=" * 72)

    print(
        f"Processing video : {video_path.name}"
    )

    print(
        f"View             : {view}"
    )

    print(
        f"Resolution       : {width} x {height}"
    )

    print(
        f"FPS              : {fps:.3f}"
    )

    print(
        f"Frames           : {frame_count}"
    )

    print(
        f"Duration         : {duration:.2f} seconds"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # CSV columns
    # --------------------------------------------------------

    fieldnames = [

        "video",

        "view",

        "frame",

        "timestamp_ms",

        "fps",

        "frame_width",

        "frame_height",

        "landmark_id",

        "landmark_name",

        "x",

        "y",

        "z",

        "visibility",

        "presence",
    ]

    # --------------------------------------------------------
    # Create a NEW MediaPipe landmarker for this video.
    #
    # This is important.
    #
    # Each video has its own timestamp sequence starting
    # from zero. Reusing one VIDEO-mode landmarker across
    # multiple videos can cause:
    #
    # "Input timestamp must be monotonically increasing"
    #
    # --------------------------------------------------------

    with create_landmarker() as landmarker:

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            # ------------------------------------------------
            # Frame/timestamp state
            # ------------------------------------------------

            frame_index = 0

            previous_timestamp_ms = -1

            # ------------------------------------------------
            # Progress bar
            # ------------------------------------------------

            progress = tqdm(
                total=frame_count,
                desc=video_path.name,
                unit="frame"
            )

            # ------------------------------------------------
            # Process frames
            # ------------------------------------------------

            while True:

                success, frame = (
                    cap.read()
                )

                if not success:

                    break

                # ------------------------------------------------
                # Generate timestamp.
                #
                # The timestamp is based on the video's actual
                # FPS and frame index.
                #
                # We additionally enforce strict monotonicity
                # because MediaPipe VIDEO mode requires it.
                # ------------------------------------------------

                timestamp_ms = int(
                    round(
                        (
                            frame_index
                            * 1000.0
                        )
                        / fps
                    )
                )

                if (
                    timestamp_ms
                    <= previous_timestamp_ms
                ):

                    timestamp_ms = (
                        previous_timestamp_ms
                        + 1
                    )

                previous_timestamp_ms = (
                    timestamp_ms
                )

                # ------------------------------------------------
                # Convert BGR → RGB
                # ------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # ------------------------------------------------
                # Create MediaPipe image
                # ------------------------------------------------

                mp_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb_frame
                )

                # ------------------------------------------------
                # Run pose detection
                # ------------------------------------------------

                try:

                    result = (
                        landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms
                        )
                    )

                    # ------------------------------------------------
                    # Pose detected
                    # ------------------------------------------------

                    if result.pose_landmarks:

                        landmarks = (
                            result.pose_landmarks[0]
                        )

                        for (
                            landmark_id,
                            landmark
                        ) in enumerate(
                            landmarks
                        ):

                            landmark_name = (
                                LANDMARK_NAMES[
                                    landmark_id
                                ]
                            )

                            # Normalized image coordinates.
                            x = landmark.x
                            y = landmark.y
                            z = landmark.z

                            visibility = getattr(
                                landmark,
                                "visibility",
                                ""
                            )

                            presence = getattr(
                                landmark,
                                "presence",
                                ""
                            )

                            writer.writerow({

                                "video":
                                    video_path.name,

                                "view":
                                    view,

                                "frame":
                                    frame_index,

                                "timestamp_ms":
                                    timestamp_ms,

                                "fps":
                                    fps,

                                "frame_width":
                                    width,

                                "frame_height":
                                    height,

                                "landmark_id":
                                    landmark_id,

                                "landmark_name":
                                    landmark_name,

                                "x":
                                    x,

                                "y":
                                    y,

                                "z":
                                    z,

                                "visibility":
                                    visibility,

                                "presence":
                                    presence,
                            })

                    # ------------------------------------------------
                    # No pose detected
                    #
                    # We preserve the frame in the CSV with empty
                    # landmark values instead of deleting the frame.
                    # ------------------------------------------------

                    else:

                        for (
                            landmark_id,
                            landmark_name
                        ) in enumerate(
                            LANDMARK_NAMES
                        ):

                            writer.writerow({

                                "video":
                                    video_path.name,

                                "view":
                                    view,

                                "frame":
                                    frame_index,

                                "timestamp_ms":
                                    timestamp_ms,

                                "fps":
                                    fps,

                                "frame_width":
                                    width,

                                "frame_height":
                                    height,

                                "landmark_id":
                                    landmark_id,

                                "landmark_name":
                                    landmark_name,

                                "x":
                                    "",

                                "y":
                                    "",

                                "z":
                                    "",

                                "visibility":
                                    "",

                                "presence":
                                    "",
                            })

                except Exception as exc:

                    print()

                    print(
                        f"[WARNING] "
                        f"Frame {frame_index} failed:"
                    )

                    print(exc)

                    # Preserve the frame even when inference
                    # fails unexpectedly.

                    for (
                        landmark_id,
                        landmark_name
                    ) in enumerate(
                        LANDMARK_NAMES
                    ):

                        writer.writerow({

                            "video":
                                video_path.name,

                            "view":
                                view,

                            "frame":
                                frame_index,

                            "timestamp_ms":
                                timestamp_ms,

                            "fps":
                                fps,

                            "frame_width":
                                width,

                            "frame_height":
                                height,

                            "landmark_id":
                                landmark_id,

                            "landmark_name":
                                landmark_name,

                            "x":
                                "",

                            "y":
                                "",

                            "z":
                                "",

                            "visibility":
                                "",

                            "presence":
                                "",
                        })

                frame_index += 1

                progress.update(1)

            progress.close()

    # --------------------------------------------------------
    # Close video
    # --------------------------------------------------------

    cap.release()

    print()

    print(
        f"[DONE] Keypoints saved:"
    )

    print(output_path)

    return True


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "SQUAT DATASET — KEYPOINT EXTRACTION"
    )
    print("=" * 72)

    print()
    print(
        f"Project root:"
    )
    print(ROOT)

    print()
    print(
        f"Pose model:"
    )
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print()

        print(
            "[ERROR] pose_landmarker_full.task "
            "was not found."
        )

        print(
            "Expected:"
        )

        print(MODEL_PATH)

        return

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    all_videos = []

    for view in ["front", "side"]:

        videos = find_videos(view)

        print()

        print(
            f"{view.upper():<10}: "
            f"{len(videos)} videos"
        )

        for video in videos:

            all_videos.append(
                (
                    video,
                    view
                )
            )

    # --------------------------------------------------------
    # No videos
    # --------------------------------------------------------

    if not all_videos:

        print()

        print(
            "[ERROR] No videos found."
        )

        print()

        print(
            "Expected folders:"
        )

        print(
            RAW_DIR / "front"
        )

        print(
            RAW_DIR / "side"
        )

        return

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        f"TOTAL VIDEOS: "
        f"{len(all_videos)}"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    start_time = time.time()

    successful = 0

    failed = 0

    for index, (
        video,
        view
    ) in enumerate(
        all_videos,
        start=1
    ):

        print()

        print(
            f"[VIDEO {index}/"
            f"{len(all_videos)}]"
        )

        try:

            success = process_video(
                video_path=video,
                view=view
            )

            if success:

                successful += 1

            else:

                failed += 1

        except KeyboardInterrupt:

            print()

            print(
                "[STOPPED] "
                "Extraction interrupted by user."
            )

            break

        except Exception as exc:

            failed += 1

            print()

            print(
                f"[ERROR] "
                f"{video.name}"
            )

            print(exc)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    elapsed = (
        time.time() - start_time
    )

    print()

    print("=" * 72)

    print(
        "POSE EXTRACTION COMPLETE"
    )

    print("=" * 72)

    print(
        f"Total videos : "
        f"{len(all_videos)}"
    )

    print(
        f"Successful   : "
        f"{successful}"
    )

    print(
        f"Failed       : "
        f"{failed}"
    )

    print(
        f"Time elapsed : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 72)


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
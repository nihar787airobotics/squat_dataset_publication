"""
03_calculate_angles.py

Squat Dataset Publication Pipeline
----------------------------------

Converts MediaPipe pose keypoints into biomechanical joint angles.

Angles calculated:
    1. Left knee
    2. Right knee
    3. Left hip
    4. Right hip
    5. Left ankle
    6. Right ankle
    7. Trunk angle

Input:
    data/processed/keypoints/front/*.csv
    data/processed/keypoints/side/*.csv

Output:
    data/processed/angles/front/*.csv
    data/processed/angles/side/*.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "keypoints"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "angles"
)


# ============================================================
# 2. LANDMARK HELPER
# ============================================================

def get_point(row, landmark_name):
    """
    Return x, y, z coordinates for a landmark.
    """

    landmark = row[
        row["landmark_name"] == landmark_name
    ]

    if landmark.empty:
        return None

    point = landmark.iloc[0]

    values = [
        point["x"],
        point["y"],
        point["z"]
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return None

    return np.array(
        values,
        dtype=float
    )


# ============================================================
# 3. ANGLE BETWEEN THREE POINTS
# ============================================================

def calculate_angle(point_a, point_b, point_c):
    """
    Calculate angle ABC in degrees.

    A ---- B ---- C

    The angle is measured at point B.
    """

    if (
        point_a is None
        or point_b is None
        or point_c is None
    ):
        return np.nan

    vector_ba = point_a - point_b

    vector_bc = point_c - point_b

    norm_ba = np.linalg.norm(
        vector_ba
    )

    norm_bc = np.linalg.norm(
        vector_bc
    )

    if (
        norm_ba == 0
        or norm_bc == 0
    ):
        return np.nan

    cosine_angle = (
        np.dot(
            vector_ba,
            vector_bc
        )
        /
        (
            norm_ba
            *
            norm_bc
        )
    )

    # Numerical protection.
    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(
            cosine_angle
        )
    )

    return angle


# ============================================================
# 4. TRUNK ANGLE
# ============================================================

def calculate_trunk_angle(row):
    """
    Calculate trunk inclination relative to the vertical axis.

    The trunk vector is defined from the midpoint of the hips
    to the midpoint of the shoulders.

    Angle:
        0 degrees   = upright
        larger angle = greater forward/lateral inclination

    The absolute inclination is reported.
    """

    left_shoulder = get_point(
        row,
        "left_shoulder"
    )

    right_shoulder = get_point(
        row,
        "right_shoulder"
    )

    left_hip = get_point(
        row,
        "left_hip"
    )

    right_hip = get_point(
        row,
        "right_hip"
    )

    if any(
        point is None
        for point in [
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip
        ]
    ):
        return np.nan

    shoulder_midpoint = (
        left_shoulder
        +
        right_shoulder
    ) / 2.0

    hip_midpoint = (
        left_hip
        +
        right_hip
    ) / 2.0

    trunk_vector = (
        shoulder_midpoint
        -
        hip_midpoint
    )

    # We use the image vertical axis.
    #
    # y increases downward in image coordinates.
    #
    # Therefore the vertical reference vector is:
    #
    # [0, -1, 0]
    #
    vertical_vector = np.array(
        [0.0, -1.0, 0.0]
    )

    trunk_norm = np.linalg.norm(
        trunk_vector
    )

    if trunk_norm == 0:
        return np.nan

    cosine_angle = (
        np.dot(
            trunk_vector,
            vertical_vector
        )
        /
        trunk_norm
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(
            cosine_angle
        )
    )

    return angle


# ============================================================
# 5. CALCULATE ALL ANGLES FOR ONE FRAME
# ============================================================

def calculate_frame_angles(frame_data):
    """
    Calculate all required angles for one frame.
    """

    # --------------------------------------------------------
    # LEFT LEG
    # --------------------------------------------------------

    left_hip = get_point(
        frame_data,
        "left_hip"
    )

    left_knee = get_point(
        frame_data,
        "left_knee"
    )

    left_ankle = get_point(
        frame_data,
        "left_ankle"
    )

    left_foot = get_point(
        frame_data,
        "left_foot_index"
    )

    left_shoulder = get_point(
        frame_data,
        "left_shoulder"
    )

    # --------------------------------------------------------
    # RIGHT LEG
    # --------------------------------------------------------

    right_hip = get_point(
        frame_data,
        "right_hip"
    )

    right_knee = get_point(
        frame_data,
        "right_knee"
    )

    right_ankle = get_point(
        frame_data,
        "right_ankle"
    )

    right_foot = get_point(
        frame_data,
        "right_foot_index"
    )

    right_shoulder = get_point(
        frame_data,
        "right_shoulder"
    )

    # --------------------------------------------------------
    # LEFT KNEE
    #
    # Hip -> Knee -> Ankle
    # --------------------------------------------------------

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    # --------------------------------------------------------
    # RIGHT KNEE
    #
    # Hip -> Knee -> Ankle
    # --------------------------------------------------------

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    # --------------------------------------------------------
    # LEFT HIP
    #
    # Shoulder -> Hip -> Knee
    # --------------------------------------------------------

    left_hip_angle = calculate_angle(
        left_shoulder,
        left_hip,
        left_knee
    )

    # --------------------------------------------------------
    # RIGHT HIP
    #
    # Shoulder -> Hip -> Knee
    # --------------------------------------------------------

    right_hip_angle = calculate_angle(
        right_shoulder,
        right_hip,
        right_knee
    )

    # --------------------------------------------------------
    # LEFT ANKLE
    #
    # Knee -> Ankle -> Foot
    # --------------------------------------------------------

    left_ankle_angle = calculate_angle(
        left_knee,
        left_ankle,
        left_foot
    )

    # --------------------------------------------------------
    # RIGHT ANKLE
    #
    # Knee -> Ankle -> Foot
    # --------------------------------------------------------

    right_ankle_angle = calculate_angle(
        right_knee,
        right_ankle,
        right_foot
    )

    # --------------------------------------------------------
    # TRUNK
    # --------------------------------------------------------

    trunk_angle = calculate_trunk_angle(
        frame_data
    )

    return {
        "left_knee_angle_deg":
            left_knee_angle,

        "right_knee_angle_deg":
            right_knee_angle,

        "left_hip_angle_deg":
            left_hip_angle,

        "right_hip_angle_deg":
            right_hip_angle,

        "left_ankle_angle_deg":
            left_ankle_angle,

        "right_ankle_angle_deg":
            right_ankle_angle,

        "trunk_angle_deg":
            trunk_angle,
    }


# ============================================================
# 6. PROCESS ONE KEYPOINT CSV
# ============================================================

def process_file(
    input_file,
    output_file
):

    print()
    print(
        f"Processing: {input_file.name}"
    )

    df = pd.read_csv(
        input_file
    )

    if df.empty:

        print(
            "[WARNING] Empty CSV."
        )

        return

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "video",
        "view",
        "frame",
        "timestamp_ms",
        "fps",
        "frame_width",
        "frame_height",
        "landmark_name",
        "x",
        "y",
        "z",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # Calculate angles frame by frame
    # --------------------------------------------------------

    results = []

    grouped = df.groupby(
        "frame",
        sort=True
    )

    for frame_number, frame_data in grouped:

        first_row = frame_data.iloc[0]

        angle_data = (
            calculate_frame_angles(
                frame_data
            )
        )

        result = {

            "video":
                first_row["video"],

            "view":
                first_row["view"],

            "frame":
                frame_number,

            "timestamp_ms":
                first_row["timestamp_ms"],

            "fps":
                first_row["fps"],

            "frame_width":
                first_row["frame_width"],

            "frame_height":
                first_row["frame_height"],

        }

        result.update(
            angle_data
        )

        results.append(
            result
        )

    angle_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    angle_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"[DONE] Saved: {output_file}"
    )


# ============================================================
# 7. PROCESS ALL VIDEOS
# ============================================================

def process_view(view):

    input_view_dir = (
        INPUT_DIR / view
    )

    output_view_dir = (
        OUTPUT_DIR / view
    )

    output_view_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_view_dir.exists():

        print()
        print(
            f"[WARNING] "
            f"Input folder does not exist:"
        )

        print(
            input_view_dir
        )

        return

    csv_files = sorted(
        input_view_dir.glob(
            "*.csv"
        )
    )

    print()
    print("=" * 70)

    print(
        f"{view.upper()} VIEW"
    )

    print(
        f"Files found: "
        f"{len(csv_files)}"
    )

    print("=" * 70)

    for input_file in tqdm(
        csv_files,
        desc=f"{view} angle extraction"
    ):

        output_file = (
            output_view_dir
            / input_file.name
        )

        process_file(
            input_file,
            output_file
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SQUAT DATASET — JOINT ANGLE EXTRACTION"
    )
    print("=" * 70)

    print()
    print(
        "Angles:"
    )

    print(
        "  • Left knee"
    )

    print(
        "  • Right knee"
    )

    print(
        "  • Left hip"
    )

    print(
        "  • Right hip"
    )

    print(
        "  • Left ankle"
    )

    print(
        "  • Right ankle"
    )

    print(
        "  • Trunk"
    )

    process_view(
        "front"
    )

    process_view(
        "side"
    )

    print()
    print("=" * 70)

    print(
        "JOINT ANGLE EXTRACTION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()
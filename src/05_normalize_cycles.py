"""
05_normalize_cycles.py

Squat Dataset Publication Pipeline
----------------------------------

Purpose:
    Normalize every detected squat repetition to a common
    0-100% movement cycle.

Why:
    Different participants perform a squat at different
    speeds and therefore have different numbers of frames.

    Temporal normalization allows the repetitions to be
    compared on the same movement-cycle axis.

Input:
    data/processed/angles/front/*.csv
    data/processed/angles/side/*.csv

    data/processed/segments/front/*_repetitions.csv
    data/processed/segments/side/*_repetitions.csv

Output:
    data/processed/normalized/front/*.csv
    data/processed/normalized/side/*.csv

Each repetition contains 101 points:

    0%   = start of squat
    50%  = approximately squat bottom
    100% = end of squat

Angles retained:
    Left knee
    Right knee
    Left hip
    Right hip
    Left ankle
    Right ankle
    Trunk
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ANGLE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "angles"
)

SEGMENT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "segments"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "normalized"
)


# ============================================================
# 2. NORMALIZATION SETTINGS
# ============================================================

# 101 points means:
#
# 0, 1, 2, ... 99, 100 %
#
# This is convenient for publication graphs.

NORMALIZED_POINTS = 101


# ============================================================
# 3. ANGLE COLUMNS
# ============================================================

ANGLE_COLUMNS = [

    "left_knee_angle_deg",

    "right_knee_angle_deg",

    "left_hip_angle_deg",

    "right_hip_angle_deg",

    "left_ankle_angle_deg",

    "right_ankle_angle_deg",

    "trunk_angle_deg",
]


# ============================================================
# 4. INTERPOLATION FUNCTION
# ============================================================

def interpolate_signal(
    x,
    y,
    new_x
):
    """
    Interpolate an angle signal onto a normalized 0-1 axis.

    Missing values are removed before interpolation.

    If insufficient valid values exist, NaN values are returned.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    new_x = np.asarray(
        new_x,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x_valid = x[valid]

    y_valid = y[valid]

    if len(x_valid) < 2:

        return np.full(
            len(new_x),
            np.nan
        )

    # Remove duplicate x positions.
    unique_x, unique_indices = (
        np.unique(
            x_valid,
            return_index=True
        )
    )

    unique_y = y_valid[
        unique_indices
    ]

    if len(unique_x) < 2:

        return np.full(
            len(new_x),
            np.nan
        )

    return np.interp(
        new_x,
        unique_x,
        unique_y
    )


# ============================================================
# 5. NORMALIZE ONE REPETITION
# ============================================================

def normalize_repetition(
    angle_df,
    start_frame,
    end_frame,
    rep_id
):
    """
    Extract one repetition and normalize it to 101 points.
    """

    # --------------------------------------------------------
    # Select frames belonging to this repetition.
    # --------------------------------------------------------

    rep_df = angle_df[
        (
            angle_df["frame"]
            >= start_frame
        )
        &
        (
            angle_df["frame"]
            <= end_frame
        )
    ].copy()

    if len(rep_df) < 3:

        return None

    rep_df = rep_df.sort_values(
        "frame"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Original normalized time axis.
    #
    # First frame = 0
    # Last frame  = 1
    # --------------------------------------------------------

    frame_values = (
        rep_df["frame"]
        .to_numpy(
            dtype=float
        )
    )

    frame_min = frame_values[0]

    frame_max = frame_values[-1]

    if frame_max <= frame_min:

        return None

    original_x = (
        frame_values
        - frame_min
    ) / (
        frame_max
        - frame_min
    )

    # --------------------------------------------------------
    # New 0-100% axis.
    # --------------------------------------------------------

    normalized_percent = np.linspace(
        0.0,
        100.0,
        NORMALIZED_POINTS
    )

    normalized_x = (
        normalized_percent
        / 100.0
    )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    result = pd.DataFrame({

        "rep_id":
            rep_id,

        "cycle_percent":
            normalized_percent,
    })

    # --------------------------------------------------------
    # Interpolate all angle signals.
    # --------------------------------------------------------

    for column in ANGLE_COLUMNS:

        if column not in rep_df.columns:

            result[column] = np.nan

            continue

        signal = pd.to_numeric(
            rep_df[column],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        result[column] = (
            interpolate_signal(
                original_x,
                signal,
                normalized_x
            )
        )

    return result


# ============================================================
# 6. PROCESS ONE PARTICIPANT
# ============================================================

def process_participant(
    angle_file,
    repetition_file,
    view
):
    """
    Normalize all detected repetitions for one participant.
    """

    angle_df = pd.read_csv(
        angle_file
    )

    repetitions = pd.read_csv(
        repetition_file
    )

    if angle_df.empty:

        print(
            f"[WARNING] Empty angle file: "
            f"{angle_file.name}"
        )

        return None

    if repetitions.empty:

        print(
            f"[WARNING] No repetitions: "
            f"{repetition_file.name}"
        )

        return None

    normalized_repetitions = []

    # --------------------------------------------------------
    # Process each detected repetition.
    # --------------------------------------------------------

    for _, rep in repetitions.iterrows():

        rep_id = int(
            rep["rep_id"]
        )

        start_frame = int(
            rep["start_frame"]
        )

        end_frame = int(
            rep["end_frame"]
        )

        normalized = normalize_repetition(

            angle_df,

            start_frame,

            end_frame,

            rep_id
        )

        if normalized is None:

            continue

        normalized.insert(
            0,
            "participant",
            angle_file.stem
        )

        normalized.insert(
            1,
            "view",
            view
        )

        normalized_repetitions.append(
            normalized
        )

    if not normalized_repetitions:

        print(
            f"[WARNING] "
            f"No repetitions normalized: "
            f"{angle_file.name}"
        )

        return None

    participant_df = pd.concat(
        normalized_repetitions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Output folder.
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_DIR / view
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save participant-level normalized data.
    # --------------------------------------------------------

    output_file = (
        output_dir
        / f"{angle_file.stem}_normalized.csv"
    )

    participant_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"[DONE] {angle_file.stem}: "
        f"{len(normalized_repetitions)} repetitions"
    )

    return participant_df


# ============================================================
# 7. PROCESS ONE VIEW
# ============================================================

def process_view(view):

    angle_dir = (
        ANGLE_DIR / view
    )

    segment_dir = (
        SEGMENT_DIR / view
    )

    output_dir = (
        OUTPUT_DIR / view
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not angle_dir.exists():

        print(
            f"[WARNING] Missing angle directory:"
        )

        print(
            angle_dir
        )

        return

    if not segment_dir.exists():

        print(
            f"[WARNING] Missing segment directory:"
        )

        print(
            segment_dir
        )

        return

    # --------------------------------------------------------
    # Find angle files.
    # --------------------------------------------------------

    angle_files = sorted(
        angle_dir.glob(
            "*.csv"
        )
    )

    print()
    print("=" * 70)

    print(
        f"{view.upper()} VIEW"
    )

    print(
        f"Angle files: "
        f"{len(angle_files)}"
    )

    print("=" * 70)

    all_participants = []

    # --------------------------------------------------------
    # Process participants.
    # --------------------------------------------------------

    for angle_file in tqdm(
        angle_files,
        desc=f"{view} normalization"
    ):

        # ----------------------------------------------------
        # Matching repetition file.
        #
        # Front:
        #     person_1.csv
        #     person_1_repetitions.csv
        #
        # Side:
        #     side_person1.csv
        #     side_person1_repetitions.csv
        # ----------------------------------------------------

        repetition_file = (
            segment_dir
            / f"{angle_file.stem}_repetitions.csv"
        )

        if not repetition_file.exists():

            print()

            print(
                f"[WARNING] "
                f"Missing repetition file:"
            )

            print(
                repetition_file
            )

            continue

        participant_df = (
            process_participant(
                angle_file,
                repetition_file,
                view
            )
        )

        if participant_df is not None:

            all_participants.append(
                participant_df
            )

    # --------------------------------------------------------
    # Combined view dataset.
    # --------------------------------------------------------

    if all_participants:

        combined = pd.concat(
            all_participants,
            ignore_index=True
        )

        combined_file = (
            output_dir
            / "all_normalized_repetitions.csv"
        )

        combined.to_csv(
            combined_file,
            index=False
        )

        print()

        print(
            f"[DONE] Combined dataset:"
        )

        print(
            combined_file
        )

        print(
            f"Rows: {len(combined)}"
        )

        print(
            f"Repetitions: "
            f"{combined['rep_id'].nunique()}"
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "SQUAT DATASET — CYCLE NORMALIZATION"
    )

    print("=" * 70)

    print()

    print(
        "Normalization:"
    )

    print(
        "0% → squat start"
    )

    print(
        "50% → approximately squat bottom"
    )

    print(
        "100% → squat end"
    )

    print()

    print(
        f"Points per repetition: "
        f"{NORMALIZED_POINTS}"
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
        "CYCLE NORMALIZATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "Output:"
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